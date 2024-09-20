#!/usr/bin/env python3
"""
reloader.py - A simple script reloader
inspired by jurigged[develoop] and watchdog
"""

from __future__ import annotations

import argparse
import ast
import code as interactive_code
import importlib.util
import inspect
import linecache
import os
import sys
import threading
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager, suppress
from ctypes import c_long, py_object, pythonapi
from importlib.abc import MetaPathFinder
from os import PathLike
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import sleep
from types import CodeType, MemberDescriptorType, ModuleType
from typing import Any, Callable, Iterator, Type, TypeVar, cast

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.utils.event_debouncer import EventDebouncer

__author__ = "EcmaXp"
__version__ = "0.15.1"
__license__ = "MIT"
__url__ = "https://pypi.org/project/reloader.py/"
__all__ = [
    "Interrupted",
    "Watcher",
    "Reloader",
    "CodeLoopReloader",
    "CodeDaemonReloader",
    "ScriptModuleLoopReloader",
    "ScriptModuleDaemonReloader",
    "ScriptFuncLoopReloader",
    "ScriptFuncDaemonReloader",
]


class Interrupted(BaseException):
    pass


T = TypeVar("T")

DEFAULT_DEBOUNCE_INTERVAL = 0.1

REPLACEABLE_AST_TYPES = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
REPLACEABLE_OBJ_TYPES = type | Callable


class ReloadUnsupportedWarning(UserWarning):
    pass


def create_module(name: str, file: str) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = file
    module.__cached__ = None
    module.__package__ = None
    return module


def print_exc(e):
    limit = 0
    for tb_frame, _ in reversed(list(traceback.walk_tb(e.__traceback__))):
        if tb_frame.f_globals.get("__file__") == __file__:
            break

        limit -= 1

    traceback.print_exc(limit=limit)


class FileSystemEventEmitter(FileSystemEventHandler):
    def __init__(
        self,
        callback: Callable[[FileSystemEvent], None],
        observer: Observer = None,
    ):
        self.observer = Observer() if observer is None else observer
        self.files = set()
        self.parents = set()
        self.callback = callback

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.src_path in self.files:
            self.callback(event)

    def schedule(self, path: Path | PathLike | str):
        path = Path(path).resolve()
        self.files.add(str(path))
        parent = str(path.parent)
        if parent not in self.parents:
            self.parents.add(parent)
            self.observer.schedule(self, parent)


class CodeChunk:
    def __init__(self, file: str, text: str, stmt: ast.stmt):
        self.file = file
        self.text = text
        self.stmt = stmt
        self.code = compile(ast.Module(body=[self.stmt], type_ignores=[]), file, "exec")
        self.is_main = self.text.startswith("if __name__")

    def __hash__(self):
        return hash((self.file, self.text))

    def __eq__(self, other):
        if isinstance(other, CodeChunk):
            return self.file == other.file and self.text == other.text

        return NotImplemented

    def exec(self, module_globals: dict):
        exec(self.code, module_globals, module_globals)

    @staticmethod
    def get_linenos(stmt: ast.stmt) -> tuple[int, int]:
        lineno = float("inf")
        end_lineno = float("-inf")

        for child in ast.walk(stmt):
            if hasattr(child, "lineno"):
                lineno = min(lineno, child.lineno)
            if hasattr(child, "end_lineno"):
                end_lineno = max(end_lineno, child.end_lineno)

        return lineno, end_lineno

    def replace(self, new_code: str) -> CodeChunk:
        if not isinstance(self.stmt, REPLACEABLE_AST_TYPES):
            raise ValueError("Only functions and classes are replaceable")

        new_stmts = ast.parse(new_code).body
        if len(new_stmts) != 1:
            raise ValueError("Only one function or class can be replaced at a time")

        old_stmt = cast(REPLACEABLE_AST_TYPES, self.stmt)
        new_stmt = new_stmts[0]
        if not isinstance(new_stmt, REPLACEABLE_AST_TYPES):
            raise ValueError("Only functions and classes are replaceable")
        if type(old_stmt) != type(new_stmt):
            raise ValueError("New code must be the same type as the old code")

        new_stmt = cast(REPLACEABLE_AST_TYPES, new_stmt)
        if old_stmt.name != new_stmt.name:
            raise ValueError("New code must have the same name as the old code")

        return type(self)(self.file, new_code, new_stmt)


class ModulePatcher:
    def exec(self, chunk: CodeChunk, module_globals: dict):
        if self.check_patch_required(chunk):
            with self.patch(module_globals):
                chunk.exec(module_globals)
        else:
            chunk.exec(module_globals)

    @staticmethod
    def check_patch_required(chunk: CodeChunk):
        return not isinstance(chunk.stmt, (ast.Import, ast.ImportFrom))

    @contextmanager
    def patch(self, module_globals: dict):
        old_globals = module_globals.copy()
        try:
            yield
        finally:
            self.patch_module(old_globals, module_globals, visited=set())

    def patch_module(self, old_globals: dict, new_globals: dict, *, visited: set[int]):
        for key, new_value in new_globals.items():
            old_value = old_globals.get(key)
            if old_value is not new_value:
                new_globals[key] = self.patch_object(
                    old_value,
                    new_value,
                    visited=visited,
                )

    def patch_object(self, old_value: Any, new_value: Any, *, visited: set[int]):
        if isinstance(new_value, MemberDescriptorType):
            warnings.warn(
                "MemberDescriptor is not supported",
                ReloadUnsupportedWarning,
            )
            return old_value
        elif not self.check_object(old_value, new_value):
            return new_value
        elif isinstance(old_value, type) and isinstance(new_value, type):
            return self.patch_class(old_value, new_value, visited=visited)
        elif callable(old_value) and callable(new_value):
            return self.patch_callable(old_value, new_value, visited=visited)
        else:
            return new_value

    @staticmethod
    def check_object(old_value: Any, new_value: Any):
        old_module = getattr(old_value, "__module__", None)
        new_module = getattr(new_value, "__module__", None)
        return old_module == new_module

    def patch_class(self, old_class: type, new_class: type, *, visited: set[int]):
        self.patch_vars(old_class, new_class, visited=visited)
        return old_class

    def patch_callable(
        self, old_callable: Callable, new_callable: Callable, *, visited: set[int]
    ):
        self.patch_vars(old_callable, new_callable, visited=visited)

        old_func = inspect.unwrap(old_callable)
        new_func = inspect.unwrap(new_callable)

        for name in (
            "__code__",
            "__doc__",
            "__annotations__",
            "__kwdefaults__",
            "__defaults__",
        ):
            try:
                if hasattr(new_callable, name):
                    setattr(old_callable, name, getattr(new_callable, name))
                elif hasattr(new_func, name):
                    setattr(old_func, name, getattr(new_func, name))
            except ValueError as e:
                if name == "__code__":
                    warnings.warn(
                        f"Function code mismatch: {e}",
                        ReloadUnsupportedWarning,
                    )
                    return new_callable

                raise

        return old_callable

    def patch_vars(self, old_obj, new_obj, *, visited: set[int]):
        if id(old_obj) in visited:
            return

        visited.add(id(old_obj))
        old_vars = vars(old_obj)
        for key, new_value in vars(new_obj).items():
            old_value = old_vars.get(key)
            if key == "__dict__":
                continue

            setattr(
                old_obj,
                key,
                self.patch_object(old_value, new_value, visited=visited),
            )


class CodeModule:
    def __init__(self, module: ModuleType):
        self.module = module
        try:
            self.chunks = self.load_chunks()
        except SyntaxError:
            self.chunks = []
        self.executed = Counter(self.chunks)
        self.patcher = ModulePatcher()

    def __repr__(self):
        return f"{type(self).__name__}({self.module!r})"

    @property
    def name(self):
        return self.module.__name__

    @property
    def file(self):
        return inspect.getfile(self.module)

    @property
    def path(self):
        return Path(self.file)

    def load_lines(self) -> list[str]:
        linecache.updatecache(self.file, self.module.__dict__)

        lines = None
        if self.module.__loader__:
            with suppress(OSError):
                lines = inspect.getsourcelines(self.module)[0][:]

        if lines is None:
            lines = self.path.read_text().splitlines(keepends=True)

        return lines

    def load_chunks(self) -> list[str | CodeChunk]:
        lines = self.load_lines()
        tree = ast.parse("".join(lines), self.file)

        for stmt in reversed(tree.body):
            line, end_line = CodeChunk.get_linenos(stmt)
            code_chunk = CodeChunk(self.file, "".join(lines[line - 1 : end_line]), stmt)
            lines[line - 1 : end_line] = [code_chunk]

        return lines

    def iter_reloadable_code_chunks(
        self, *, with_main: bool = False
    ) -> Iterator[CodeChunk]:
        chunks = self.chunks = self.load_chunks()
        counter = Counter(chunks)

        for chunk in self.executed | counter:
            count = min(self.executed[chunk], counter[chunk])
            if count > 0:
                self.executed[chunk] = count
            else:
                self.executed.pop(chunk, None)

        for chunk in chunks:
            if isinstance(chunk, CodeChunk):
                if chunk.is_main:
                    if with_main:
                        yield chunk
                elif counter[chunk] > self.executed[chunk]:
                    yield chunk

    def reload(self, *, with_main: bool = False):
        for code_chunk in self.iter_reloadable_code_chunks(with_main=with_main):
            self.reload_code_chunk(code_chunk)

    def reload_code_chunk(self, chunk: CodeChunk):
        self.patcher.exec(chunk, self.module.__dict__)
        self.executed[chunk] += 1

    def get_code_chunk(self, obj: REPLACEABLE_OBJ_TYPES) -> CodeChunk:
        for chunk in self.chunks:
            if isinstance(chunk, CodeChunk) and isinstance(
                chunk.stmt, REPLACEABLE_AST_TYPES
            ):
                chunk_stmt = cast(REPLACEABLE_AST_TYPES, chunk.stmt)
                if chunk_stmt.name == obj.__name__:
                    return chunk

        raise ValueError(f"Object {obj} not found")

    def replace_code_chunk(
        self, old: CodeChunk | REPLACEABLE_OBJ_TYPES, new_code: str
    ) -> CodeChunk:
        if isinstance(old, REPLACEABLE_OBJ_TYPES):
            old = self.get_code_chunk(old)

        new_chunk = old.replace(new_code.rstrip("\n") + "\n")
        self.chunks[self.chunks.index(old)] = new_chunk
        self.executed.pop(old, None)
        self.reload_code_chunk(new_chunk)
        self.executed[new_chunk] += 1
        return new_chunk

    def write(self):
        with self.path.open("w") as file:
            for chunk in self.chunks:
                if isinstance(chunk, CodeChunk):
                    file.write(chunk.text)
                else:
                    file.write(chunk)


class ScriptModule(CodeModule):
    def __init__(self, script_path: Path | PathLike):
        script_path = Path(script_path).resolve()
        module = create_module("__main__", str(script_path))
        super().__init__(module)
        self.executed.clear()

    def install_as_main_module(self):
        sys.modules[self.name] = self.module

    def run(self, *, with_main: bool = True):
        self.reload(with_main=with_main)


class SysModulesWatcher(MetaPathFinder, Thread):
    def __init__(self, callback: Callable[[ModuleType], None]):
        super().__init__(name=type(self).__name__, daemon=True)
        self.callback = callback
        self.running = None
        self.found = set()
        self.event = Event()

    def start(self):
        self.watch_all()
        sys.meta_path.insert(0, self)
        super().start()

    def find_spec(self, fullname, path=None, target=None):
        self.event.set()
        return None

    def run(self):
        self.running = True
        while self.running:
            sleep(0.1)
            self.watch_all()
            self.event.wait()
            self.event.clear()

    def stop(self):
        self.running = False
        self.event.set()

        if self in sys.meta_path:
            sys.meta_path.remove(self)

    def watch_all(self):
        modified_names = sys.modules.keys() - self.found
        if not modified_names:
            return

        for module_name in reversed(list(sys.modules)):
            if module_name not in modified_names:
                continue

            module = sys.modules[module_name]
            self.callback(module)
            self.found.add(module_name)


class Watcher:
    def __init__(self, *, debounce_interval: float = DEFAULT_DEBOUNCE_INTERVAL):
        self.handlers: list[Callable[[list[CodeModule | Path]], None]] = []
        self._code_modules: dict[str, CodeModule | ScriptModule] = {}
        self._resources: set[str] = set()
        self._interrupt = None
        self._watchdog_observer = Observer()
        self._watchdog_debouncer = EventDebouncer(
            debounce_interval_seconds=debounce_interval or sys.float_info.min,
            events_callback=self._events_callback,
        )
        self._watchdog_handler = FileSystemEventEmitter(
            callback=self._watchdog_debouncer.handle_event,
            observer=self._watchdog_observer,
        )
        self._sys_modules_watcher = SysModulesWatcher(self._module_callback)

    def _get_threads(self):
        return (
            self._watchdog_observer,
            self._watchdog_debouncer,
            self._sys_modules_watcher,
        )

    def _events_callback(self, events: list[FileSystemEvent]):
        new_events = []
        for event in events:
            if isinstance(event, FileSystemEvent):
                if event.event_type not in ("modified", "created"):
                    continue

                code_module = self._code_modules.get(event.src_path)
                if code_module:
                    new_events.append(code_module)

                if event.src_path in self._resources:
                    new_events.append(Path(event.src_path))

        if new_events:
            for handler in self.handlers:
                handler(new_events)

    def _module_callback(self, module: ModuleType):
        if self._check_module(module):
            self.watch_module(module)

    def get_code_module(self, module_name: str) -> CodeModule | ScriptModule | None:
        for code_module in self._code_modules.values():
            if code_module.name == module_name:
                return code_module

        module = sys.modules.get(module_name)
        if module is not None:
            return self.watch_module(module)

        raise ValueError(f"Module {module_name!r} not found")

    @staticmethod
    def _check_module(module: ModuleType):
        try:
            inspect.getsource(module)
            return True
        except (OSError, TypeError):
            return False

    def watch_module(self, module: ModuleType) -> CodeModule:
        code_module = self._code_modules.get(inspect.getfile(module))
        if code_module is None:
            code_module = CodeModule(module)
            self.watch_code_module(code_module)

        return code_module

    def watch_script(self, script_path: Path | PathLike) -> ScriptModule:
        script_module = ScriptModule(script_path)
        script_module.install_as_main_module()
        self.watch_code_module(script_module)
        return script_module

    def watch_code_module(self, code_module: CodeModule) -> None:
        self._code_modules[code_module.file] = code_module
        self._watchdog_handler.schedule(code_module.file)

    def watch_resource(self, path: Path | PathLike | str) -> None:
        path = Path(path).resolve()
        self._resources.add(str(path))
        self._watchdog_handler.schedule(path)

    def start(self):
        for thread in self._get_threads():
            thread.start()

    def stop(self):
        for thread in self._get_threads():
            if thread.is_alive():
                thread.stop()
                thread.join()


class Reloader:
    def __init__(self, *, watcher: Watcher):
        self.watcher = watcher
        self._queue = Queue()
        self._running = False
        self._interruptable_ident = None

    def __enter__(self):
        self._running = True
        self.watcher.handlers.append(self._events_callback)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        self.watcher.handlers.remove(self._events_callback)
        return None

    def run(self):
        with self:
            while self._running:
                self._tick()

    def stop(self):
        self._running = False
        self._interrupt()

    def _events_callback(self, events: list[CodeModule | Path]):
        if self._running:
            self._queue.put(events)
            self._interrupt()

    def _tick(self):
        try:
            events = self._queue.get(timeout=0.1)
        except Empty:
            return

        self._before_reload()

        code_modules = {
            code_module for code_module in events if isinstance(code_module, CodeModule)
        }

        for code_module in code_modules:
            self.reload(code_module)

        self._after_reload()
        self._queue.task_done()

    def reload(
        self,
        code_module: CodeModule | ScriptModule,
        is_main_script: bool = False,
    ):
        if is_main_script:
            self._interruptable_ident = threading.get_ident()

        try:
            code_module.reload(with_main=is_main_script)
        except Interrupted:
            pass
        except Exception as e:  # noqa
            print_exc(e)
        finally:
            if is_main_script:
                self._interruptable_ident = None

    def _interrupt(self):
        if self._interruptable_ident is not None:
            pythonapi.PyThreadState_SetAsyncExc(
                c_long(self._interruptable_ident),
                py_object(Interrupted),
            )

    def _before_reload(self):
        pass

    def _after_reload(self):
        pass


class ExecReloader(Reloader):
    def exec(self):
        raise NotImplementedError


class LoopReloader(ExecReloader):
    def run(self):
        self._queue.put([])
        super().run()

    def _after_reload(self):
        try:
            self.exec()
        except Exception as e:  # noqa
            print_exc(e)


class DaemonReloader(ExecReloader):
    def __init__(self, *, watcher: Watcher):
        super().__init__(watcher=watcher)
        self.thread = Thread(target=super().run, name=type(self).__name__, daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        self.start()
        try:
            self._before_reload()
            try:
                self.exec()
            except Exception as e:  # noqa
                print_exc(e)
            self._after_reload()
        finally:
            self.stop()
            self.join()

    def join(self):
        self.thread.join()


class ScriptModuleReloader(Reloader):
    def __init__(self, script_module: ScriptModule, *, watcher: Watcher):
        super().__init__(watcher=watcher)
        self.script_module = script_module

    @property
    def global_dict(self) -> dict[str, Any]:
        return self.script_module.module.__dict__ if self.script_module else None


class ScriptModuleLoopReloader(ScriptModuleReloader, LoopReloader):
    def exec(self):
        return self.reload(self.script_module, is_main_script=True)


class ScriptModuleDaemonReloader(ScriptModuleReloader, DaemonReloader):
    def exec(self):
        return self.script_module.run(with_main=True)


class ScriptFuncReloader(ScriptModuleReloader):
    def __init__(
        self,
        script_module: ScriptModule,
        module_name: str,
        func_name: str,
        *,
        watcher: Watcher,
    ):
        super().__init__(script_module, watcher=watcher)
        self.module_name = module_name
        self.func_name = func_name

    def exec(self):
        self.script_module.run(with_main=False)
        try:
            func = self.global_dict[self.func_name]
        except KeyError:
            raise ImportError(
                f"cannot import name {self.func_name!r} from {self.module_name!r}"
            ) from None

        return func()


class ScriptFuncLoopReloader(ScriptFuncReloader, ScriptModuleLoopReloader):
    pass


class ScriptFuncDaemonReloader(ScriptFuncReloader, ScriptModuleDaemonReloader):
    pass


class CodeReloader(Reloader):
    def __init__(self, code: CodeType, *, watcher: Watcher):
        super().__init__(watcher=watcher)
        self.code = code
        self.module = create_module("__main__", code.co_name)

    @property
    def global_dict(self) -> dict[str, Any]:
        return self.module.__dict__

    def exec(self):
        return eval(self.code, self.global_dict, self.global_dict)


class CodeLoopReloader(CodeReloader, LoopReloader):
    pass


class CodeDaemonReloader(CodeReloader, DaemonReloader):
    pass


class InteractiveReloader(DaemonReloader):
    def __init__(self, global_dict: dict, *, watcher: Watcher):
        super().__init__(watcher=watcher)
        self.global_dict = global_dict

    def exec(self):
        interactive_code.interact(local=self.global_dict)


def get_reloader_cls(reloader_cls: Type[T], loop: bool) -> Type[T]:
    return {
        CodeReloader: (CodeLoopReloader, CodeDaemonReloader),
        ScriptModuleReloader: (ScriptModuleLoopReloader, ScriptModuleDaemonReloader),
        ScriptFuncReloader: (ScriptFuncLoopReloader, ScriptFuncDaemonReloader),
    }[reloader_cls][not loop]


parser = argparse.ArgumentParser(description=__doc__.strip())
parser.add_argument("-i", "--interactive", action="store_true")
parser.add_argument("-c", "--code", type=str)
parser.add_argument("-m", "--module", type=str)
parser.add_argument("-f", "--func", type=str)
parser.add_argument("-l", "--loop", action="store_true")
parser.add_argument("-w", "--watch", type=Path, action="append", default=[])
parser.add_argument("--clear", "-C", action="store_true")
parser.add_argument("--debounce", "-d", type=float, default=DEFAULT_DEBOUNCE_INTERVAL)
parser.add_argument("--no-cwd-python-path", "-P", action="store_true")
parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
parser.add_argument("script", type=Path, nargs=argparse.OPTIONAL)
parser.add_argument("argv", type=str, nargs=argparse.REMAINDER)


def main():
    args = parser.parse_args()
    if not args.loop and args.watch:
        parser.error("argument -w/--watch: --loop is required")

    if args.code and (args.module or args.func):
        parser.error("argument -c/--code: not allowed with -m/--module or -f/--func")

    if not args.no_cwd_python_path:
        sys.path.insert(0, str(Path.cwd().resolve()))

    watcher = Watcher(debounce_interval=args.debounce)
    for path in args.watch:
        watcher.watch_resource(path)

    if args.code:
        if args.script is not None:
            args.argv.insert(0, args.script)
            args.script = None

        code = compile(args.code, "<code>", "single")
        sys.argv = [code.co_name, *args.argv]

        reloader_cls = get_reloader_cls(CodeReloader, loop=args.loop)
        reloader = reloader_cls(code, watcher=watcher)
    else:
        if args.module:
            module_name, sep, func_name = args.module.partition(":")

            spec = importlib.util.find_spec(module_name)
            if not spec:
                parser.error(f"module not found: {module_name}")

            args.script = Path(spec.origin)
            if sep:
                args.module = module_name
                args.func = func_name

        if not args.script:
            args.script = Path(interactive_code.__file__)
            args.interactive = False

        script_path = args.script.resolve()
        if (script_path / "__main__.py").exists():
            script_path /= "__main__.py"

        script_path = str(script_path)

        sys.argv = [script_path, *args.argv]

        script_module = watcher.watch_script(Path(script_path))

        if args.func:
            reloader_cls = get_reloader_cls(ScriptFuncReloader, loop=args.loop)
            reloader = reloader_cls(
                script_module,
                args.module,
                args.func,
                watcher=watcher,
            )
        else:
            reloader_cls = get_reloader_cls(ScriptModuleReloader, loop=args.loop)
            reloader = reloader_cls(script_module, watcher=watcher)

    if args.clear:
        reloader._before_reload = lambda: os.system("clear")

    watcher.start()

    try:
        reloader.run()
    finally:
        if args.interactive:
            interactive = InteractiveReloader(
                reloader.global_dict,
                watcher=watcher,
            )
            interactive.run()


if __name__ == "__main__":
    main()
