#!/usr/bin/env python3
"""
reloader.py - A simple script reloader
inspired by jurigged[develoop] and watchdog
"""

from __future__ import annotations

import argparse
import ast
import inspect
import linecache
import os
import sys
import threading
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from ctypes import pythonapi, c_long, py_object
from importlib.abc import MetaPathFinder
from os import PathLike
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from time import sleep
from types import ModuleType, MemberDescriptorType
from typing import Any, Callable, ClassVar, cast, TypeVar

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.utils.event_debouncer import EventDebouncer

__author__ = "EcmaXp"
__version__ = "0.12.2"
__license__ = "MIT"
__url__ = "https://pypi.org/project/reloader.py/"
__all__ = [
    "current_reloader",
    "Reloader",
    "DaemonReloader",
    "ScriptLoopReloader",
    "ScriptDaemonReloader",
]


T = TypeVar("T")

DEFAULT_DEBOUNCE_INTERVAL = 0.1

REPLACEABLE_AST_TYPES = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
REPLACEABLE_OBJ_TYPES = type | Callable


class ReloadUnsupportedWarning(UserWarning):
    pass


class Interrupted(BaseException):
    pass


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
            self.patch_module(old_globals, module_globals)

    def patch_module(self, old_globals: dict, new_globals: dict):
        for key, new_value in new_globals.items():
            old_value = old_globals.get(key)
            if old_value is not new_value:
                new_globals[key] = self.patch_object(old_value, new_value)

    def patch_object(self, old_value: Any, new_value: Any):
        if isinstance(new_value, MemberDescriptorType):
            warnings.warn(
                "MemberDescriptor is not supported",
                ReloadUnsupportedWarning,
            )
            return old_value
        elif not self.check_object(old_value, new_value):
            return new_value
        elif isinstance(old_value, type) and isinstance(new_value, type):
            return self.patch_class(old_value, new_value)
        elif callable(old_value) and callable(new_value):
            return self.patch_callable(old_value, new_value)
        else:
            return new_value

    @staticmethod
    def check_object(old_value: Any, new_value: Any):
        old_module = getattr(old_value, "__module__", None)
        new_module = getattr(new_value, "__module__", None)
        return old_module == new_module

    def patch_class(self, old_class: type, new_class: type):
        self.patch_vars(old_class, new_class)
        return old_class

    def patch_callable(self, old_callable: Callable, new_callable: Callable):
        self.patch_vars(old_callable, new_callable)

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

    def patch_vars(self, old_obj, new_obj):
        old_vars = vars(old_obj)
        for key, new_value in vars(new_obj).items():
            old_value = old_vars.get(key)
            if key == "__dict__":
                continue

            setattr(old_obj, key, self.patch_object(old_value, new_value))


class CodeModule:
    def __init__(self, module: ModuleType):
        self.module = module
        self.code_chunks = self.load_chunks()
        self.executed = Counter(self.code_chunks)
        self.patcher = ModulePatcher()

    def __repr__(self):
        return f"{type(self).__name__}({self.module!r})"

    @property
    def name(self):
        return self.module.__name__

    @property
    def file(self):
        return self.module.__file__

    @property
    def path(self):
        return Path(self.file)

    def load_lines(self) -> list[str]:
        linecache.updatecache(self.file, self.module.__dict__)

        if self.module.__loader__:
            lines = inspect.getsourcelines(self.module)[0][:]
        else:
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

    def iter_reloadable_code_chunks(self, *, with_main: bool = False):
        code_chunks = self.code_chunks = self.load_chunks()
        counter = Counter(code_chunks)

        for code_chunk in self.executed | counter:
            count = min(self.executed[code_chunk], counter[code_chunk])
            if count > 0:
                self.executed[code_chunk] = count
            else:
                self.executed.pop(code_chunk, None)

        for code_chunk in code_chunks:
            if isinstance(code_chunk, CodeChunk):
                if code_chunk.is_main:
                    if with_main:
                        yield code_chunk
                elif counter[code_chunk] > self.executed[code_chunk]:
                    yield code_chunk

    def reload(self, *, with_main: bool = False):
        for code_chunk in self.iter_reloadable_code_chunks(with_main=with_main):
            self.reload_code_chunk(code_chunk)

    def reload_code_chunk(self, chunk: CodeChunk):
        self.patcher.exec(chunk, self.module.__dict__)
        self.executed[chunk] += 1

    def get_code_chunk(self, obj: REPLACEABLE_OBJ_TYPES) -> CodeChunk:
        for chunk in self.code_chunks:
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
        self.code_chunks[self.code_chunks.index(old)] = new_chunk
        self.executed.pop(old, None)
        self.reload_code_chunk(new_chunk)
        self.executed[new_chunk] += 1
        return new_chunk

    def write(self):
        with self.path.open("w") as file:
            for chunk in self.code_chunks:
                if isinstance(chunk, CodeChunk):
                    file.write(chunk.text)
                else:
                    file.write(chunk)


class ScriptModule(CodeModule):
    def __init__(self, script_path: Path | PathLike):
        module = self.create_script_module(script_path)
        super().__init__(module)
        self.executed.clear()

    def run(self):
        self.reload(with_main=True)

    @classmethod
    def create_script_module(cls, script_path: Path | PathLike) -> ModuleType:
        module = ModuleType("__main__")
        module.__file__ = str(Path(script_path).resolve())
        module.__cached__ = None
        module.__package__ = None
        return module


class SysModulesWatcher(MetaPathFinder, Thread):
    def __init__(self, callback: Callable[[ModuleType], None]):
        super().__init__(name=type(self).__name__, daemon=True)
        self.callback = callback
        self.running = None
        self.found = set()
        self.event = threading.Event()

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


class Reloader:
    _CURRENT_INSTANCE: ClassVar[ContextVar] = ContextVar("Reloader.CURRENT_INSTANCE")

    def __init__(self, *, debounce_interval: float = DEFAULT_DEBOUNCE_INTERVAL):
        self._code_modules: dict[str, CodeModule | ScriptModule] = {}
        self._queue = Queue[list[FileSystemEvent | bool]]()
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
        self._ident = None
        self._running = False
        self._interruptable = False

    def _get_threads(self):
        return (
            self._watchdog_observer,
            self._watchdog_debouncer,
            self._sys_modules_watcher,
        )

    def _events_callback(self, events: list[FileSystemEvent | bool]):
        new_events = []
        for event in events:
            if isinstance(event, FileSystemEvent):
                if event.event_type not in ("modified", "created"):
                    continue

                new_events.append(event)
            elif isinstance(event, bool):
                new_events.append(event)

        if new_events:
            self._queue.put(new_events)
            self._interrupt()

    def _module_callback(self, module: ModuleType):
        if self._check_module(module):
            self.watch_module(module)

    @classmethod
    def get_instance(cls) -> Reloader:
        try:
            return cls._CURRENT_INSTANCE.get()
        except LookupError as e:
            raise RuntimeError(f"{cls.__name__} is not running") from e

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
            return bool(module.__loader__.get_source(module.__name__))  # noqa
        except (AttributeError, ImportError, TypeError):
            return False

    def watch_module(self, module: ModuleType) -> CodeModule:
        code_module = self._code_modules.get(module.__file__)
        if code_module is not None:
            return code_module

        code_module = CodeModule(module)
        self._code_modules[module.__file__] = code_module
        self._watchdog_handler.schedule(module.__file__)
        return code_module

    def watch_resource(self, path: Path | PathLike | str) -> None:
        self._watchdog_handler.schedule(path)

    def _run(self):
        try:
            current_reloader()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(f"{type(self).__name__} is already running")

        current_instance_token = type(self)._CURRENT_INSTANCE.set(self)
        self._ident = threading.get_ident()
        self._running = True

        try:
            for thread in self._get_threads():
                thread.start()

            self._first_tick()
            while self._running:
                self._tick()
        finally:
            self._ident = None
            self._running = False

            with suppress(ValueError):
                type(self)._CURRENT_INSTANCE.reset(current_instance_token)

            for thread in self._get_threads():
                if thread.is_alive():
                    thread.stop()
                    thread.join()

    def _tick(self):
        try:
            events = self._queue.get(timeout=0.1)
        except Empty:
            return

        self.before_tick()

        code_modules = {
            self._code_modules.get(event.src_path)
            for event in events
            if isinstance(event, FileSystemEvent)
            and event.event_type in ("modified", "created")
        } - {None}

        for code_module in code_modules:
            self._reload(code_module)

        self.after_tick()
        self._queue.task_done()

    def _first_tick(self):
        pass

    def before_tick(self):
        pass

    def after_tick(self):
        pass

    def run(self):
        self._run()

    def stop(self):
        self._running = False

    def _reload(
        self,
        code_module: CodeModule | ScriptModule,
        *,
        is_main_script: bool = False,
    ):
        self._interruptable = is_main_script

        try:
            code_module.reload(with_main=is_main_script)
        except Interrupted:
            pass
        except Exception:  # noqa
            traceback.print_exc()
        finally:
            self._interruptable = False

    def _interrupt(self):
        if self._interruptable and self._ident is not None:
            pythonapi.PyThreadState_SetAsyncExc(
                c_long(self._ident),
                py_object(Interrupted),
            )


class DaemonReloader(Reloader):
    def __init__(self, *, debounce_interval: float = DEFAULT_DEBOUNCE_INTERVAL):
        super().__init__(debounce_interval=debounce_interval)
        self.thread: Thread | None = None

    def start(self):
        self.thread = Thread(
            target=self._run,
            name=type(self).__name__,
            daemon=True,
        )
        self.thread.start()

    def run(self):
        raise NotImplementedError

    def join(self):
        if self.thread:
            self.thread.join()


class ScriptReloader(Reloader):
    def __init__(
        self,
        script_path: Path | PathLike = None,
        *,
        debounce_interval: float = DEFAULT_DEBOUNCE_INTERVAL,
    ):
        super().__init__(debounce_interval=debounce_interval)
        self._script_module: ScriptModule | None = None
        if script_path is not None:
            self.watch_script(script_path)

    def get_code_module(self, module_name: str) -> CodeModule | ScriptModule:
        if self._script_module and module_name == self._script_module.name:
            return self._script_module

        return super().get_code_module(module_name)

    def watch_script(self, script_path: Path | PathLike) -> ScriptModule:
        self._script_module = script_module = ScriptModule(script_path)
        self._code_modules[script_module.module.__file__] = script_module
        self._watchdog_handler.schedule(script_module.module.__file__)
        return script_module


class ScriptLoopReloader(ScriptReloader):
    def _first_tick(self):
        if self._script_module:
            self._queue.put([True])

    def after_tick(self):
        self._reload(self._script_module, is_main_script=True)


class ScriptDaemonReloader(ScriptReloader, DaemonReloader):
    def run(self):
        self.start()
        try:
            self.before_tick()
            self._script_module.run()
            self.after_tick()
        finally:
            self.stop()
            self.join()


def current_reloader() -> Reloader:
    return Reloader.get_instance()


parser = argparse.ArgumentParser(description=__doc__.strip())
parser.add_argument("--loop", "-l", action="store_true")
parser.add_argument("--clear", "-c", action="store_true")
parser.add_argument(
    "--debounce-interval",
    "-i",
    type=float,
    default=DEFAULT_DEBOUNCE_INTERVAL,
)
parser.add_argument(
    "-f",
    "--resource-file",
    type=Path,
    action="append",
    default=[],
)
parser.add_argument(
    "--pwd-python-path",
    action=argparse.BooleanOptionalAction,
    default=True,
)
parser.add_argument(
    "--version",
    action="version",
    version=f"%(prog)s {__version__}",
)
parser.add_argument("script", type=Path)
parser.add_argument("argv", nargs=argparse.REMAINDER)


def main():
    args = parser.parse_args()
    if not args.loop and args.resource_file:
        parser.error("argument -f/--resource-file: --loop is required")

    script_path: Path = args.script.resolve()
    if (script_path / "__main__.py").exists():
        script_path /= "__main__.py"

    sys.argv = [str(script_path), *args.argv]

    if args.pwd_python_path:
        sys.path.insert(0, ".")

    reloader_cls = ScriptLoopReloader if args.loop else ScriptDaemonReloader
    reloader = reloader_cls(script_path, debounce_interval=args.debounce_interval)

    for path in args.resource_file:
        reloader.watch_resource(path)

    if args.clear:
        reloader.before_tick = lambda: os.system("clear")

    reloader.run()


if __name__ == "__main__":
    main()
