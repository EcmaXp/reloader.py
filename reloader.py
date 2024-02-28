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
__version__ = "0.10.4"
__license__ = "MIT"
__url__ = "https://pypi.org/project/reloader.py/"
__all__ = ["Reloader", "DaemonReloader", "ScriptLoopReloader", "ScriptDaemonReloader"]


T = TypeVar("T")

DEFAULT_DEBOUNCE_INTERVAL = 0.1


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
    def __init__(self, stmt: ast.stmt, source_code: str, filename: str):
        self.stmt = stmt
        self.source_code = source_code
        self.compiled_code = compile(
            ast.Module(body=[self.stmt], type_ignores=[]), filename, "exec"
        )
        self.filename = filename
        self.is_main = self.source_code.startswith("if __name__")

    def __hash__(self):
        return hash(self.source_code)

    def __eq__(self, other):
        if isinstance(other, CodeChunk):
            return self.source_code == other.source_code

        return NotImplemented

    def exec(self, module_globals: dict):
        exec(self.compiled_code, module_globals, module_globals)

    @classmethod
    def from_file(cls, stmt: ast.stmt, lines: list[str], file: str):
        lineno, end_lineno = cls.get_linenos(stmt)
        source_code = "".join(lines[lineno - 1 : end_lineno])
        return cls(stmt, source_code, file)

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
        self.passed_chunks = self.init_chunks()
        self.patcher = ModulePatcher()

    def init_chunks(self):
        return self.load_chunks()

    def load_chunks(self):
        lines = self.load_lines()
        source = "".join(lines)
        tree = ast.parse(source, self.module.__file__)

        return [
            CodeChunk.from_file(stmt, lines, self.module.__file__) for stmt in tree.body
        ]

    def iter_reloadable_code_chunks(self, *, with_main: bool = False):
        chunks = self.load_chunks()
        counter = Counter(chunks) - Counter(self.passed_chunks)

        self.passed_chunks = passed_chunks = []
        for chunk in chunks:
            if chunk.is_main:
                if with_main:
                    yield chunk
            elif counter[chunk] > 0:
                yield chunk

            counter[chunk] -= 1
            passed_chunks.append(chunk)

    def reload(self, *, with_main: bool = False):
        for chunk in self.iter_reloadable_code_chunks(with_main=with_main):
            self.reload_code_chunk(chunk)

    def reload_code_chunk(self, chunk: CodeChunk):
        self.patcher.exec(chunk, self.module.__dict__)

    def load_lines(self):
        linecache.updatecache(self.module.__file__, self.module.__dict__)

        if self.module.__loader__:
            return inspect.getsourcelines(self.module)[0]
        else:
            return Path(self.module.__file__).read_text().splitlines(keepends=True)


class ScriptModule(CodeModule):
    def __init__(self, script_path: Path | PathLike):
        module = self.create_script_module(script_path)
        super().__init__(module)

    def init_chunks(self):
        return []

    def run(self):
        self.reload(with_main=True)

    @classmethod
    def create_script_module(cls, script_path: Path | PathLike) -> ModuleType:
        module = ModuleType("__main__")
        module.__file__ = str(Path(script_path).resolve())
        module.__cached__ = None
        module.__package__ = None
        return module


class SysModulesWatcher(Thread):
    def __init__(
        self,
        check: Callable[[ModuleType], bool],
        callback: Callable[[ModuleType], None],
    ):
        self.watched = set()
        self.ignored = set()
        self.check = check
        self.callback = callback
        self.running = None
        super().__init__(name=type(self).__name__, daemon=True)

    def start(self):
        self.watch_all()
        self.running = True
        super().start()

    def run(self):
        while self.running:
            self.watch_all()
            sleep(1)

    def stop(self):
        self.running = False

    def watch_all(self):
        module_names = list(sys.modules)
        modified_names = set(module_names) - self.watched - self.ignored
        if not modified_names:
            return

        for module_name in reversed(module_names):
            if module_name not in modified_names:
                continue

            module = sys.modules[module_name]
            if self.check(module):
                self.watched.add(module_name)
                self.callback(module)
            else:
                self.ignored.add(module_name)


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
        self._sys_modules_watcher = SysModulesWatcher(
            check=self._check_module,
            callback=cast(Callable[[ModuleType], None], self.watch_module),
        )
        self._ident = None
        self._running = False
        self._interruptable = False

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

    @classmethod
    def get_instance(cls) -> Reloader:
        try:
            return cls._CURRENT_INSTANCE.get()
        except LookupError as e:
            raise RuntimeError(f"{cls.__name__} is not running") from e

    @staticmethod
    def _check_module(module: ModuleType):
        try:
            return bool(module.__loader__.get_source(module.__name__))  # noqa
        except (AttributeError, ImportError, TypeError):
            return False

    def watch_module(self, module: ModuleType) -> CodeModule:
        self._code_modules[module.__file__] = code_module = CodeModule(module)
        self._watchdog_handler.schedule(module.__file__)
        return code_module

    def watch_resource(self, path: Path | PathLike | str) -> None:
        self._watchdog_handler.schedule(path)

    def _run(self):
        try:
            self._CURRENT_INSTANCE.get()
        except LookupError:
            pass
        else:
            raise RuntimeError(f"{type(self).__name__} is already running")

        threads = [
            self._watchdog_observer,
            self._watchdog_debouncer,
            self._sys_modules_watcher,
        ]
        current_instance_token = type(self)._CURRENT_INSTANCE.set(self)
        self._ident = threading.get_ident()
        self._running = True

        try:
            for thread in threads:
                thread.start()

            self._first_tick()
            while self._running:
                self._tick()
        finally:
            self._running = False
            self._ident = None

            with suppress(ValueError):
                type(self)._CURRENT_INSTANCE.reset(current_instance_token)

            for thread in threads:
                thread.stop()
                thread.join()

    def _tick(self):
        try:
            events = self._queue.get(timeout=1)
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
        self._interrupt()

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
parser.add_argument("script", type=Path)
parser.add_argument("argv", nargs=argparse.REMAINDER)


def main():
    args = parser.parse_args()
    if not args.loop and args.resource_file:
        parser.error("argument -f/--resource-file: --loop is required")

    script_path = args.script.resolve()
    sys.argv = [str(script_path), *args.argv]

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
