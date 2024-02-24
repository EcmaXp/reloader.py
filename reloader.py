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
import subprocess
import sys
import traceback
from collections import Counter
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from ctypes import pythonapi, c_long, py_object
from os import PathLike
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from time import monotonic, sleep
from types import ModuleType, MemberDescriptorType
from typing import Any, Callable, ClassVar, cast, Generic, TypeVar

from watchdog.events import FileSystemEvent
from watchdog.observers import Observer

__author__ = "EcmaXp"
__version__ = "0.8.6"
__license__ = "The MIT License"
__url__ = "https://github.com/EcmaXp/reloader.py"


T = TypeVar("T")


class InterruptExecution(BaseException):
    pass


class Debouncer:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.last = 0

    def __call__(self):
        now = monotonic()
        if now - self.last < self.interval:
            return False

        self.last = now
        return True


class DebounceFileSystemEventHandler(Generic[T]):
    def __init__(self, observer: Observer, queue: Queue, interrupt: Callable[[], None]):
        self.observer = observer
        self.queue = queue
        self.parents = set()
        self.paths = {}
        self.debouncers = {}
        self.interrupt = interrupt

    def dispatch(self, event: FileSystemEvent):
        if event.event_type in ("created", "modified"):
            path = Path(event.src_path).resolve()
            obj = self.paths.get(path)
            if obj and self.debouncers[path]():
                self.queue.put(obj)
                self.interrupt()

    def add(self, path: Path | PathLike | str, obj: T):
        path = Path(path).resolve()
        parent = path.parent

        self.paths[path] = obj
        self.debouncers[path] = Debouncer()

        if parent not in self.parents:
            self.parents.add(parent)
            self.observer.schedule(self, str(parent))


class Chunk:
    def __init__(self, stmt: ast.stmt, source_code: str, filename: str):
        self.stmt = stmt
        self.source_code = source_code
        self.compiled_code = compile(
            ast.Module(body=[self.stmt], type_ignores=[]), filename, "exec"
        )
        self.filename = filename
        self.is_main = self.source_code.startswith("if __name__")

    @classmethod
    def from_file(cls, stmt: ast.stmt, lines: list[str], file: str):
        lineno, end_lineno = Chunk.get_linenos(stmt)
        source_code = "".join(lines[lineno - 1 : end_lineno])
        return cls(stmt, source_code, file)

    def __hash__(self):
        return hash(self.source_code)

    def __eq__(self, other):
        if isinstance(other, Chunk):
            return self.source_code == other.source_code

        return NotImplemented

    def exec(self, module_globals: dict):
        exec(self.compiled_code, module_globals, module_globals)

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


class Patcher:
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
            # TypeError: descriptor '<new_member>' for '<old_class>' objects doesn't apply to a '<old_class>' object # noqa
            return old_value
        if isinstance(old_value, type) and isinstance(new_value, type):
            return self.patch_class(old_value, new_value)
        elif callable(old_value) and callable(new_value):
            return self.patch_callable(old_value, new_value)
        else:
            return new_value

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
            if hasattr(new_callable, name):
                setattr(old_callable, name, getattr(new_callable, name))
            elif hasattr(new_func, name):
                setattr(old_func, name, getattr(new_func, name))

        return old_callable

    def patch_vars(self, old_obj, new_obj):
        old_vars = vars(old_obj)
        for key, new_value in vars(new_obj).items():
            old_value = old_vars.get(key)
            if key == "__dict__":
                continue

            setattr(old_obj, key, self.patch_object(old_value, new_value))


class Reloader:
    def __init__(self, module: ModuleType, *, is_script: bool = False):
        self.module = module
        self.is_script = is_script
        self.passed_chunks = [] if self.is_script else self.load_chunks()
        self.patcher = Patcher()

    def iter_reloadable_chunks(self, *, with_main: bool = True):
        chunks = self.load_chunks()
        counter = Counter(chunks) - Counter(self.passed_chunks)

        self.passed_chunks = passed_chunks = []
        for chunk in chunks:
            if chunk.is_main:
                if with_main and self.is_script:
                    yield chunk
            elif counter[chunk] > 0:
                yield chunk

            counter[chunk] -= 1
            passed_chunks.append(chunk)

    def reload(self, *, with_main: bool = True):
        for chunk in self.iter_reloadable_chunks(with_main=with_main):
            self.reload_chunk(chunk)

    def reload_chunk(self, chunk: Chunk):
        with self.patcher.patch(self.module.__dict__):
            chunk.exec(self.module.__dict__)

    def load_lines(self):
        linecache.updatecache(self.module.__file__, self.module.__dict__)

        if self.module.__loader__:
            return inspect.getsourcelines(self.module)[0]
        else:
            return Path(self.module.__file__).read_text().splitlines(keepends=True)

    def load_chunks(self):
        lines = self.load_lines()
        source = "".join(lines)
        tree = ast.parse(source, self.module.__file__)

        return [
            Chunk.from_file(stmt, lines, self.module.__file__) for stmt in tree.body
        ]

    @classmethod
    def from_script(cls, script_path: Path | PathLike):
        module = cls.create_script_module(script_path)
        return cls(module, is_script=True)

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


class AutoReloader(Thread):
    CURRENT_INSTANCE: ClassVar[ContextVar] = ContextVar("AutoReloader.CURRENT_INSTANCE")

    def __init__(self, *, reload_with_main: bool, daemon: bool):
        super().__init__(name=type(self).__name__, daemon=daemon)
        self.running = False
        self.queue = Queue[Reloader]()
        self.sys_modules_watcher = SysModulesWatcher(
            check=self._check_module,
            callback=cast(Callable[[ModuleType], None], self.watch_module),
        )
        self.watchdog_observer = Observer()
        self.watchdog_handler = DebounceFileSystemEventHandler[Reloader](
            self.watchdog_observer,
            self.queue,
            self.interrupt,
        )
        self.script_reloader: Reloader | None = None
        self.reload_with_main = reload_with_main
        self.interruptable = False

    @classmethod
    def get_instance(cls) -> AutoReloader:
        try:
            return cls.CURRENT_INSTANCE.get()
        except LookupError as e:
            raise RuntimeError(f"{cls.__name__} is not running") from e

    @staticmethod
    def _check_module(module: ModuleType):
        try:
            return bool(module.__loader__.get_source(module.__name__))  # noqa
        except (AttributeError, ImportError, TypeError):
            return False

    def watch_module(self, module: ModuleType) -> Reloader:
        reloader = Reloader(module)
        self.watchdog_handler.add(module.__file__, reloader)
        return reloader

    def run(self):
        try:
            self.CURRENT_INSTANCE.get()
        except LookupError:
            pass
        else:
            raise RuntimeError(f"{type(self).__name__} is already running")

        threads = [self.watchdog_observer, self.sys_modules_watcher]
        current_instance_token = type(self).CURRENT_INSTANCE.set(self)
        self.running = True

        try:
            for thread in threads:
                thread.start()

            while self.running:
                self.tick()
        finally:
            with suppress(ValueError):
                type(self).CURRENT_INSTANCE.reset(current_instance_token)

            self.running = False
            for thread in threads:
                thread.stop()
                thread.join()

    def tick(self):
        try:
            reloader = self.queue.get(timeout=1)
        except Empty:
            return

        self.on_reload(reloader)
        self.reload(reloader)
        self.queue.task_done()

    def on_reload(self, reloader: Reloader):
        pass

    def stop(self):
        self.running = False

    def reload(self, reloader: Reloader):
        self.interruptable = self.reload_with_main and reloader.is_script

        try:
            reloader.reload(with_main=self.reload_with_main)
        except InterruptExecution:
            pass
        except Exception:  # noqa
            traceback.print_exc()
        finally:
            self.interruptable = False

    def interrupt(self):
        if not self.interruptable:
            return

        pythonapi.PyThreadState_SetAsyncExc(
            c_long(self.ident),
            py_object(InterruptExecution),
        )

    def execute(self):
        raise NotImplementedError


class ScriptReloader(AutoReloader):
    def __init__(self, script_path: Path | PathLike = None):
        super().__init__(reload_with_main=True, daemon=False)
        self.script_debouncer = Debouncer()
        if script_path is not None:
            self.watch_script(script_path)

    def watch_script(self, script_path: Path | PathLike) -> Reloader:
        reloader = Reloader.from_script(script_path)
        self.watchdog_handler.add(reloader.module.__file__, reloader)
        self.queue.put(reloader)
        self.script_debouncer()
        self.script_reloader = reloader
        return reloader

    def reload(self, reloader: Reloader):
        super().reload(reloader)

        if (
            self.script_reloader is not None
            and reloader is not self.script_reloader
            and self.script_debouncer()
        ):
            super().reload(self.script_reloader)

    def execute(self):
        self.start()
        self.join()


class DaemonReloader(AutoReloader):
    def __init__(self, script_path: Path | PathLike = None):
        super().__init__(reload_with_main=False, daemon=True)
        if script_path is not None:
            self.set_script(script_path)

    def set_script(self, script_path: Path | PathLike) -> Reloader:
        reloader = Reloader.from_script(script_path)
        self.watchdog_handler.add(reloader.module.__file__, reloader)
        self.script_reloader = reloader
        return reloader

    def execute(self):
        self.start()
        self.script_reloader.reload(with_main=True)
        self.stop()
        self.join()


parser = argparse.ArgumentParser(description=__doc__.strip())
parser.add_argument("--loop", action="store_true")
parser.add_argument("--clear", action="store_true")
parser.add_argument("script", type=Path)
parser.add_argument("argv", nargs=argparse.REMAINDER)


def main():
    args = parser.parse_args()
    script_path = args.script.resolve()
    sys.argv = [str(script_path), *args.argv]

    auto_reloader_cls = ScriptReloader if args.loop else DaemonReloader
    auto_reloader = auto_reloader_cls(script_path)

    if args.clear:
        auto_reloader.on_reload = lambda reloader: subprocess.call(["clear"])

    auto_reloader.execute()


if __name__ == "__main__":
    main()
