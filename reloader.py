#!/usr/bin/env python3
"""
reloader.py - A simple script reloader
inspired by jurigged[develoop] and watchdog

usage: python3 -m reloader script.py
"""

import ast
import inspect
import linecache
import sys
import traceback
from collections import Counter
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from functools import cache
from os import PathLike
from pathlib import Path
from queue import Queue
from threading import Thread
from time import monotonic, sleep
from types import ModuleType
from typing import Any, Callable, ClassVar

from watchdog.events import FileSystemEvent
from watchdog.observers import Observer

__author__ = "EcmaXp"
__version__ = "0.6.0"
__license__ = "The MIT License"
__url__ = "https://github.com/EcmaXp/reloader.py"


def get_linenos(node: ast.AST) -> tuple[int, int]:
    lineno = float("inf")
    end_lineno = float("-inf")

    for child in ast.walk(node):
        if hasattr(child, "lineno"):
            lineno = min(lineno, child.lineno)
        if hasattr(child, "end_lineno"):
            end_lineno = max(end_lineno, child.end_lineno)

    return lineno, end_lineno


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


class DebounceFileSystemEventHandler[T]:
    def __init__(self, observer: Observer):
        self.observer = observer
        self.queue = Queue()
        self.parents = set()
        self.paths = {}
        self.debouncers = {}

    def dispatch(self, event: FileSystemEvent):
        if event.event_type in ("created", "modified"):
            path = Path(event.src_path).resolve()
            obj = self.paths.get(path)
            if obj and self.debouncers[path]():
                self.queue.put(obj)

    def add(self, path: Path | PathLike | str, obj: T):
        path = Path(path).resolve()
        parent = path.parent

        self.paths[path] = obj
        self.debouncers[path] = Debouncer()

        if parent not in self.parents:
            self.parents.add(parent)
            self.observer.schedule(self, str(parent))

    def wait(self) -> T:
        return self.queue.get()


class Chunk:
    def __init__(self, node: ast.AST, code: str, filename: str):
        self.node = node
        self.code = code
        self.filename = filename
        self.lineno = node.lineno if hasattr(node, "lineno") else 0

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, Chunk):
            return self.code == other.code

        return NotImplemented

    @cache
    def is_main(self):
        return self.code.splitlines()[0].startswith("if __name__")

    @cache
    def compile(self):
        padding = "\n" * (self.lineno - 1)
        return compile(padding + self.code, self.filename, "exec")

    def exec(self, module_globals: dict):
        exec(self.compile(), module_globals, module_globals)


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

        for name in ("__code__", "__annotations__", "__kwdefaults__", "__defaults__"):
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
    def __init__(self, module: ModuleType, *, module_is_script: bool = False):
        self.module = module
        self.module_is_script = module_is_script
        self.executed_chunks = [] if self.module_is_script else self.load_chunks()
        self.patcher = Patcher()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.module!r}, is_script={self.module_is_script!r})"

    def reload(self):
        found_chunks = self.load_chunks()
        counter = Counter(found_chunks) - Counter(self.executed_chunks)

        self.executed_chunks = []
        for chunk in found_chunks:
            if counter[chunk] > 0 or (self.module_is_script and chunk.is_main()):
                counter[chunk] -= 1
                with self.patcher.patch(self.module.__dict__):
                    chunk.exec(self.module.__dict__)

            self.executed_chunks.append(chunk)

    def load_lines(self):
        linecache.updatecache(self.module.__file__, self.module.__dict__)

        if self.module.__loader__:
            return inspect.getsourcelines(self.module)[0]
        else:
            return Path(self.module.__file__).read_text().splitlines(keepends=True)

    def load_chunks(self):
        lines = self.load_lines()
        source = "".join(lines)
        tree = ast.parse(source)

        chunks = []
        for node in tree.body:
            lineno, end_lineno = get_linenos(node)
            code = "".join(lines[lineno - 1 : end_lineno])
            chunks.append(Chunk(node, code, self.module.__file__))

        return chunks

    @classmethod
    def from_script(cls, script_path: Path | PathLike):
        script_module = cls.create_script_module(script_path)
        return cls(script_module, module_is_script=True)

    @classmethod
    def create_script_module(cls, script_path: Path | PathLike) -> ModuleType:
        script_path = Path(script_path).resolve()
        script_module = ModuleType("__main__")
        script_module.__file__ = str(script_path)
        script_module.__cached__ = None
        script_module.__package__ = None
        return script_module


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

            if module_name == "script":
                module = sys.modules[module_name]
                print(module_name, self.check(module))

            module = sys.modules[module_name]
            if self.check(module):
                self.watched.add(module_name)
                self.callback(module)
            else:
                self.ignored.add(module_name)


class AutoReloader:
    CURRENT_INSTANCE: ClassVar[ContextVar] = ContextVar("AutoReloader.CURRENT_INSTANCE")

    def __init__(self, script_path: Path | PathLike = None):
        self.sys_modules_watcher = SysModulesWatcher(
            check=self._check_module,
            callback=self.watch_module,
        )
        self.watchdog_observer = Observer()
        self.watchdog_handler = DebounceFileSystemEventHandler(self.watchdog_observer)
        self.script_reloader: Reloader | None = None
        self.script_debouncer = Debouncer()
        if script_path is not None:
            self.watch_script(script_path)

    @classmethod
    def get_instance(cls) -> "AutoReloader":
        try:
            return cls.CURRENT_INSTANCE.get()
        except LookupError as e:
            raise RuntimeError("AutoReloader is not running") from e

    @staticmethod
    def _check_module(module: ModuleType):
        loader = getattr(module, "__loader__", None)
        try:
            if not loader or not loader.get_source(module.__name__):
                raise ValueError("no source available")
        except (ImportError, TypeError, ValueError):
            return False
        else:
            return True

    def watch_module(self, module: ModuleType) -> None:
        reloader = Reloader(module)
        self.watchdog_handler.add(module.__file__, reloader)

    def watch_script(self, script_path: Path | PathLike):
        self.script_reloader = Reloader.from_script(script_path)
        self.watchdog_handler.add(
            self.script_reloader.module.__file__,
            self.script_reloader,
        )
        self.watchdog_handler.queue.put_nowait(self.script_reloader)

    def run(self):
        try:
            self.CURRENT_INSTANCE.get()
        except LookupError:
            pass
        else:
            raise RuntimeError(f"{type(self).__name__} is already running")

        current_instance_token = type(self).CURRENT_INSTANCE.set(self)

        try:
            self.watchdog_observer.start()
            self.sys_modules_watcher.start()
            self.script_debouncer()

            while True:
                reloader = self.watchdog_handler.wait()
                self.reload(reloader)

                if (
                    self.script_reloader is not None
                    and reloader is not self.script_reloader
                    and self.script_debouncer()
                ):
                    self.reload(self.script_reloader)
        finally:
            with suppress(ValueError):
                type(self).CURRENT_INSTANCE.reset(current_instance_token)

            self.sys_modules_watcher.stop()
            self.sys_modules_watcher.join()
            self.watchdog_observer.stop()
            self.watchdog_observer.join()

    @staticmethod
    def reload(reloader: Reloader):
        try:
            reloader.reload()
        except Exception:  # noqa
            traceback.print_exc()


def main():
    sys.argv.pop(0)
    if not sys.argv:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    script_path = Path(sys.argv[0]).resolve()

    auto_reloader = AutoReloader(script_path)
    auto_reloader.run()


if __name__ == "__main__":
    main()
