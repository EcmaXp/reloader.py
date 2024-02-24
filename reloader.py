#!/usr/bin/env python3
"""
reloader.py - A simple script reloader
inspired by jurigged[develoop] and watchdog

usage: python3 -m reloader script.py
"""

import ast
import inspect
import sys
from collections import Counter
from contextlib import contextmanager
from functools import cache
from os import PathLike
from pathlib import Path
from queue import Queue
from time import monotonic
from types import ModuleType
from typing import Any, Callable, Iterator

from watchdog.events import FileSystemEvent
from watchdog.observers import Observer

__author__ = "EcmaXp"
__version__ = "0.5.0"
__license__ = "The MIT License"
__url__ = "https://github.com/EcmaXp/reloader.py"


class DebounceFileEventHandler[T]:
    def __init__(self):
        self.queue = Queue()
        self.interval = 0.1
        self.paths = {}
        self.lasts = {}

    @property
    def parents(self):
        return {str(Path(path).parent) for path in self.paths}

    def dispatch(self, event: FileSystemEvent):
        if event.event_type not in ("created", "modified"):
            return

        path = event.src_path
        obj = self.paths.get(path)
        if not obj:
            return

        now = monotonic()
        if now - self.lasts[path] < self.interval:
            return

        self.lasts[path] = now
        self.queue.put(obj)

    def add(self, path: Path | PathLike | str | bytes, obj: T):
        path = str(Path(path).resolve())
        self.paths[path] = obj
        self.lasts[path] = 0
        self.queue.put_nowait(obj)

    def schedule(self, observer: Observer):
        for parent in self.parents:
            observer.schedule(self, str(parent))

    def wait(self) -> T:
        return self.queue.get()


class Chunk:
    def __init__(self, node: ast.AST, filename: str):
        self.node = node
        self.code = ast.unparse(node)
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


class Chunks(Chunk):
    def __init__(self, node: ast.Module, filename: str):
        self.filename = filename
        self.chunks = [Chunk(block, filename) for block in node.body]
        super().__init__(node, filename)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)

    @classmethod
    def from_path(cls, path: Path | PathLike | str | bytes):
        path = Path(path).resolve()
        tree = ast.parse(path.read_text())
        return cls(tree, str(path))


class Patcher:
    def __init__(self, module_globals: dict):
        self.module_globals = module_globals

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

        try:
            old_callable.__code__ = new_callable.__code__  # noqa
        except AttributeError:
            old_func = inspect.unwrap(old_callable)
            new_func = inspect.unwrap(new_callable)
            old_func.__code__ = new_func.__code__  # noqa

        return old_callable

    def patch_vars(self, old_obj, new_obj):
        old_vars = vars(old_obj)
        for key, new_value in vars(new_obj).items():
            old_value = old_vars.get(key)
            if key == "__dict__":
                continue

            setattr(old_obj, key, self.patch_object(old_value, new_value))

    @contextmanager
    def patch(self, module_globals: dict):
        old_globals = module_globals.copy()
        try:
            yield
        finally:
            self.patch_module(old_globals, module_globals)


class Reloader:
    def __init__(
        self,
        module_name: str,
        module_path: Path | PathLike | str | bytes,
        module_globals: dict,
        source: Path | ModuleType,
    ):
        self.module_name = module_name
        self.module_path = Path(module_path).resolve()
        self.globals = module_globals
        self.chunks = None
        self.patcher = Patcher(self.globals)
        self.source = source

    def __post_init__(self):
        self.chunks = Chunks.from_path(self.module_path)
        self.patcher = Patcher(self.globals)

    def step(self):
        if self.chunks is None:
            self.run()
        else:
            self.reload()

    def run(self):
        if self.chunks is not None:
            return

        chunks = Chunks.from_path(self.module_path)
        if isinstance(self.source, Path):
            chunks.exec(self.globals)

        self.chunks = chunks

    def reload(self):
        chunks = Chunks.from_path(self.module_path)
        counter = Counter(chunks) - Counter(self.chunks)
        self.chunks = chunks

        for chunk in self.chunks:
            if counter[chunk] > 0 or (
                not isinstance(self.source, ModuleType) and chunk.is_main()
            ):
                counter[chunk] -= 1
                with self.patcher.patch(self.globals):
                    chunk.exec(self.globals)

    @classmethod
    def from_script(
        cls,
        script_path: Path | PathLike | str | bytes,
        module_name: str = "__main__",
    ):
        script_path = Path(script_path).resolve()

        return cls(
            module_path=script_path,
            module_name=module_name,
            module_globals={
                "__name__": module_name,
                "__file__": str(script_path),
                "__cached__": None,
                "__doc__": None,
                "__loader__": None,
                "__package__": None,
                "__spec__": None,
            },
            source=script_path,
        )

    @classmethod
    def from_module(cls, module: Any):
        return cls(
            module_path=Path(module.__file__).resolve(),
            module_name=module.__name__,
            module_globals=module.__dict__,
            source=module,
        )


class REPL:
    def __init__(self, reloader_or_source: Reloader | Path | ModuleType):
        self.observer = Observer()
        self.handler = DebounceFileEventHandler()
        self.reloader = self.watch(reloader_or_source)
        self.executed = False

    def watch(self, reloader_or_source: Reloader | Path | ModuleType) -> Reloader:
        if isinstance(reloader_or_source, Reloader):
            reloader = reloader_or_source
        elif isinstance(reloader_or_source, Path):
            path = reloader_or_source
            reloader = Reloader.from_script(path)
        elif isinstance(reloader_or_source, ModuleType):
            module = reloader_or_source
            reloader = Reloader.from_module(module)
        else:
            raise TypeError(f"unsupported type: {type(reloader_or_source)}")

        self.handler.add(reloader.module_path, reloader)
        return reloader

    def run(self):
        self.handler.schedule(self.observer)
        self.observer.start()

        while True:
            reloader = self.handler.wait()
            reloader.step()

            if reloader is not self.reloader:
                self.reloader.step()


def main():
    sys.argv.pop(0)
    if not sys.argv:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    repl = REPL(Path(sys.argv[0]).resolve())
    repl.run()


if __name__ == "__main__":
    main()
