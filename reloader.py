#!/usr/bin/env python3
"""
reloader.py - A simple script reloader
inspired by jurigged[develoop] and watchdog

usage: python3 -m reloader script.py
"""

import ast
import sys
import traceback
from collections import Counter
from contextlib import contextmanager
from functools import cache
from os import PathLike
from pathlib import Path
from queue import Queue
from time import monotonic
from typing import Any, Callable, Iterator

from watchdog.events import FileSystemEvent
from watchdog.observers import Observer

__author__ = "EcmaXp"
__version__ = "0.3.0"
__license__ = "The MIT License"
__url__ = "https://github.com/EcmaXp/reloader.py"


class ScriptFileEventHandler:
    def __init__(self):
        self.paths = set()
        self.queue = Queue()
        self.interval = 0.1
        self.last = 0

    @property
    def parents(self):
        return {str(Path(path).parent) for path in self.paths}

    def dispatch(self, event: FileSystemEvent):
        if event.event_type not in ("created", "modified"):
            return
        if event.src_path not in self.paths:
            return

        now = monotonic()
        if now - self.last < self.interval:
            return

        self.last = now
        self.queue.put(True)

    def add(self, path: Path | PathLike | str | bytes):
        self.paths.add(str(Path(path).resolve()))

    def schedule(self, observer: Observer):
        for parent in self.parents:
            observer.schedule(self, str(parent))

    def wait(self) -> bool:
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

    def exec(self, mod_globals: dict):
        exec(self.compile(), mod_globals, mod_globals)


class Chunks(Chunk):
    def __init__(self, node: ast.Module, filename: str):
        self.filename = filename
        self.chunks = [Chunk(block, filename) for block in node.body]
        super().__init__(node, filename)

    def __iter__(self) -> Iterator[Chunk]:
        return iter(self.chunks)


class Patcher:
    def __init__(self, mod_globals: dict):
        self.mod_globals = mod_globals

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
        old_callable.__code__ = new_callable.__code__  # noqa
        return old_callable

    def patch_vars(self, old_obj, new_obj):
        old_vars = vars(old_obj)
        for key, new_value in vars(new_obj).items():
            old_value = old_vars.get(key)
            if key == "__dict__":
                continue

            setattr(old_obj, key, self.patch_object(old_value, new_value))

    @contextmanager
    def patch(self, mod_globals: dict):
        old_globals = mod_globals.copy()
        try:
            yield
        finally:
            self.patch_module(old_globals, mod_globals)


class ScriptFile:
    def __init__(
        self,
        path: Path | PathLike | str | bytes | None,
        module_name: str = "__main__",
    ):
        self.path = Path(path).resolve()
        self.module_name = module_name
        self.chunks = self._load()

        self.globals = {
            "__name__": self.module_name,
            "__file__": str(self.path),
            "__cached__": None,
            "__doc__": None,
            "__loader__": None,
            "__package__": None,
            "__spec__": None,
        }

        self.patcher = Patcher(self.globals)

    def run(self):
        self.chunks.exec(self.globals)

    def reload(self) -> bool:
        chunks = self._load()
        counter = Counter(chunks) - Counter(self.chunks)
        self.chunks = chunks

        for chunk in self.chunks:
            if counter[chunk] > 0 or chunk.is_main():
                counter[chunk] -= 1
                with self.patcher.patch(self.globals):
                    chunk.exec(self.globals)

        return True

    def _load(self) -> Chunks:
        tree = ast.parse(self.path.read_text())
        return Chunks(tree, str(self.path))


def main():
    sys.argv.pop(0)
    if not sys.argv:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    script_file = ScriptFile(sys.argv[0])
    script_file.run()

    observer = Observer()
    observer.start()

    handler = ScriptFileEventHandler()
    handler.add(script_file.path)
    handler.schedule(observer)

    while handler.wait():
        try:
            script_file.reload()
        except Exception:  # noqa
            traceback.print_exc()


if __name__ == "__main__":
    main()
