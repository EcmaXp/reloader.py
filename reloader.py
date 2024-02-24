#!/usr/bin/env python3
"""
reloader.py - A simple script reloader
inspired by jurigged[develoop] and watchdog

usage: python3 -m reloader script.py
"""

import ast
import inspect
import sys
import traceback
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
__version__ = "0.4.0"
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

    def run(self):
        if self.chunks is not None:
            return

        chunks = Chunks.from_path(self.module_path)
        if isinstance(self.source, Path):
            chunks.exec(self.globals)

        self.chunks = chunks

    def reload(self) -> bool:
        chunks = Chunks.from_path(self.module_path)
        counter = Counter(chunks) - Counter(self.chunks)
        self.chunks = chunks

        for chunk in self.chunks:
            if counter[chunk] > 0 or chunk.is_main():
                counter[chunk] -= 1
                with self.patcher.patch(self.globals):
                    chunk.exec(self.globals)

        return True

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
    def __init__(self, reloader: Reloader):
        self.reloader = reloader
        self.observer = Observer()
        self.handler = ScriptFileEventHandler()
        self.handler.add(self.reloader.module_path)
        self.executed = False

    def observe(self):
        self.handler.schedule(self.observer)
        self.observer.start()

    def run(self):
        self.step()
        while self.handler.wait():
            self.step()

    def step(self):
        if not self.executed:
            return self.step_first()
        else:
            return self.step_next()

    def step_first(self):
        if self.executed:
            return True

        try:
            self.reloader.run()
            self.executed = True
            return True
        except Exception:  # noqa
            traceback.print_exc()
            return False

    def step_next(self):
        try:
            self.reloader.reload()
            return True
        except Exception:  # noqa
            traceback.print_exc()
            return False


def main():
    sys.argv.pop(0)
    if not sys.argv:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    reloader = Reloader.from_script(sys.argv[0])

    repl = REPL(reloader)
    repl.observe()
    repl.run()


if __name__ == "__main__":
    main()
