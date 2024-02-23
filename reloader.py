#!/usr/bin/env python3
"""
reloader.py - A simple script reloader
inspired by jurigged[develoop] and watchdog

usage: python3 -m reloader script.py
"""

import ast
import runpy
import sys
import traceback
from os import PathLike
from pathlib import Path
from queue import Queue
from time import monotonic

from watchdog.events import FileSystemEvent
from watchdog.observers import Observer

__author__ = "EcmaXp"
__version__ = "0.1.0"
__license__ = "The MIT License"
__url__ = "https://github.com/EcmaXp/reloader.py"


class ScriptFileEventHandler:
    def __init__(self, file: PathLike | str | bytes):
        self.path = Path(file).resolve()
        self.src_path = str(self.path)
        self.queue = Queue()
        self.interval = 0.1
        self.last = 0

    def dispatch(self, event: FileSystemEvent):
        if event.event_type not in ("created", "modified"):
            return
        if event.src_path != self.src_path:
            return

        now = monotonic()
        if now - self.last < self.interval:
            return

        self.last = now
        self.queue.put(True)

    def wait(self) -> bool:
        return self.queue.get()


class ScriptFile:
    def __init__(
        self,
        path: Path | PathLike | str | bytes | None,
        module_name: str = "__main__",
    ):
        self.path = Path(path).resolve()
        self.handler = ScriptFileEventHandler(self.path)
        self.codes = []
        self.globals = {"__name__": module_name}

    def run(self):
        self.codes = self._load()
        self.globals.update(
            runpy.run_path(
                str(self.path),
                self.globals,
                self.globals["__name__"],
            )
        )

    def reload(self) -> bool:
        codes = self._load()
        added = set(codes) - set(self.codes)
        self.codes = codes
        if not added:
            return False

        for code in self.codes:
            if code in added:
                self._exec(code)
            elif code.lstrip("\n").splitlines()[0].startswith("if __name__"):
                self._exec(code)

        return True

    def _load(self) -> list[str]:
        return [
            ("\n" * (block.lineno - 1)) + ast.unparse(block)
            for block in ast.parse(self.path.read_text()).body
        ]

    def _exec(self, code: str):
        compiled = compile(code, str(self.path), "exec")
        exec(compiled, self.globals, self.globals)

    def loop(self):
        observer = Observer()
        observer.start()

        self.observe(observer)
        self.run()

        while self.wait():
            try:
                self.reload()
            except Exception:  # noqa
                traceback.print_exc()

    def observe(self, observer: Observer):
        observer.schedule(self.handler, str(self.path.parent))

    def wait(self) -> bool:
        return self.handler.wait()


def main():
    sys.argv.pop(0)
    if not sys.argv:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)

    ScriptFile(sys.argv[0]).loop()


if __name__ == "__main__":
    main()
