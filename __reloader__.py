#!/usr/bin/env python3
import os
import sys

root = os.path.join(os.path.dirname(__file__))
sys.modules.pop("reloader", None)
sys.path.insert(0, root)

try:
    from reloader import *  # noqa
    from reloader import __author__, __version__, __license__, __url__, __all__, main  # noqa
finally:
    sys.path.remove(root)

del os, sys, root


if __name__ == "__main__":
    main()
