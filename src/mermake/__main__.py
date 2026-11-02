"""``python -m mermake`` is the same as the ``mermake`` command."""
import sys

from .cli import main

if __name__ == "__main__":
	sys.exit(main())
