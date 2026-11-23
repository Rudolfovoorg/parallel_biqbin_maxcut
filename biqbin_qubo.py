"""Compatibility entry point for the shared qubo runtime."""

from biqbin.cli import main

if __name__ == "__main__":
    raise SystemExit(main(kind="qubo"))
