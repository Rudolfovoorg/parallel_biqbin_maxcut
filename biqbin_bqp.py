"""Compatibility entry point for the shared bqp runtime."""

from biqbin.cli import main

if __name__ == "__main__":
    raise SystemExit(main(kind="bqp"))
