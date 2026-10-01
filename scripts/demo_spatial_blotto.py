#!/usr/bin/env python3
"""Compatibility entry point for the installed spatial-blotto demo CLI."""

from spatial_blotto.cli import build_parser, main, simulate

__all__ = ["build_parser", "main", "simulate"]

if __name__ == "__main__":
    raise SystemExit(main())
