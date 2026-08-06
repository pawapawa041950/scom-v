"""Thin launcher so the app can be started with ``python scomv.py``."""
from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
