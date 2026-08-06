"""Thin launcher so the app can be started with ``python scom.py``."""
from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
