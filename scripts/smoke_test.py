from __future__ import annotations

import importlib


MODULES = [
    "fastapi",
    "uvicorn",
    "cv2",
    "numpy",
    "PIL.Image",
    "sam2",
    "app.main",
]


def main() -> int:
    for module in MODULES:
        importlib.import_module(module)
        print(f"{module}: ok")
    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
