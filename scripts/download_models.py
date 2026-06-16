from __future__ import annotations

from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

CHECKPOINTS = [
    (
        "sam2.1_hiera_large.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    ),
    (
        "sam2.1_hiera_base_plus.pt",
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
    ),
]


def download(name: str, url: str) -> None:
    target = MODELS / name
    if target.exists() and target.stat().st_size > 10_000_000:
        print(f"{name}: already present")
        return

    print(f"{name}: downloading from {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
        tmp.replace(target)
    print(f"{name}: saved to {target}")


def main() -> int:
    for name, url in CHECKPOINTS:
        download(name, url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
