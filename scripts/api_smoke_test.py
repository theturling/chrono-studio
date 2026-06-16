from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import requests


BASE_URL = "http://127.0.0.1:8766"


def make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (320, 180))
    assert writer.isOpened()
    for index in range(12):
        frame = np.full((180, 320, 3), (24, 28, 32), dtype=np.uint8)
        cv2.rectangle(frame, (30 + index * 8, 50), (100 + index * 8, 140), (40, 130, 240), -1)
        writer.write(frame)
    writer.release()


def main() -> int:
    assert requests.get(f"{BASE_URL}/api/health", timeout=60).status_code == 200
    for existing in requests.get(f"{BASE_URL}/api/projects", timeout=30).json():
        if existing["name"] in {"API smoke test", "Browser QA fixture"}:
            requests.delete(f"{BASE_URL}/api/projects/{existing['project_id']}", timeout=30).raise_for_status()

    project = requests.post(
        f"{BASE_URL}/api/projects", json={"name": "Browser QA fixture"}, timeout=30
    ).json()
    assert requests.get(f"{BASE_URL}/api/projects/{project['project_id']}", timeout=30).status_code == 200

    with tempfile.TemporaryDirectory() as temp:
        video_path = Path(temp) / "smoke.mp4"
        make_video(video_path)
        with video_path.open("rb") as stream:
            response = requests.post(
                f"{BASE_URL}/api/video/upload",
                files={"file": ("smoke.mp4", stream, "video/mp4")},
                timeout=30,
            )
        response.raise_for_status()
        video = response.json()
        project.update(
            {
                "video_id": video["video_id"],
                "active_mode": "storyboard",
                "background_frame_index": 0,
                "selected_frames": [
                    {
                        "id": f"qa-{frame_index}",
                        "frame_index": frame_index,
                        "mask_id": None,
                        "overlay_url": None,
                        "crop": {"x": 20, "y": 30, "width": 150, "height": 120},
                        "effects": {
                            "opacity": 1,
                            "feather": 2,
                            "outline_width": 3,
                            "outline_color": "#ff8a3d",
                        },
                    }
                    for frame_index in (0, 5, 11)
                ],
            }
        )
        response = requests.put(
            f"{BASE_URL}/api/projects/{project['project_id']}", json=project, timeout=30
        )
        response.raise_for_status()

        response = requests.post(
            f"{BASE_URL}/api/storyboard",
            json={
                "frames": [
                    {
                        "video_id": video["video_id"],
                        "frame_index": frame_index,
                        "crop": {"x": 20, "y": 30, "width": 150, "height": 120},
                    }
                    for frame_index in (0, 5, 11)
                ],
                "gap": 8,
                "gap_color": "#ffffff",
            },
            timeout=30,
        )
        response.raise_for_status()
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 1000

    requests.delete(f"{BASE_URL}/api/projects/{project['project_id']}", timeout=30).raise_for_status()
    print("API smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
