from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_demo_video(path: str = "workspace/uploads/demo_motion.mp4", seconds: float = 6.0, fps: int = 24) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 360
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frames = int(seconds * fps)
    for i in range(frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (22, 24, 28)
        cv2.rectangle(frame, (40, 260), (600, 330), (50, 50, 54), -1)
        if t < 2.2:
            x = int(80 + t * 110)
            cv2.circle(frame, (x, 250), 22, (210, 210, 210), -1)
            cv2.rectangle(frame, (x - 12, 272), (x + 12, 315), (180, 180, 180), -1)
        elif t < 3.5:
            angle = min(70, int((t - 2.2) * 90))
            cv2.rectangle(frame, (420, 110), (435, 315), (140, 95, 40), -1)
            pts = np.array([[435, 110], [435 + angle, 125], [435 + angle, 315], [435, 315]], np.int32)
            cv2.fillPoly(frame, [pts], (120, 70, 30))
        else:
            x = int(500 - (t - 3.5) * 70)
            cv2.rectangle(frame, (x, 220), (x + 55, 280), (60, 130, 230), -1)
        writer.write(frame)
    writer.release()
    return str(out_path)


if __name__ == "__main__":
    print(create_demo_video())
