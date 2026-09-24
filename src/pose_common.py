"""Pose-only POC; anatomical left/right, no clinical evaluation."""
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
CONNECTIONS = ((7, 11), (8, 12), (11, 12), (11, 13), (13, 15),
               (12, 14), (14, 16), (11, 23), (12, 24), (23, 24))
JOINTS = {0: 'nose', 7: 'left_ear', 8: 'right_ear',
          11: 'left_shoulder', 12: 'right_shoulder',
          13: 'left_elbow', 14: 'right_elbow',
          15: 'left_wrist', 16: 'right_wrist',
          23: 'left_hip', 24: 'right_hip'}


def load_config(path=ROOT / 'configs' / 'pose.json'):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    for key in ('min_pose_detection_confidence', 'min_pose_presence_confidence',
                'min_tracking_confidence', 'visibility_threshold', 'presence_threshold'):
        if not 0 <= config[key] <= 1:
            raise ValueError(f'{key} must be within [0, 1]')
    if config['num_poses'] != 1:
        raise ValueError('This POC supports exactly one person')
    if not math.isfinite(config['fallback_fps']) or config['fallback_fps'] <= 0:
        raise ValueError('fallback_fps must be finite and positive')
    if len(config['codec']) != 4:
        raise ValueError('codec must contain four characters')
    return config


def create_landmarker(config, mode):
    model = ROOT / config['model_path']
    if not model.is_file():
        raise FileNotFoundError(f'Model not found: {model}')
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mode, num_poses=config['num_poses'],
        min_pose_detection_confidence=config['min_pose_detection_confidence'],
        min_pose_presence_confidence=config['min_pose_presence_confidence'],
        min_tracking_confidence=config['min_tracking_confidence'],
        output_segmentation_masks=False)
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def timestamp_ms(index, fps, previous):
    """CFR timeline in milliseconds; strictly increasing even at high FPS."""
    return max(previous + 1, round(index * 1000 / fps))


def visible(point, config):
    values = (point.x, point.y, point.visibility, point.presence)
    return (all(v is not None and math.isfinite(v) for v in values)
            and 0 <= point.x < 1 and 0 <= point.y < 1
            and point.visibility >= config['visibility_threshold']
            and point.presence >= config['presence_threshold'])


def draw_upper_body(frame, landmarks, config):
    """Draw image coordinates only, without mirroring or clamping off-screen joints."""
    height, width = frame.shape[:2]
    points = {i: (int(landmarks[i].x * width), int(landmarks[i].y * height))
              for i in JOINTS if visible(landmarks[i], config)}
    for a, b in CONNECTIONS:
        if a in points and b in points:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 3)
    for i, point in points.items():
        color = (0, 0, 255) if i in (11, 12) else (255, 100, 0)
        cv2.circle(frame, point, 6, color, -1)
        cv2.putText(frame, str(i), (point[0] + 7, point[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
