"""Geometric measurements only; world XYZ in metres, anatomical sides.

ROM and elbow use 3D vectors. Trunk lateral is projected onto world XY;
it assumes an upright, frontal camera and is not a clinical diagnosis.
"""
import math


def angle_deg(a, b):
    """3D angle in degrees; undefined/degenerate vectors return None."""
    if len(a) != 3 or len(b) != 3:
        return None
    if not all(math.isfinite(v) for v in (*a, *b)):
        return None
    na, nb = math.hypot(*a), math.hypot(*b)
    if na <= 1e-12 or nb <= 1e-12:
        return None
    cosine = sum((x / na) * (y / nb) for x, y in zip(a, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def measure_side(joints, side):
    """Consume semantic world-metre joints; caller quality-gates coordinates."""
    if side not in ('left', 'right'):
        raise ValueError('side must be left or right')
    result = dict(rom_deg=None, elbow_angle_deg=None, elbow_flexion_deg=None,
                  trunk_lateral_deg=None, valid=False)
    required = [f'{side}_{name}' for name in ('shoulder', 'elbow', 'wrist', 'hip')]
    required += ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
    if any(joints.get(name) is None for name in required):
        return result
    shoulder, elbow, wrist, hip = (joints[name] for name in required[:4])
    rom = angle_deg(subtract(hip, shoulder), subtract(elbow, shoulder))
    elbow_angle = angle_deg(subtract(shoulder, elbow), subtract(wrist, elbow))
    spine = tuple((joints['left_shoulder'][i] + joints['right_shoulder'][i]
                   - joints['left_hip'][i] - joints['right_hip'][i]) / 2
                  for i in range(3))
    if rom is None or elbow_angle is None or not all(math.isfinite(v) for v in spine):
        return result
    if math.hypot(spine[0], spine[1]) <= 1e-12:
        return result
    result.update(rom_deg=rom, elbow_angle_deg=elbow_angle,
                  elbow_flexion_deg=180 - elbow_angle,
                  trunk_lateral_deg=math.degrees(math.atan2(abs(spine[0]), abs(spine[1]))),
                  valid=True)
    return result


def extract_bilateral(result, config, joint_names):
    """Adapt MediaPipe world landmarks; use image visibility/presence as gate.

    No automatic active-side inference. Both sides are measured separately.
    """
    joints = {}
    if result.pose_landmarks and result.pose_world_landmarks:
        image, world = result.pose_landmarks[0], result.pose_world_landmarks[0]
        for index, name in joint_names.items():
            if index >= len(image) or index >= len(world):
                continue
            p, w = image[index], world[index]
            values = (p.x, p.y, p.visibility, p.presence, w.x, w.y, w.z)
            if not all(v is not None and math.isfinite(v) for v in values):
                continue
            if (0 <= p.x < 1 and 0 <= p.y < 1
                    and p.visibility >= config['visibility_threshold']
                    and p.presence >= config['presence_threshold']):
                joints[name] = (w.x, w.y, w.z)
    return {side: measure_side(joints, side) for side in ('left', 'right')}
