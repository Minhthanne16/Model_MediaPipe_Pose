import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from motion_features import angle_deg, measure_side, extract_bilateral


class MotionTests(unittest.TestCase):
    def test_angles(self):
        for vector, expected in [((1, 0, 0), 0), ((0, 1, 0), 90), ((-1, 0, 0), 180)]:
            self.assertAlmostEqual(angle_deg((1, 0, 0), vector), expected)
        self.assertIsNone(angle_deg((0, 0, 0), (1, 0, 0)))
        self.assertIsNone(angle_deg((math.nan, 0, 0), (1, 0, 0)))
        self.assertAlmostEqual(angle_deg((1, 1, 1), (1, 1, 1)), 0)

    def test_geometry_both_sides(self):
        joints = dict(left_shoulder=(-1, 0, 0), right_shoulder=(1, 0, 0),
                      left_hip=(-1, 1, 0), right_hip=(1, 1, 0),
                      left_elbow=(-2, 0, 0), left_wrist=(-3, 0, 0),
                      right_elbow=(1, 0.5, 0), right_wrist=(1, 1, 0))
        left, right = measure_side(joints, 'left'), measure_side(joints, 'right')
        self.assertTrue(left['valid'])
        self.assertEqual(left['rom_deg'], 90)
        self.assertEqual(right['rom_deg'], 0)
        self.assertEqual(left['elbow_flexion_deg'], 0)
        self.assertEqual(left['trunk_lateral_deg'], 0)
        joints['left_wrist'] = (-2, 1, 0)
        self.assertEqual(measure_side(joints, 'left')['elbow_flexion_deg'], 90)
        joints['left_hip'] = (-2, 1, 0)
        joints['right_hip'] = (0, 1, 0)
        self.assertEqual(measure_side(joints, 'left')['trunk_lateral_deg'], 45)
        del joints['left_wrist']
        self.assertFalse(measure_side(joints, 'left')['valid'])
        self.assertTrue(measure_side(joints, 'right')['valid'])

    def test_missing_pose(self):
        result = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[])
        measured = extract_bilateral(result, {}, {})
        for values in measured.values():
            self.assertFalse(values['valid'])
            self.assertIsNone(values['rom_deg'])


if __name__ == '__main__':
    unittest.main()
