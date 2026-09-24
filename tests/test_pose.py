import json
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
import numpy as np
from pose_common import JOINTS, load_config, timestamp_ms, visible


class PoseTests(unittest.TestCase):
    def test_anatomical_mapping(self):
        self.assertEqual(JOINTS[11], 'left_shoulder')
        self.assertEqual(JOINTS[12], 'right_shoulder')
        self.assertEqual(JOINTS[23], 'left_hip')
        self.assertEqual(JOINTS[24], 'right_hip')

    def test_timestamps(self):
        previous = -1
        for index in range(100):
            current = timestamp_ms(index, 2000, previous)
            self.assertGreater(current, previous)
            previous = current
        self.assertEqual(timestamp_ms(30, 30, 966), 1000)

    def test_visibility(self):
        config = load_config()
        point = SimpleNamespace(x=0.5, y=0.5, visibility=1.0, presence=1.0)
        self.assertTrue(visible(point, config))
        for field, value in [('visibility', None), ('x', 1.2), ('presence', 0.1), ('y', float('nan'))]:
            bad = SimpleNamespace(**vars(point))
            setattr(bad, field, value)
            self.assertFalse(visible(bad, config))

    def test_missing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, str(ROOT / 'src/run_video.py'),
                '--input', str(Path(tmp) / 'missing.mp4'), '--no-preview'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Input video not found', result.stderr)

    def test_blank_video_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'blank.mp4', Path(tmp) / 'result.mp4'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
            self.assertTrue(writer.isOpened())
            try:
                for _ in range(6):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
            finally:
                writer.release()
            result = subprocess.run([sys.executable, str(ROOT / 'src/run_video.py'),
                '--input', str(source), '--output', str(output), '--no-preview'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(report['frames'], 6)
            self.assertEqual(report['detected_frames'], 0)
            self.assertGreater(output.stat().st_size, 0)
            with output.with_suffix('.features.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertEqual({row['side'] for row in rows}, {'left', 'right'})
            self.assertTrue(all(row['valid'] == 'False' and row['rom_deg'] == '' for row in rows))
            with output.with_suffix('.landmarks.csv').open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(row['x'] == '' for row in rows))


if __name__ == '__main__':
    unittest.main()
