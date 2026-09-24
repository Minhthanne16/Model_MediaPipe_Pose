"""Load official Full model and run a blank-image inference (not accuracy validation)."""
import hashlib
import json
import struct
import sys

import cv2
import mediapipe as mp
import numpy as np

from pose_common import ROOT, create_landmarker, load_config


def main():
    config = load_config()
    model = ROOT / config['model_path']
    manifest = json.loads((ROOT / 'models/manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise RuntimeError('Model checksum does not match manifest')
    assert struct.calcsize('P') * 8 == 64
    with create_landmarker(config, mp.tasks.vision.RunningMode.IMAGE) as detector:
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=np.zeros((256, 256, 3), dtype=np.uint8)))
        assert not result.pose_landmarks, 'Unexpected pose in blank image'
    print(f'Python: {sys.version}; 64-bit')
    print(f'MediaPipe: {mp.__version__}; OpenCV: {cv2.__version__}')
    print(f'Model: {model}; SHA256: {digest}')
    print('SUCCESS: Pose Landmarker Full loaded; blank-image inference passed.')


if __name__ == '__main__':
    main()
