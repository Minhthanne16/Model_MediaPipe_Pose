"""Download official model and record/verify its provenance."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'


def main():
    for folder in ('models', 'input', 'output', 'reports/runs'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    model = ROOT / 'models/pose_landmarker_full.task'
    manifest = ROOT / 'models/manifest.json'

    # If model already exists, verify its checksum
    if model.exists():
        current_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
        if manifest.exists():
            data = json.loads(manifest.read_text(encoding='utf-8'))
            expected = data.get('sha256')
            if current_sha256 == expected:
                print(f'[OK] Model already exists and matches manifest SHA256: {current_sha256}')
                return
            print(f'[WARN] Model checksum mismatch (expected: {expected}, got: {current_sha256}). Re-downloading...')
        else:
            print(f'[OK] Model exists ({len(model.read_bytes())} bytes). Creating manifest...')
            manifest.write_text(json.dumps(dict(url=URL, variant='Full', revision=1,
                downloaded_utc=datetime.now(timezone.utc).isoformat(), bytes=model.stat().st_size,
                sha256=current_sha256), indent=2), encoding='utf-8')
            return

    # Download model
    print(f'Downloading MediaPipe Pose Landmarker Full model from:\n  {URL}...')
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    if len(data) < 1000000:
        raise RuntimeError('Unexpectedly small model download')

    downloaded_sha256 = hashlib.sha256(data).hexdigest()
    if manifest.exists():
        expected = json.loads(manifest.read_text(encoding='utf-8')).get('sha256')
        if expected and downloaded_sha256 != expected:
            raise RuntimeError(f'Downloaded model SHA256 ({downloaded_sha256}) does not match manifest ({expected})')

    model.write_bytes(data)
    manifest.write_text(json.dumps(dict(url=URL, variant='Full', revision=1,
        downloaded_utc=datetime.now(timezone.utc).isoformat(), bytes=len(data),
        sha256=downloaded_sha256), indent=2), encoding='utf-8')
    print(f'[OK] Successfully downloaded model ({len(data):,} bytes).')
    print(f'[OK] SHA256: {downloaded_sha256}')


if __name__ == '__main__':
    main()

