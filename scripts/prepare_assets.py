"""Download official model and record its provenance; never replace an existing model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'


def main():
    for folder in ('models', 'input', 'output', 'reports/runs'):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    model = ROOT / 'models/pose_landmarker_full.task'
    manifest = ROOT / 'models/manifest.json'
    if model.exists() or manifest.exists():
        raise FileExistsError('Existing model/manifest: verify rather than overwrite')
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    if len(data) < 1000000:
        raise RuntimeError('Unexpectedly small model download')
    model.write_bytes(data)
    manifest.write_text(json.dumps(dict(url=URL, variant='Full', revision=1,
        downloaded_utc=datetime.now(timezone.utc).isoformat(), bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest()), indent=2), encoding='utf-8')
    print(manifest.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
