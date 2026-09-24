"""Offline single-person video overlay. Does not evaluate exercise quality."""
import argparse
import csv
from contextlib import ExitStack
from motion_features import extract_bilateral
from pose_common import JOINTS
from datetime import datetime
import json
import math
from pathlib import Path
import time

import cv2
import mediapipe as mp

from pose_common import ROOT, create_landmarker, draw_upper_body, load_config, timestamp_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'input/shoulder_abduction.mp4')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pose.json')
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    source = args.input.resolve()
    output = (args.output or ROOT / 'output' / datetime.now().strftime('%Y%m%d_%H%M%S_%f') / 'pose_result.mp4').resolve()
    if not source.is_file():
        raise FileNotFoundError(f'Input video not found: {source}')
    if output == source or output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    capture = cv2.VideoCapture(str(source))
    writer = None
    count = detected = 0
    inference_total = 0.0
    previous = -1
    preview = not args.no_preview
    started = time.perf_counter()
    stopped_early = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f'Cannot open video: {source}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = config['fallback_fps']
            print(f'WARNING: Invalid input FPS; using {fps}')
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        with ExitStack() as stack:
            output.parent.mkdir(parents=True, exist_ok=True)
            feature_path = output.with_suffix('.features.csv')
            landmark_path = output.with_suffix('.landmarks.csv')
            feature_file = stack.enter_context(feature_path.open('x', newline='', encoding='utf-8'))
            landmark_file = stack.enter_context(landmark_path.open('x', newline='', encoding='utf-8'))
            features = csv.DictWriter(feature_file, fieldnames=[
                'frame_index', 'timestamp_ms', 'side', 'rom_deg', 'elbow_angle_deg',
                'elbow_flexion_deg', 'trunk_lateral_deg', 'valid'])
            features.writeheader()
            landmarks = csv.writer(landmark_file)
            landmarks.writerow(['frame_index', 'timestamp_ms', 'coordinate_space',
                                'landmark_id', 'x', 'y', 'z', 'visibility', 'presence'])
            landmarker = stack.enter_context(create_landmarker(config, mp.tasks.vision.RunningMode.VIDEO))
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    output.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*config['codec']), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError('Cannot create output video; check codec in config')
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                previous = timestamp_ms(count, fps, previous)
                start = time.perf_counter()
                result = landmarker.detect_for_video(image, previous)
                elapsed = (time.perf_counter() - start) * 1000
                inference_total += elapsed
                measurements = extract_bilateral(result, config, JOINTS)
                for side, values in measurements.items():
                    features.writerow(dict(frame_index=count, timestamp_ms=previous, side=side, **values))
                for space, poses in [('normalized_image', result.pose_landmarks),
                                     ('world_m', result.pose_world_landmarks)]:
                    if not poses:
                        landmarks.writerow([count, previous, space, '', '', '', '', '', ''])
                    else:
                        for index, point in enumerate(poses[0]):
                            raw = [point.x, point.y, point.z, point.visibility, point.presence]
                            clean = [v if v is not None and math.isfinite(v) else None for v in raw]
                            landmarks.writerow([count, previous, space, index, *clean])
                for row, (side, values) in enumerate(measurements.items()):
                    text = f'{side}: insufficient_data'
                    if values['valid']:
                        text = (f"{side}: ROM {values['rom_deg']:.1f} | "
                                f"elbow flex {values['elbow_flexion_deg']:.1f} | "
                                f"trunk {values['trunk_lateral_deg']:.1f} deg")
                    cv2.putText(frame, text, (20, 65 + row * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                has_pose = bool(result.pose_landmarks)
                if has_pose:
                    detected += 1
                    draw_upper_body(frame, result.pose_landmarks[0], config)
                status = 'POSE DETECTED' if has_pose else 'NO POSE'
                cv2.putText(frame, f'{status} | inference {elapsed:.1f} ms', (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if has_pose else (0, 0, 255), 2)
                writer.write(frame)
                count += 1
                if preview:
                    try:
                        cv2.imshow('MediaPipe Full - Q to stop', frame)
                        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                            stopped_early = True
                            break
                    except cv2.error as error:
                        print(f'WARNING: Preview disabled; processing continues: {error}')
                        preview = False
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    if not count:
        raise RuntimeError('No decodable frames; output was not produced')
    if expected > 0 and count < expected and not stopped_early:
        print(f'WARNING: Decoding ended at {count}/{expected} frames')
    probe = cv2.VideoCapture(str(output))
    try:
        readable, _ = probe.read()
        output_count = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        probe.release()
    if not readable or output_count != count:
        raise RuntimeError(f'Output verification failed: {output_count} vs {count} frames')
    duration = time.perf_counter() - started
    report = dict(input=str(source), output=str(output), frames=count,
                  detected_frames=detected, detection_ratio=detected / count,
                  mean_inference_ms=inference_total / count,
                  inference_fps=1000 * count / inference_total if inference_total else None,
                  end_to_end_fps=count / duration, stopped_early=stopped_early,
                  timestamp_policy='CFR frame index / reported FPS; VFR not preserved',
                  audio_preserved=False, config=config,
                  mediapipe=mp.__version__, opencv=cv2.__version__)
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
