"""DroidCam/Windows camera preview and synchronous MediaPipe Full inference."""
import argparse
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=('msmf', 'dshow', 'auto'), default='msmf')
    parser.add_argument('--preview-only', action='store_true', help='Test camera without importing MediaPipe')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/pose.json')
    parser.add_argument('--max-frames', type=int, default=0, help='0 means run until Q/Escape')
    args = parser.parse_args()
    if min(args.width, args.height, args.fps) <= 0 or args.camera < 0 or args.max_frames < 0:
        parser.error('Invalid camera index, dimensions, FPS or frame limit')
    backend = {'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF, 'auto': cv2.CAP_ANY}[args.backend]
    detector = None
    capture = None
    frames = detected = 0
    inference_ms = 0.0
    started = time.perf_counter()
    previous = -1
    title = 'DroidCam preview - Q/Escape' if args.preview_only else 'MediaPipe Full - Q/Escape'
    try:
        if not args.preview_only:
            import mediapipe as mp
            from pose_common import create_landmarker, draw_upper_body, load_config
            config = load_config(args.config)
            detector = create_landmarker(config, mp.tasks.vision.RunningMode.VIDEO)
        capture = cv2.VideoCapture(args.camera, backend)
        if not capture.isOpened():
            raise RuntimeError('Cannot open camera. Check DroidCam, index, backend and Windows camera permissions.')
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        print(f'Camera {args.camera}, backend {capture.getBackendName()}; requested {args.width}x{args.height} at {args.fps} FPS')
        print('No recording. No mirroring by this script. Press Q/Escape in the preview window to stop.')
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        started = time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('Camera stopped delivering frames. Check phone connection and DroidCam client.')
            if frames == 0:
                print(f'Actual frame size: {frame.shape[1]}x{frame.shape[0]}')
            status = 'CAMERA ONLY'
            if detector is not None:
                stamp = max(previous + 1, int((time.perf_counter() - started) * 1000))
                previous = stamp
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                start = time.perf_counter()
                result = detector.detect_for_video(image, stamp)
                elapsed = (time.perf_counter() - start) * 1000
                inference_ms += elapsed
                status = 'NO POSE'
                if result.pose_landmarks:
                    detected += 1
                    draw_upper_body(frame, result.pose_landmarks[0], config)
                    status = 'POSE DETECTED'
                status += f' | inference {elapsed:.1f} ms'
            frames += 1
            fps = frames / max(time.perf_counter() - started, 1e-9)
            cv2.putText(frame, f'{status} | loop {fps:.1f} FPS', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            cv2.imshow(title, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.max_frames and frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print('Stopped by Ctrl+C')
    finally:
        if capture is not None:
            capture.release()
        if detector is not None:
            detector.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        if frames:
            print(f'Frames: {frames}; detected: {detected if not args.preview_only else "not evaluated"}')
            if detector is not None:
                print(f'Mean inference: {inference_ms / frames:.2f} ms')


if __name__ == '__main__':
    main()
