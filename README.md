# MediaPipe Pose Landmarker Full — PC proof-of-concept

## Cập nhật: xuất landmarks và đo góc hai bên (video)

Chạy video hiện có:

```powershell
& 'D:\Learn\Model_Mediapipe_Pose\.venv\Scripts\python.exe' 'D:\Learn\Model_Mediapipe_Pose\src\run_video.py' --input 'D:\Learn\Model_Mediapipe_Pose\input\input1.mp4'
```

Trong thư mục lần chạy dưới `D:\Learn\Model_Mediapipe_Pose\output`:
- `pose_result.mp4`: skeleton và góc trái/phải.
- `pose_result.features.csv`: hai dòng/frame; ROM vai 3D, góc khuỷu 3D,
  elbow flexion = 180 - elbow angle, trunk lateral trên world XY; đơn vị độ.
- `pose_result.landmarks.csv`: 33 điểm/frame/hệ tọa độ, normalized_image và world_m
  được lưu riêng bằng cột coordinate_space; kèm visibility/presence thô.
- `pose_result.json`: thống kê lần chạy (không phải dữ liệu landmarks).

CSV tọa độ là dữ liệu thô, không phải mọi điểm đều đủ tin cậy để tính toán.
CSV features dùng confidence từ cấu hình pose; thiếu joint bắt buộc thì valid=False
và các góc trống, không thay bằng 0. Frame mất pose vẫn có dòng trống trong CSV
landmarks. Left/right theo giải phẫu; chưa suy luận tay hoạt động.
Góc là số đo từng frame chưa làm mượt, chưa phải peak ROM theo repetition.
Camera cần chính diện và thẳng đứng; world coordinates là ước lượng đơn mắt.
Chưa tách repetition, chưa chạy DTW hoặc kết luận bù trừ/đúng sai; cần mẫu chuẩn
và hiệu chỉnh ngưỡng trước khi đánh giá. Camera trực tiếp chưa xuất CSV/tính góc.
Các mô tả pose-only bên dưới là trạng thái trước cập nhật này.


Thư mục: `D:\Learn\Model_Mediapipe_Pose`.
Đã đọc và sao chép AGENTS.md của đồ án. Đây là luồng B: cài pose estimator,
không phải pipeline nghiên cứu UI-PRMD, không tính ROM/FSM/DTW hoặc kết luận y khoa.

## Trạng thái hiện tại

- Python 3.11.9 Windows x64 tại `D:\Learn\Model_Mediapipe_Pose\.venv`.
- MediaPipe 1.0.1 và opencv-contrib-python 5.0.0.86 hoạt động bình thường sau khi tắt Smart App Control.
- Model Pose Landmarker Full float16 revision 1 chính thức (`pose_landmarker_full.task`) khớp SHA-256 manifest.
- Kiểm tra cài đặt (`check_install.py`) và toàn bộ 5 unit tests trong `tests/test_pose.py` đều đạt (100% PASS).
- Đã kiểm tra kết nối camera DroidCam qua MSMF: nhận diện pose trực tiếp đạt tốc độ xử lý ~9-10 ms/frame (~100 FPS inference CPU).
- Đã cập nhật `run_camera.py` mặc định dùng `--backend msmf` để tương thích hoàn toàn với DroidCam trên Windows 11.


## Kiểm tra cài đặt

```powershell
& 'D:\Learn\Model_Mediapipe_Pose\.venv\Scripts\python.exe' 'D:\Learn\Model_Mediapipe_Pose\src\check_install.py'
& 'D:\Learn\Model_Mediapipe_Pose\.venv\Scripts\python.exe' -m unittest discover -s 'D:\Learn\Model_Mediapipe_Pose\tests' -v
```

check_install xác minh checksum và inference ảnh đen. Tests gồm mapping trái/phải,
timestamp tăng nghiêm ngặt, confidence/điểm ngoài ảnh, video thiếu và video đen end-to-end.
Video đen chỉ kiểm tra plumbing/NO POSE, không xác thực nhận diện người.

## Chạy video sau khi lỗi môi trường được xử lý

Đặt video tại `D:\Learn\Model_Mediapipe_Pose\input\shoulder_abduction.mp4`.
Quay một người chính diện, nhìn rõ đầu, vai, khuỷu, cổ tay và hông; một tay hoạt động.
Ưu tiên MP4 H.264, 720p/1080p, 30 FPS cố định.

```powershell
& 'D:\Learn\Model_Mediapipe_Pose\.venv\Scripts\python.exe' 'D:\Learn\Model_Mediapipe_Pose\src\run_video.py'
& 'D:\Learn\Model_Mediapipe_Pose\.venv\Scripts\python.exe' 'D:\Learn\Model_Mediapipe_Pose\src\run_video.py' --no-preview
```

Có thể truyền --input và --output bằng đường dẫn tuyệt đối. Mặc định mỗi lần chạy tạo
thư mục timestamp riêng trong `D:\Learn\Model_Mediapipe_Pose\output`, không ghi đè.
Q dừng sớm; output khi đó chỉ chứa các frame đã xử lý. JSON cạnh video lưu config,
phiên bản, số frame, tỷ lệ phát hiện, inference latency/FPS và end-to-end FPS.
End-to-end bao gồm xử lý, preview nếu bật và kiểm tra output; không phải benchmark mobile.

## Quy ước và giới hạn

- Tasks API, CPU, VIDEO, num_poses=1, không segmentation. Không dùng legacy Solutions API.
- Model trả 33 điểm nhưng overlay chọn thân trên, vai 11/12 đỏ.
- Left/right theo cơ thể, không mirror. Chưa xuất canonical frames hoặc landmarks từng frame.
- Cấu hình tại `D:\Learn\Model_Mediapipe_Pose\configs\pose.json`.
  Confidence 0.5 là cấu hình detection/tracking/hiển thị, không phải threshold lâm sàng.
- Không trộn normalized_image và world_m. Chưa tính góc hay suy luận active_side.
- Video xuất không có âm thanh. Timeline CFR theo FPS; không bảo toàn timing VFR.
- Điểm ngoài ảnh hoặc thiếu confidence không được vẽ; không ép về viền ảnh.
- Khi preview lỗi, tiếp tục ghi video. Kiểm tra output đọc được và số frame khớp.
- Các PDF ưu tiên cao, UI-PRMD, thuật toán nghiên cứu và Android chưa được triển khai.
- Chạy cục bộ; script inference không tải hoặc gửi video lên mạng.

## Tái tạo môi trường

Snapshot dependency hiện tại nằm tại `D:\Learn\Model_Mediapipe_Pose\requirements.txt`;
đây là môi trường đã cài nhưng đang bị policy chặn, không phải môi trường đã nghiệm thu.
Trên máy mới có Python 3.11 x64, tạo venv rồi pip install -r requirements.txt.
Script `D:\Learn\Model_Mediapipe_Pose\scripts\prepare_assets.py` tải model revision cố định
và từ chối ghi đè model/manifest hiện có. Giữ manifest để đối chiếu checksum khi tái tạo.

Nguồn: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
và https://pypi.org/project/mediapipe/ .

## DroidCam / webcam PC

Script: `D:\Learn\Model_Mediapipe_Pose\src\run_camera.py`.
Kết nối điện thoại với DroidCam Windows client, kiểm tra có hình trước; nếu dùng
DroidCam OBS, cần Start Virtual Camera trong OBS. Đóng các ứng dụng khác đang dùng
camera, nhưng giữ client/OBS cung cấp webcam hoạt động.

```powershell
& 'D:\Learn\Model_Mediapipe_Pose\.venv\Scripts\python.exe' 'D:\Learn\Model_Mediapipe_Pose\src\run_camera.py' --camera 0 --preview-only
& 'D:\Learn\Model_Mediapipe_Pose\.venv\Scripts\python.exe' 'D:\Learn\Model_Mediapipe_Pose\src\run_camera.py' --camera 0
```

Thử index 1/2/3 nếu 0 không phải DroidCam. Backend mặc định msmf; nếu cần có thể
chọn --backend dshow hoặc --backend auto. --preview-only không import MediaPipe,
chỉ kiểm tra camera thô. Mặc định yêu cầu 640x480/30 FPS phù hợp với DroidCam.
Q/Escape hoặc Ctrl+C để dừng và giải phóng camera.
Không ghi video/âm thanh, không mirror, không xuất landmarks. VIDEO mode đồng bộ
với timestamp monotonic; đây là POC PC, không thay thiết kế LIVE_STREAM native Android.
Chưa kiểm thử webcam vật lý hoặc nhận diện người trên máy này.

