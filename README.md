# MediaPipe Pose Landmarker Full — Phân tích bài tập Shoulder Abduction

Pipeline trích xuất tư thế 3D (33 landmarks) và tính toán các góc đặc trưng vận động cho bài tập phục hồi chức năng **Standing Shoulder Abduction** bằng Google MediaPipe Pose Landmarker Full chính thức.

Hỗ trợ chạy thời gian thực qua **Camera/Webcam/DroidCam** và xử lý ngoại tuyến qua **file Video (MP4)**, tự động xuất video trực quan hóa kèm bảng dữ liệu tọa độ và đặc trưng góc (CSV/JSON).

---

## 📋 Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Hướng dẫn cài đặt từ đầu](#2-hướng-dẫn-cài-đặt-từ-đầu)
3. [Kiểm tra môi trường và Model](#3-kiểm-tra-môi-trường-và-model)
4. [Hướng dẫn chạy chương trình](#4-hướng-dẫn-chạy-chương-trình)
   - [4.1 Chạy trực tiếp từ Camera / Webcam / DroidCam](#41-chạy-trực-tiếp-từ-camera--webcam--droidcam)
   - [4.2 Chạy phân tích từ file Video](#42-chạy-phân-tích-từ-file-video)
5. [Cấu trúc dữ liệu đầu ra (Output)](#5-cấu-trúc-dữ-liệu-đầu-ra-output)
6. [Cấu hình tham số (Configuration)](#6-cấu-hình-tham-số-configuration)
7. [Cấu trúc thư mục dự án](#7-cấu-trúc-thư-mục-dự-án)
8. [Tài liệu tham khảo chuyên sâu](#8-tài-liệu-tham-khảo-chuyên-sâu)
9. [Xử lý lỗi thường gặp (Troubleshooting)](#9-xử-lý-lỗi-thường-gặp-troubleshooting)

---

## 1. Yêu cầu hệ thống

- **Hệ điều hành:** Windows 10/11 (64-bit), Ubuntu/Linux hoặc macOS.
- **Python:** Phiên bản **3.10** hoặc **3.11** (khuyến nghị **Python 3.11 x64**).
- **Git:** Đã cài đặt trên máy.
- **Phần cứng:** CPU tiêu chuẩn (tốc độ suy luận CPU đạt ~90–100 FPS đối với model Full).

---

## 2. Hướng dẫn cài đặt từ đầu

### Bước 1: Clone mã nguồn về máy

Mở Terminal / PowerShell và chạy:

```bash
git clone https://github.com/Minhthanne16/Model_MediaPipe_Pose.git
cd Model_MediaPipe_Pose
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)

Khuyến nghị sử dụng môi trường ảo riêng biệt để tránh xung đột thư viện:

- **Trên Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Nếu gặp lỗi Execution Policy trên PowerShell, chạy lệnh: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` rồi kích hoạt lại).*

- **Trên Windows (Command Prompt - CMD):**
  ```cmd
  python -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

- **Trên Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

Khi kích hoạt thành công, bạn sẽ thấy tiền tố `(.venv)` xuất hiện ở đầu dòng lệnh terminal.

### Bước 3: Cài đặt các thư viện phụ thuộc

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Tải mô hình Pose Landmarker Full chính thức

File trọng số mô hình `.task` (~9.4 MB) không lưu trực tiếp trên Git repo. Chạy script sau để tự động tải model từ Google Cloud Storage và kiểm tra toàn vẹn mã SHA-256:

```bash
python scripts/prepare_assets.py
```

Khi tải thành công, file `pose_landmarker_full.task` sẽ được lưu tại thư mục `models/` và xác nhận khớp với `models/manifest.json`.

---

## 3. Kiểm tra môi trường và Model

Chạy script kiểm tra tự động để xác nhận mô hình nạp thành công và suy luận được trên ảnh thử nghiệm:

```bash
python src/check_install.py
```

Nếu cài đặt thành công, màn hình sẽ hiển thị:
```text
Python: 3.11.x; 64-bit
MediaPipe: 1.0.1; OpenCV: 5.0.0.x
Model: .../models/pose_landmarker_full.task; SHA256: 5134a3aa...
SUCCESS: Pose Landmarker Full loaded; blank-image inference passed.
```

Chạy toàn bộ unit tests:

```bash
python -m unittest discover tests -v
```

---

## 4. Hướng dẫn chạy chương trình

### 4.1 Chạy trực tiếp từ Camera / Webcam / DroidCam

Script: `src/run_camera.py`.

#### A. Kiểm tra kết nối Camera trước (không qua MediaPipe)
Kiểm tra xem camera/DroidCam có nhận tín hiệu và hiển thị hình ảnh bình thường không:

```bash
python src/run_camera.py --preview-only
```

#### B. Chạy nhận diện tư thế Realtime
Bật camera và suy luận tư thế trực tiếp:

```bash
python src/run_camera.py
```

**Các tham số tùy chọn hữu ích:**
- `--camera <index>`: Chỉ số camera (mặc định: `0`). Nếu dùng DroidCam hoặc gắn nhiều webcam, thử index `1`, `2`,...
- `--backend {msmf, dshow, auto}`:
  - Trên Windows 11 / DroidCam: khuyến nghị `--backend msmf` (mặc định) hoặc `--backend dshow`.
- `--width <w> --height <h> --fps <fps>`: Thiết lập độ phân giải và FPS (mặc định: `640x480` ở `30 FPS`).
- **Phím tắt điều khiển:** Nhấn phím `Q` hoặc `Esc` trên cửa sổ camera để thoát.

---

### 4.2 Chạy phân tích từ file Video

Script: `src/run_video.py`.

Chuẩn bị một video người thực hiện bài tập Shoulder Abduction (định dạng MP4/AVI, camera đặt chính diện người tập).

#### A. Chạy phân tích kèm cửa sổ Preview trực tiếp

```bash
python src/run_video.py --input path/to/your_video.mp4
```

*(Nếu không truyền `--input`, chương trình sẽ mặc định tìm file `input/shoulder_abduction.mp4`).*

#### B. Chạy chế độ nền (Headless - Không mở cửa sổ hiển thị)
Chế độ này xử lý nhanh nhất và thích hợp khi phân tích hàng loạt:

```bash
python src/run_video.py --input path/to/your_video.mp4 --no-preview
```

---

## 5. Cấu trúc dữ liệu đầu ra (Output)

Mỗi lần chạy phân tích video, chương trình tự động tạo một thư mục con theo mốc thời gian tại `output/YYYYMMDD_HHMMSS_xxxxxx/` chứa 4 tệp kết quả:

| Tệp kết quả | Định dạng | Nội dung chi tiết |
|---|---|---|
| `pose_result.mp4` | Video | Video đã vẽ khung xương thân trên và hiển thị trực tiếp các thông số góc thời gian thực hai bên trái/phải. |
| `pose_result.features.csv` | CSV | Bảng đặc trưng góc từng khung hình (2 dòng/frame cho bên Left và Right):<br>• `rom_deg`: Góc dang khớp vai 3D.<br>• `elbow_angle_deg`: Góc khớp khuỷu 3D.<br>• `elbow_flexion_deg`: Độ gập khuỷu (`180° - elbow_angle_deg`).<br>• `trunk_lateral_deg`: Độ nghiêng thân sang bên trên mặt phẳng XY.<br>• `valid`: Trạng thái khung hình đủ độ tin cậy để tính toán (`True/False`). |
| `pose_result.landmarks.csv` | CSV | Dữ liệu tọa độ 33 khớp giải phẫu cho từng frame theo 2 hệ tọa độ riêng biệt:<br>• `normalized_image`: Tọa độ ảnh chuẩn hóa $[0, 1]$ kèm `visibility` và `presence`.<br>• `world_m`: Tọa độ 3D theo mét thực với gốc tọa độ đặt tại trung điểm hai hông. |
| `pose_result.json` | JSON | Thống kê kỹ thuật: Cấu hình sử dụng, tổng số frame, số frame phát hiện pose, tỷ lệ phát hiện (%), FPS suy luận model (CPU), và FPS toàn chu trình (End-to-End). |

---

## 6. Cấu hình tham số (Configuration)

File cấu hình chính: `configs/pose.json`.

```json
{
  "model_path": "models/pose_landmarker_full.task",
  "num_poses": 1,
  "min_pose_detection_confidence": 0.5,
  "min_pose_presence_confidence": 0.5,
  "min_tracking_confidence": 0.5,
  "output_segmentation_masks": false,
  "fallback_fps": 30.0
}
```

- `num_poses = 1`: Tối ưu hóa suy luận cho 1 người duy nhất trong khung hình.
- `output_segmentation_masks = false`: Tắt mặt nạ tách người để giảm tải CPU/RAM và tăng FPS.
- `confidence`: Ngưỡng tin cậy phát hiện/theo dõi điểm khớp (không phải ngưỡng đánh giá bài tập).

---

## 7. Cấu trúc thư mục dự án

```text
Model_MediaPipe_Pose/
├── configs/
│   └── pose.json                  # File cấu hình mô hình và tham số suy luận
├── docs/
│   └── HUONG_DAN_SU_DUNG_MODEL_VA_THUAT_TOAN.md  # Tài liệu toán học & thuật toán chi tiết
├── models/
│   ├── manifest.json              # Thông tin nguồn tải, phiên bản và SHA-256 model
│   └── pose_landmarker_full.task  # Trọng số mô hình tải về (bị gitignore)
├── reports/
│   └── runs/                      # Nhật ký kiểm tra cài đặt và test
├── scripts/
│   └── prepare_assets.py          # Script tự động tải và kiểm tra model
├── src/
│   ├── check_install.py           # Kiểm tra tính toàn vẹn mô hình và thư viện
│   ├── motion_features.py         # Module trích xuất vector góc: ROM, Khuỷu, Thân
│   ├── pose_common.py             # Cấu hình chung, vẽ skeleton, nạp model
│   ├── run_camera.py              # Ứng dụng chạy nhận diện trực tiếp qua Camera/Webcam
│   └── run_video.py               # Ứng dụng phân tích video và xuất CSV/MP4
├── tests/
│   ├── test_motion_features.py    # Unit test kiểm tra công thức vector góc
│   └── test_pose.py               # Unit test kiểm tra plumbing model và pipeline
├── .gitignore                     # Bỏ qua .venv, models/*.task, input, output
├── AGENTS.md                      # Đặc tả kỹ thuật kiến trúc hệ thống tổng thể
├── README.md                      # Hướng dẫn sử dụng tổng quan
└── requirements.txt               # Danh sách thư viện Python phụ thuộc
```

---

## 8. Tài liệu tham khảo chuyên sâu

Để tìm hiểu chi tiết về cơ sở toán học, cách trích xuất góc giải phẫu, thuật toán máy trạng thái đếm số lần lặp (**Repetition FSM**), thuật toán so khớp chuỗi thời gian (**DTW**), phương pháp đánh giá lỗi bài tập (**Fixed Threshold**), và kiến trúc triển khai ứng dụng di động Android:

👉 **Xem tài liệu chi tiết tại:** [`docs/HUONG_DAN_SU_DUNG_MODEL_VA_THUAT_TOAN.md`](docs/HUONG_DAN_SU_DUNG_MODEL_VA_THUAT_TOAN.md)  
👉 **Đặc tả kiến trúc chuẩn đồ án:** [`AGENTS.md`](AGENTS.md)

---

## 9. Xử lý lỗi thường gặp (Troubleshooting)

1. **Lỗi `Cannot open camera` khi chạy `run_camera.py`:**
   - Đảm bảo ứng dụng camera khác (hoặc trình duyệt) không chiếm dụng webcam.
   - Nếu dùng DroidCam: đảm bảo DroidCam Client trên Windows đang hiển thị hình ảnh trước khi chạy script.
   - Thử đổi chỉ số camera: `python src/run_camera.py --camera 1` hoặc `--camera 2`.
   - Thử đổi backend: `python src/run_camera.py --backend dshow`.

2. **Lỗi `ExecutionPolicy` khi kích hoạt `.venv` trên Windows PowerShell:**
   - Chạy lệnh sau trong PowerShell quyền thông thường:
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
     ```
   - Sau đó chạy lại: `.\.venv\Scripts\Activate.ps1`.

3. **Lỗi DLL load failed hoặc Application Control / Antivirus chặn file:**
   - Một số bản Windows bật Smart App Control có thể chặn các file `.pyd` mới biên dịch. Đảm bảo sử dụng bản Python x64 chính thức tải từ `python.org`.

---

## 📄 Bản quyền và Ghi nhận

- Dự án sử dụng mô hình **MediaPipe Pose Landmarker Full** bản quyền của Google (Apache License 2.0).
- Hệ thống được phát triển phục vụ đề tài nghiên cứu phân tích động tác phục hồi chức năng qua camera đơn mắt.


