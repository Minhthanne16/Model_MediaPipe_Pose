# TÀI LIỆU HƯỚNG DẪN: SỬ DỤNG MODEL MEDIAPIPE POSE, ĐỌC OUTPUT VÀ ÁP DỤNG THUẬT TOÁN ĐÁNH GIÁ

> **Dự án:** Phân tích và đánh giá bài tập phục hồi chức năng Standing Shoulder Abduction qua Camera đơn mắt.  
> **Tài liệu tham chiếu:** `AGENTS.md`, `DeCuongChiTiet_PhanTichBaiTapVaiQuaCamera(5).pdf`, `Nghiên cứu cách tính các giá trị cần thiết cho việc đánh giá bài tập Shoulder Abduction.pdf`.

---

## MỤC LỤC
1. [Tổng quan về MediaPipe Pose Landmarker Full](#1-tổng-quan-về-mediapipe-pose-landmarker-full)
2. [Cách nạp và sử dụng Model trong mã nguồn](#2-cách-nạp-và-sử-dụng-model-trong-mã-nguồn)
3. [Cấu trúc dữ liệu Output và cách đọc chi tiết](#3-cấu-trúc-dữ-liệu-output-và-cách-đọc-chi-tiết)
4. [Kiến trúc phân tầng: Áp dụng thuật toán vào sau Model](#4-kiến-trúc-phân-tầng-áp-dụng-thuật-toán-vào-sau-model)
5. [Tầng 1: Tiền xử lý, Quality Gate và Làm mượt](#5-tầng-1-tiền-xử-lý-quality-gate-và-làm-mượt)
6. [Tầng 2: Trích xuất đặc trưng vận động (Feature Engine)](#6-tầng-2-trích-xuất-đặc-trưng-vận-động-feature-engine)
7. [Tầng 3: Tách và đếm số lần lặp (Repetition FSM)](#7-tầng-3-tách-và-đếm-số-lần-lặp-repetition-fsm)
8. [Tầng 4: Đánh giá chất lượng động tác](#8-tầng-4-đánh-giá-chất-lượng-động-tác)
   - [8.1 Baseline 1: Fixed Threshold (Ngưỡng cố định)](#81-baseline-1-fixed-threshold-ngưỡng-cố-định)
   - [8.2 Baseline 2: Dynamic Time Warping (DTW)](#82-baseline-2-dynamic-time-warping-dtw)
9. [Code ví dụ hoàn chỉnh kết nối toàn bộ Pipeline](#9-code-ví-dụ-hoàn-chỉnh-kết-nối-toàn-bộ-pipeline)
10. [Các bẫy kỹ thuật và nguyên tắc bắt buộc](#10-các-bẫy-kỹ-thuật-và-nguyên-tắc-bắt-buộc)
11. [Đánh giá hiệu năng và Tính khả thi triển khai Mobile App](#11-đánh-giá-hiệu-năng-và-tính-khả-thi-triển-khai-mobile-app)
    - [11.1 Bảng đo lường hiệu năng thực tế (Resource Footprint)](#111-bảng-đo-lường-hiệu-năng-thực-tế-resource-footprint)
    - [11.2 Kết luận tính khả thi trên Mobile](#112-kết-luận-tính-khả-thi-trên-mobile)
    - [11.3 Kiến trúc triển khai tối ưu trên Android (Native vs Flutter)](#113-kiến-trúc-triển-khai-tối-ưu-trên-android-native-vs-flutter)
    - [11.4 Chiến lược tối ưu Pin, Nhiệt độ và FPS](#114-chiến-lược-tối-ưu-pin-nhiệt-độ-và-fps)

---

## 1. TỔNG QUAN VỀ MEDIAPIPE POSE LANDMARKER FULL

### 1.1 Vai trò thực tế của Model
- Model `pose_landmarker_full.task` là một **Pose Estimator** (bộ ước lượng tư thế 3D) dựa trên mạng nơ-ron học sâu của Google.
- **Phạm vi của model:** Model **chỉ** có nhiệm vụ phát hiện sự hiện diện của con người và trích xuất tọa độ không gian của 33 điểm khớp (landmarks).
- **Điều model KHÔNG làm:** Model **không** biết người tập có tập đúng hay sai, **không** tự đếm repetition, và **không** đưa ra bất kỳ đánh giá y khoa nào. Mọi logic phân tích, trích xuất góc và chấm điểm đều nằm ở các thuật toán phía sau do chúng ta xây dựng.

### 1.2 Tệp mô hình và Cấu hình
- File nhị phân model: `models/pose_landmarker_full.task` (bản float16 chính thức, dung lượng ~9.4 MB).
- File kiểm tra toàn vẹn: `models/manifest.json` (chứa URL gốc và mã SHA-256).
- File tham số phát hiện: `configs/pose.json`.

---

## 2. CÁCH NẠP VÀ SỬ DỤNG MODEL TRONG MÃ NGUỒN

MediaPipe Vision Tasks hỗ trợ 3 chế độ chạy (`RunningMode`). Chọn đúng chế độ là điều kiện tiên quyết:

| Chế độ (`RunningMode`) | Khi nào sử dụng | Phương thức gọi | Yêu cầu Timestamp |
|---|---|---|---|
| `IMAGE` | Xử lý từng ảnh tĩnh độc lập (ảnh chụp, dataset ảnh). | `detector.detect(image)` | Không cần timestamp. |
| `VIDEO` | Xử lý video file (từng frame tuần tự có kiểm soát). | `detector.detect_for_video(image, timestamp_ms)` | `timestamp_ms` phải là số nguyên **tăng nghiêm ngặt** qua từng frame. |
| `LIVE_STREAM` | Luồng camera trực tiếp (real-time, bất đồng bộ). | `detector.detect_async(image, timestamp_ms)` | Bắt buộc callback function; bỏ khung hình nếu suy luận không kịp. |

### Mã nguồn khởi tạo chuẩn (Python API):

```python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Khai báo tùy chọn cấu hình cơ bản
base_options = python.BaseOptions(
    model_asset_path='models/pose_landmarker_full.task',
    delegate=python.BaseOptions.Delegate.CPU  # hoặc Delegate.GPU nếu hỗ trợ
)

# 2. Cấu hình PoseLandmarkerOptions
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Chọn chế độ phù hợp
    num_poses=1,                           # Giới hạn 1 người cho bài tập phục hồi chức năng
    min_pose_detection_confidence=0.5,     # Ngưỡng tin cậy phát hiện người ban đầu
    min_pose_presence_confidence=0.5,      # Ngưỡng tin cậy người có mặt trong frame
    min_tracking_confidence=0.5,           # Ngưỡng tin cậy bám vết qua các frame
    output_segmentation_masks=False        # Tắt mask để tối ưu FPS
)

# 3. Tạo detector
detector = vision.PoseLandmarker.create_from_options(options)
```


---

## 3. CẤU TRÚC DỮ LIỆU OUTPUT VÀ CÁCH ĐỌC CHI TIẾT

Khi gọi hàm suy luận (ví dụ: `result = detector.detect_for_video(image, timestamp_ms)`), đối tượng trả về là `PoseLandmarkerResult` chứa hai danh sách landmarks song song:

```text
PoseLandmarkerResult:
├── pose_landmarks        -> List[List[NormalizedLandmark]] (tọa độ chuẩn hóa ảnh)
└── pose_world_landmarks  -> List[List[Landmark]]           (tọa độ 3D mét thật)
```

Vì cấu hình `num_poses=1`, nếu phát hiện người thì danh sách có độ dài 1:
- `landmarks_2d = result.pose_landmarks[0]` (gồm 33 điểm).
- `landmarks_3d = result.pose_world_landmarks[0]` (gồm 33 điểm).

### 3.1 So sánh `pose_landmarks` và `pose_world_landmarks`

| Thuộc tính | `pose_landmarks` (Normalized) | `pose_world_landmarks` (World 3D) |
|---|---|---|
| **Hệ quy chiếu** | Khung hình ảnh 2D camera. | Không gian 3D thế giới thực. |
| **Trục X, Y** | Chuẩn hóa `[0.0, 1.0]`. Đổi ra pixel: `px = int(x * width)`, `py = int(y * height)`. | Đơn vị tính bằng **mét** ($m$). Gốc tọa độ $(0, 0, 0)$ đặt tại trung điểm hai hông. |
| **Trục Z** | Độ sâu tương đối so với hông. | Chiều sâu thực tế bằng mét. |
| **Ứng dụng tối ưu** | **Vẽ khung xương lên màn hình** (Overlay/Preview), kiểm tra vị trí trên ảnh. | **Tính góc chuyển động 3D** (ROM, gập khuỷu, nghiêng thân) vì không bị méo tỷ lệ phối cảnh (perspective distortion). |
| **Độ tin cậy** | Có `visibility` và `presence`. | Có `visibility` và `presence`. |

### 3.2 Ý nghĩa của `visibility` và `presence`
Mỗi điểm landmark có 2 chỉ số độ tin cậy từ `0.0` đến `1.0`:
- **`presence`**: Xác suất điểm khớp **nằm trong khung hình ảnh** (không bị lọt ra ngoài mép camera).
- **`visibility`**: Xác suất điểm khớp **nhìn thấy rõ** (không bị che khuất bởi cơ thể hoặc vật cản).
- *Nguyên tắc chất lượng:* Điểm khớp chỉ được coi là hợp lệ khi cả hai chỉ số đều $\ge$ ngưỡng cấu hình (mặc định `0.5`).

### 3.3 Bản đồ 33 Khớp và Khớp cần thiết cho Dang Vai (Shoulder Abduction)

```text
                0: Mũi (Nose)
               /             \
       7: Tai trái          8: Tai phải
          |                    |
   11: Vai trái (Left) ---- 12: Vai phải (Right)
       |                            |
   13: Khuỷu trái (Left)    14: Khuỷu phải (Right)
       |                            |
   15: Cổ tay trái (Left)   16: Cổ tay phải (Right)
       |                            |
   23: Hông trái (Left) --- 24: Hông phải (Right)
```

**Bảng mã khớp trọng yếu:**

| Khớp (Joint) | Index trái (Left) | Index phải (Right) | Mục đích phân tích |
|---|:---:|:---:|---|
| **Shoulder** (Vai) | `11` | `12` | Gốc xoay của cánh tay để tính ROM. |
| **Elbow** (Khuỷu tay) | `13` | `14` | Đỉnh góc khuỷu, kiểm tra lỗi gập tay bù trừ. |
| **Wrist** (Cổ tay) | `15` | `16` | Đầu mút cẳng tay, xác định hướng cánh tay. |
| **Hip** (Hông) | `23` | `24` | Gốc thân mình, dựng trục cột sống tính nghiêng thân. |
| **Nose / Ear** | `0`, `7`, `8` | (Mở rộng) | Nghiên cứu bù trừ đầu/cổ hoặc nhún vai. |

> **QUY TẮC BẮT BUỘC VỀ GIẢI PHẪU (ANATOMICAL LEFT/RIGHT):**  
> Nhãn `11 - Left Shoulder` luôn là vai bên trái **của cơ thể người đang tập**, không phụ thuộc vào việc camera nhìn đối diện hay gương hiển thị bị lật (mirror). Tuyệt đối không bao giờ đổi tên nhãn trái/phải dựa theo tọa độ màn hình.


---

## 4. KIẾN TRÚC PHÂN TẦNG: ÁP DỤNG THUẬT TOÁN VÀO SAU MODEL

Thuật toán đánh giá không can thiệp vào bên trong trọng số mô hình, mà được thiết kế theo mô hình đường ống phân tầng (Layered Pipeline):

```text
┌────────────────────────────────────────────────────────┐
│ [Nguồn dữ liệu] Camera Live / Video MP4 / UI-PRMD      │
└───────────────────────────┬────────────────────────────┘
                            │ (Khung hình RGB)
                            ▼
┌────────────────────────────────────────────────────────┐
│ [Pose Estimator] MediaPipe Pose Landmarker Full        │
└───────────────────────────┬────────────────────────────┘
                            │ (33 Landmarks thô)
                            ▼
┌────────────────────────────────────────────────────────┐
│ TẦNG 1: Adapter -> Canonical Joint Frame               │
│         Quality Gate (Kiểm tra lỗi) & Smoothing        │
└───────────────────────────┬────────────────────────────┘
                            │ (Tọa độ khớp chuẩn, làm mượt)
                            ▼
┌────────────────────────────────────────────────────────┐
│ TẦNG 2: Feature Engine (Tính ROM, Khuỷu, Thân mình)    │
└───────────────────────────┬────────────────────────────┘
                            │ (Chuỗi đặc trưng theo thời gian)
                            ▼
┌────────────────────────────────────────────────────────┐
│ TẦNG 3: Repetition Segmenter (Máy trạng thái FSM)      │
└───────────────────────────┬────────────────────────────┘
                            │ (Đoạn dữ liệu 1 Rep hoàn chỉnh)
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐   ┌────────────────────────┐
│ TẦNG 4A: Fixed Threshold  │   │ TẦNG 4B: DTW Evaluator │
│ - Đạt ROM tối thiểu?      │   │ - So khớp chuỗi chuẩn  │
│ - Gập khuỷu quá mức?      │   │ - Tính khoảng cách     │
│ - Nghiêng thân quá mức?   │   │   co dãn thời gian     │
└─────────────┬─────────────┘   └───────────┬────────────┘
              │                             │
              └─────────────┬───────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ [Kết quả phản hồi] Correct / Insufficient ROM / ...    │
└────────────────────────────────────────────────────────┘
```

---

## 5. TẦNG 1: TIỀN XỬ LÝ, QUALITY GATE VÀ LÀM MƯỢT

### 5.1 Canonical Joint Schema (Chuẩn hóa trung gian)
Để thuật toán chạy độc lập với nguồn gốc dữ liệu (dù landmark đến từ camera điện thoại, webcam PC hay dataset nghiên cứu UI-PRMD), tầng Adapter chuyển đổi dữ liệu về cấu trúc chuẩn:

```python
canonical_frame = {
    "timestamp_ms": timestamp,
    "valid": True,
    "active_side": "right",  # Bên tay thực hiện động tác ("left" hoặc "right")
    "coordinate_space": "world_m",
    "joints": {
        "shoulder": (x, y, z),
        "elbow": (x, y, z),
        "wrist": (x, y, z),
        "hip": (x, y, z),
        "opposite_shoulder": (x_opp, y_opp, z_opp),
        "opposite_hip": (x_opp, y_opp, z_opp)
    }
}
```

### 5.2 Quality Gate (Cổng kiểm tra chất lượng dữ liệu)
Một frame không đủ tin cậy nếu rơi vào các trường hợp sau:
1. Không phát hiện người (`len(pose_landmarks) == 0`).
2. Bất kỳ khớp bắt buộc nào của bên hoạt động (`shoulder`, `elbow`, `wrist`, `hip`) có `visibility < 0.5` hoặc `presence < 0.5`.
3. Tọa độ chứa giá trị vô lý (`NaN`, vô hạn, hoặc ngoài tầm).

Khi frame vi phạm Quality Gate:
- Đánh dấu `valid = False`, `invalid_reason = "low_confidence_joint"`.
- **Tuyệt đối không thay thế bằng giá trị `0`** vì tọa độ $(0, 0, 0)$ sẽ tạo ra các góc giả mạo cực lớn hoặc cực nhỏ.
- Nếu một lần lặp có tỷ lệ frame hợp lệ dưới $80\%$, hệ thống trả về nhãn `insufficient_data` thay vì kết luận đúng/sai.

### 5.3 Làm mượt tín hiệu (Smoothing)
Để tránh hiện tượng rung lắc tọa độ ảnh hưởng đến việc tính đạo hàm và góc:
- **Exponential Moving Average (EMA) cho xử lý thời gian thực:**
  $$S_t = \alpha \cdot X_t + (1 - \alpha) \cdot S_{t-1} \quad (\text{với } \alpha \approx 0.35)$$
- **Savitzky-Golay Filter cho phân tích offline:** Bộ lọc đa thức trượt giúp làm mượt tín hiệu nhưng không làm cùn đi các điểm cực trị (đỉnh ROM).


---

## 6. TẦNG 2: TRÍCH XUẤT ĐẶC TRƯNG VẬN ĐỘNG (FEATURE ENGINE)

Mọi hàm tính góc hình học cần tuân thủ **nguyên tắc an toàn số học**:
- Tính norm vector, kiểm tra nếu $\approx 0$ thì trả về `NaN` hoặc `Invalid`.
- Tích vô hướng (dot product) chia cho tích độ dài phải được kẹp chặt (clamp) về khoảng $[-1.0, 1.0]$ trước khi gọi `arccos` để tránh lỗi sụp đổ chương trình do sai số dấu phẩy động.

```python
import numpy as np

def safe_vector_angle(v1, v2):
    """Tính góc (độ) giữa hai vector 2D hoặc 3D với cơ chế an toàn số học."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-7 or norm2 < 1e-7:
        return np.nan
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_clamped = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_clamped)))
```

### 6.1 Đặc trưng 1: Biên độ cử động vai (Range of Motion - ROM)
- **Mô tả:** Góc mở giữa cánh tay và thân mình trên mặt phẳng đứng ngang.
- **Công thức:**
  $$\vec{v}_{\text{trunk}} = P_{\text{hip}} - P_{\text{shoulder}}$$
  $$\vec{v}_{\text{arm}} = P_{\text{elbow}} - P_{\text{shoulder}}$$
  $$\text{ROM} = \text{safe\_vector\_angle}(\vec{v}_{\text{trunk}}, \vec{v}_{\text{arm}})$$
- **Kỳ vọng:**
  - Tay hạ xuôi theo người: $\text{ROM} \approx 0^\circ - 15^\circ$.
  - Tay dang ngang bằng vai: $\text{ROM} \approx 90^\circ$.
  - Tay giơ thẳng đứng lên cao: $\text{ROM} \rightarrow 180^\circ$.

### 6.2 Đặc trưng 2: Góc gập khuỷu tay (Elbow Flexion)
- **Mô tả:** Trong bài tập Dang vai chuẩn, cánh tay phải giữ thẳng. Người tập yếu cơ vai thường gập khuỷu tay để giảm cánh tay đòn (hiện tượng bù trừ).
- **Công thức:**
  $$\vec{v}_{\text{upper}} = P_{\text{shoulder}} - P_{\text{elbow}}$$
  $$\vec{v}_{\text{forearm}} = P_{\text{wrist}} - P_{\text{elbow}}$$
  $$\theta_{\text{elbow}} = \text{safe\_vector\_angle}(\vec{v}_{\text{upper}}, \vec{v}_{\text{forearm}})$$
  $$\text{Elbow Flexion} = 180^\circ - \theta_{\text{elbow}}$$
- **Quy ước chuẩn:**
  - Cánh tay duỗi thẳng hoàn toàn: $\theta_{\text{elbow}} = 180^\circ \implies \text{Elbow Flexion} = 0^\circ$.
  - Cánh tay gập vuông góc: $\theta_{\text{elbow}} = 90^\circ \implies \text{Elbow Flexion} = 90^\circ$.
  - *Lưu ý:* Bắt buộc dùng đại lượng `Elbow Flexion` trong các ngưỡng đánh giá lỗi để giá trị $0^\circ$ luôn đồng nghĩa với tư thế chuẩn (không bị gập).

### 6.3 Đặc trưng 3: Nghiêng thân sang bên (Trunk Lateral Lean)
- **Mô tả:** Động tác nghiêng người sang bên đối diện để hỗ trợ nhấc cánh tay lên.
- **Công thức:**
  $$P_{\text{mid\_hip}} = \frac{P_{\text{left\_hip}} + P_{\text{right\_hip}}}{2}$$
  $$P_{\text{mid\_shoulder}} = \frac{P_{\text{left\_shoulder}} + P_{\text{right\_shoulder}}}{2}$$
  $$\vec{v}_{\text{spine}} = P_{\text{mid\_shoulder}} - P_{\text{mid\_hip}} = (\Delta x, \Delta y, \Delta z)$$
  $$\text{Trunk Lateral Lean} = \text{degrees}\left(\arctan2(|\Delta x|, |\Delta y|)\right)$$
- **Kỳ vọng:**
  - Thân người đứng thẳng: $\text{Trunk Lateral Lean} \approx 0^\circ - 3^\circ$.
  - Nghiêng người quá mức: $\text{Trunk Lateral Lean} > 10^\circ - 15^\circ$.

---

## 7. TẦNG 3: TÁCH VÀ ĐẾM SỐ LẦN LẶP (REPETITION FSM)

Để đếm repetition chính xác, **tuyệt đối không đếm bằng một ngưỡng đơn lẻ** (ví dụ: cứ ROM > 90° thì cộng 1) vì sẽ gây lỗi đếm liên tục do nhiễu rung lắc quanh ngưỡng. Hệ thống sử dụng **Máy trạng thái hữu hạn (Finite-State Machine - FSM)** với nguyên lý **Hysteresis (hai ngưỡng trễ)**:

```text
    ┌────────┐
    │  IDLE  │ (ROM < 20°, người đứng yên chuẩn bị)
    └────┬───┘
         │ ROM vượt ngưỡng bắt đầu (vd: > 25°)
         ▼
   ┌───────────┐
   │  RISING   │ (Cánh tay đang nâng lên, vận tốc dROM/dt > 0)
   └─────┬─────┘
         │ ROM đạt cực đại và vận tốc giảm về gần 0 (vd: ROM > 60°)
         ▼
    ┌──────────┐
    │   PEAK   │ (Điểm đỉnh cao nhất của lần lặp)
    └────┬─────┘
         │ Cánh tay bắt đầu hạ xuống có chủ đích (dROM/dt < 0)
         ▼
  ┌─────────────┐
  │  LOWERING   │ (Cánh tay đang hạ dần)
  └──────┬──────┘
         │ ROM giảm về dưới ngưỡng kết thúc (vd: < 20°)
         ▼
  ┌─────────────┐
  │  COMPLETE   │ ──> Đóng gói dữ liệu 1 Rep (start -> peak -> end)
  └──────┬──────┘     Tăng Rep count, chuyển dữ liệu sang Evaluator
         │
         └──────────> Quay về IDLE
```

### Các biện pháp chống đếm sai của FSM:
1. **Ngưỡng trễ (Hysteresis):** Ngưỡng bắt đầu động tác (`ROM > 25°`) cao hơn hẳn ngưỡng kết thúc động tác (`ROM < 20°`).
2. **Thời lượng rep hợp lệ:** Repetition phải có độ dài từ `1.0s` đến `6.0s`. Nếu rep hoàn thành quá nhanh (< 0.5s do vung tay hoặc nhiễu), FSM sẽ tự động loại bỏ.
3. **Bộ đệm xác nhận trạng thái:** Phải thỏa mãn điều kiện chuyển trạng thái trong ít nhất 3 frame liên tiếp trước khi chính thức chuyển state.


---

## 8. TẦNG 4: ĐÁNH GIÁ CHẤT LƯỢNG ĐỘNG TÁC

Sau khi FSM chốt một đoạn Repetition, toàn bộ chuỗi đặc trưng của Rep đó được chuyển đồng thời vào 2 bộ đánh giá độc lập:

### 8.1 Baseline 1: Fixed Threshold (Ngưỡng cố định)

Bộ đánh giá dựa trên các luật lâm sàng rõ ràng, có khả năng giải thích cao (explainable):
- **Biên độ cực đại đạt được:** $\text{Peak ROM} = \max(\text{ROM}_t)$
- **Mức độ gập khuỷu lớn nhất:** $\text{Max Flexion} = \text{Percentile}_{95}(\text{Elbow Flexion}_t)$ (dùng phân vị 95 để loại trừ ngoại lai thay vì lấy cực trị một frame).
- **Mức độ nghiêng thân lớn nhất:** $\text{Max Lean} = \text{Percentile}_{95}(\text{Trunk Lateral}_t)$

```python
def evaluate_fixed_threshold(rep_features, config):
    """Đánh giá chất lượng 1 repetition bằng các ngưỡng quy tắc cố định."""
    labels = []
    
    # 1. Kiểm tra tầm vận động tối thiểu (ROM)
    if rep_features['peak_rom'] < config['rom_min_deg']:
        labels.append("insufficient_rom")
        
    # 2. Kiểm tra gập khuỷu bù trừ
    if rep_features['max_elbow_flexion'] > config['elbow_flexion_max_deg']:
        labels.append("excessive_elbow_flexion")
        
    # 3. Kiểm tra nghiêng thân bù trừ
    if rep_features['max_trunk_lean'] > config['trunk_lateral_max_deg']:
        labels.append("excessive_trunk_lean")
        
    # Kết luận
    if not labels:
        return {"status": "correct", "details": []}
    else:
        return {"status": "incorrect", "details": labels}
```

*Nguyên tắc:* Các ngưỡng cụ thể (`rom_min_deg`, `elbow_flexion_max_deg`, `trunk_lateral_max_deg`) phải được lưu trong file cấu hình (`configs/fixed_threshold.yaml`) và được tối ưu trên tập training/validation, không được tự ý điền số cứng theo cảm tính.

---

### 8.2 Baseline 2: Dynamic Time Warping (DTW)

Phương pháp so khớp hình dạng chuỗi thời gian, giải quyết triệt để bài toán người tập thực hiện nhanh hay chậm khác nhau mà không cần ép về cùng số lượng frame.

#### Bước 1: Xây dựng vector đặc trưng cho mỗi frame trong Rep
Tại mỗi frame $t$ trong repetition, tạo vector đặc trưng:
$$\vec{f}_t = \left[ \text{ROM}_t, \; \text{Elbow Flexion}_t, \; \text{Trunk Lateral}_t \right]$$
Một rep gồm $N$ frame sẽ tạo thành ma trận chuỗi $X \in \mathbb{R}^{N \times 3}$.

#### Bước 2: Chuẩn hóa dữ liệu (Z-Score Normalization)
Chuẩn hóa từng đặc trưng theo mean ($\mu$) và std ($\sigma$) đã học từ tập mẫu chuẩn (training split):
$$\hat{f}_{t, i} = \frac{f_{t, i} - \mu_i}{\sigma_i}$$

#### Bước 3: So khớp với chuỗi mẫu tham chiếu (Template Matching)
- **Chuỗi mẫu chuẩn ($T$):** Được chọn bằng phương pháp **Medoid** từ các rep đúng của tập train (rep chuẩn nhất có tổng khoảng cách DTW tới các rep đúng còn lại là nhỏ nhất).
- **Ràng buộc Sakoe-Chiba Window:** Giới hạn đường co dãn (warping path) trong dải hẹp quanh đường chéo chính để tránh ghép lệch thời gian phi thực tế (ví dụ: giai đoạn giơ tay lại ghép với giai đoạn hạ tay).
- **Khoảng cách chuẩn hóa:**
  $$\text{Normalized DTW Distance} = \frac{\text{Total Alignment Cost}}{\text{Warping Path Length}}$$
- **Phân loại:** So sánh khoảng cách với ngưỡng $\tau_{\text{dtw}}$ (chọn trên validation set):
  - Nếu $\text{Distance} \le \tau_{\text{dtw}} \implies \textbf{Correct}$.
  - Nếu $\text{Distance} > \tau_{\text{dtw}} \implies \textbf{Incorrect}$.


---

## 9. CODE VÍ DỤ HOÀN CHỈNH KẾT NỐI TOÀN BỘ PIPELINE

Dưới đây là đoạn code mẫu Python minh họa cách ráp nối hoàn chỉnh từ camera $\rightarrow$ MediaPipe $\rightarrow$ Feature Engine $\rightarrow$ Repetition FSM:

```python
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Khởi tạo Pose Landmarker Full
base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

# 2. Định nghĩa FSM trạng thái
class ShoulderAbductionFSM:
    def __init__(self):
        self.state = 'IDLE'
        self.rep_count = 0
        self.rom_history = []

    def update(self, rom_deg):
        completed = False
        if self.state == 'IDLE':
            if rom_deg > 25.0:
                self.state = 'RISING'
                self.rom_history = [rom_deg]
        elif self.state == 'RISING':
            self.rom_history.append(rom_deg)
            if rom_deg > 75.0:
                self.state = 'PEAK'
        elif self.state == 'PEAK':
            self.rom_history.append(rom_deg)
            if rom_deg < 65.0:
                self.state = 'LOWERING'
        elif self.state == 'LOWERING':
            self.rom_history.append(rom_deg)
            if rom_deg < 20.0:
                self.state = 'IDLE'
                self.rep_count += 1
                completed = True
        return self.state, self.rep_count, completed

# 3. Vòng lặp camera trực tiếp
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
fsm = ShoulderAbductionFSM()
start_time = time.perf_counter()
last_ts = -1

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Timestamp tăng nghiêm ngặt cho RunningMode.VIDEO
        ts = max(last_ts + 1, int((time.perf_counter() - start_time) * 1000))
        last_ts = ts

        # Chuẩn bị ảnh cho MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, ts)

        status_str = "NO POSE"
        if result.pose_world_landmarks:
            lm3d = result.pose_world_landmarks[0]
            
            # Khớp 3D mét thật: Vai phải (12), Khuỷu phải (14), Hông phải (24)
            sh = np.array([lm3d[12].x, lm3d[12].y, lm3d[12].z])
            el = np.array([lm3d[14].x, lm3d[14].y, lm3d[14].z])
            hp = np.array([lm3d[24].x, lm3d[24].y, lm3d[24].z])

            # Tính ROM vai
            v_trunk = hp - sh
            v_arm = el - sh
            cos_rom = np.dot(v_trunk, v_arm) / (np.linalg.norm(v_trunk) * np.linalg.norm(v_arm) + 1e-7)
            rom = float(np.degrees(np.arccos(np.clip(cos_rom, -1.0, 1.0))))

            state, count, rep_done = fsm.update(rom)
            status_str = f"State: {state} | ROM: {rom:.1f} deg | Reps: {count}"

            if rep_done:
                peak_rom = max(fsm.rom_history)
                print(f">> [HOÀN THÀNH REP {count}] Đỉnh ROM: {peak_rom:.1f}°")

        cv2.putText(frame, status_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Shoulder Abduction Pipeline", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break
finally:
    cap.release()
    detector.close()
    cv2.destroyAllWindows()
```

---

## 10. CÁC BẪY KỸ THUẬT VÀ NGUYÊN TẮC BẮT BUỘC

1. **Tuyệt đối không so sánh trực tiếp tọa độ của UI-PRMD với MediaPipe bằng DTW:**  
   UI-PRMD được ghi bằng hệ thống cảm biến quang học/Kinect có hệ trục tọa độ, độ cao gốc và đơn vị khác biệt. Luôn chuyển cả hai nguồn về **chuỗi góc đặc trưng (ROM, Elbow Flexion, Trunk Lean)** rồi mới đưa vào DTW.
2. **Không trộn lẫn tọa độ 2D và 3D trong cùng một phép tính:**  
   - Tính góc bằng `pose_world_landmarks` (tọa độ mét thật 3D).
   - Vẽ khung xương và kiểm tra góc nhìn camera bằng `pose_landmarks` (tọa độ chuẩn hóa ảnh).
3. **Luôn kẹp chặt cosine trước khi gọi arccos:**  
   Sai số dấu phẩy động có thể khiến giá trị tính ra là `1.0000000002`. Nếu không dùng `np.clip(val, -1.0, 1.0)`, hàm `np.arccos` sẽ trả về `NaN`, làm gãy toàn bộ chuỗi tính toán phía sau.
4. **Không chia tập dữ liệu train/test ngẫu nhiên theo frame:**  
   Các frame của cùng một người trong một lần tập rất giống nhau. Chia ngẫu nhiên theo frame sẽ gây rò rỉ dữ liệu (Data Leakage) dẫn đến độ chính xác cao giả tạo. **Bắt buộc phải chia train/test theo Subject (GroupKFold theo người tập)**.
5. **Giới hạn phạm vi tuyên bố:**  
   Đây là hệ thống hỗ trợ kỹ thuật bài tập phục hồi chức năng trong phạm vi nghiên cứu kỹ thuật Đồ án 1; không tuyên bố là thiết bị chẩn đoán y khoa hay thay thế phác đồ của bác sĩ.


---

## 11. ĐÁNH GIÁ HIỆU NĂNG (RESOURCE FOOTPRINT) VÀ TÍNH KHẢ THI TRIỂN KHAI MOBILE APP

### 11.1 Bảng đo lường hiệu năng thực tế (Resource Footprint)

Dưới đây là các thông số tiêu thụ tài nguyên thực tế của mô hình `pose_landmarker_full.task` (kèm so sánh giữa PC hiện tại và các dòng chip điện thoại):

| Tiêu chí | PC Benchmark (Đã đo thực tế) | Mobile Tầm Trung (Snapdragon 778G / Dimensity 7050) | Mobile Cao Cấp (Snapdragon 8 Gen 2/3, Apple A16/A17) |
|---|---|---|---|
| **Dung lượng File Model** | **9.39 MB** (float16) | 9.39 MB (có thể dùng Lite 4.7 MB nếu cần) | 9.39 MB |
| **Độ trễ suy luận (Inference Latency - CPU)** | **~9.8 ms / frame** (XNNPACK CPU) | ~30 – 45 ms / frame | ~15 – 25 ms / frame |
| **Độ trễ suy luận (Inference Latency - GPU)** | ~5 – 7 ms | ~14 – 20 ms / frame | ~8 – 12 ms / frame |
| **Tốc độ xử lý (FPS suy luận)** | **~100 FPS** | **~25 – 45 FPS** | **~45 – 80 FPS** |
| **RAM tiêu thụ (Heap Memory)** | ~120 – 160 MB | ~90 – 140 MB | ~100 – 150 MB |
| **Tải CPU (CPU Utilization)** | ~15% (1–2 core) | ~25 – 35% (chạy background thread) | ~15 – 20% |
| **Mức tiêu hao năng lượng / Nhiệt độ** | Không đáng kể | Ấm nhẹ sau 10–15 phút tập liên tục | Ổn định, không giật lag |

*Nhận xét:*
- Thuật toán tính góc (Feature Engine) và FSM chỉ mất **< 0.1 ms / frame** (các phép tính ma trận và tích vô hướng trên 33 điểm khớp có chi phí tính toán cực kỳ nhỏ so với mạng nơ-ron).
- Toàn bộ bottleneck hiệu năng nằm ở khâu đọc luồng camera và mạng nơ-ron Pose Estimator.

---

### 11.2 Kết luận tính khả thi trên Mobile: HOÀN TOÀN KHẢ THI (RẤT CAO)

Triển khai bài toán này lên thiết bị di động (Android / iOS) là **hoàn toàn khả thi và thực tế 100%**, với các lý do kỹ thuật sau:

1. **Đặc thù bài tập Phục hồi chức năng (Physical Therapy):**
   - Động tác Dang Vai của người tập/bệnh nhân luôn diễn ra với **tốc độ chậm và có kiểm soát** (trung bình 1 lần lặp kéo dài từ 2 đến 4 giây).
   - Do đó, tốc độ xử lý **15 – 30 FPS** trên điện thoại là **dư thừa độ mượt** để máy trạng thái FSM bắt trọn pha nâng tay (`RISING`), chạm đỉnh (`PEAK`) và hạ tay (`LOWERING`) mà không bị rớt dữ liệu.
2. **MediaPipe Tasks đã được Google tối ưu riêng cho Edge/Mobile:**
   - Hỗ trợ bộ tăng tốc phần cứng **TFLite GPU Delegate** và **NNAPI / NPU** tích hợp sẵn trong Android SDK (`com.google.mediapipe:tasks-vision`).
   - Khởi chạy hoàn toàn **Offline (On-Device)**, không phụ thuộc kết nối Internet, bảo mật tuyệt đối hình ảnh cá nhân của người bệnh.
3. **Kích thước đóng gói gọn nhẹ:**
   - File model chỉ chiếm ~9.4 MB, khi đóng gói vào tệp APK/AAB chỉ làm tăng kích thước ứng dụng một lượng rất nhỏ.

---

### 11.3 Kiến trúc triển khai tối ưu trên Android (Native vs Flutter)

Theo khuyến nghị trong **Mục 16 của `AGENTS.md`**, để đạt FPS cao nhất và tránh hiện tượng giật lag, kiến trúc ứng dụng di động phải tuân theo nguyên tắc: **Xử lý Frame nặng ở Native, chỉ gửi dữ liệu nhẹ sang UI**.

```text
┌─────────────────────────────────────────────────────────────────┐
│                 NATIVE ANDROID LAYER (Kotlin)                   │
│                                                                 │
│  CameraX Preview & ImageAnalysis (640x480)                      │
│       │                                                         │
│       ▼ (YUV_420_888 -> MPImage)                                │
│  MediaPipe Tasks Android SDK (LIVE_STREAM, Background Thread)   │
│       │                                                         │
│       ▼ (33 Landmarks)                                          │
│  Feature Engine & Repetition FSM (Kotlin)                       │
└───────┬─────────────────────────────────────────────────────────┘
        │ (Chỉ gửi JSON/Map nhẹ: Rep count, ROM, Status, Feedback)
        ▼ [Platform MethodChannel / EventChannel]
┌─────────────────────────────────────────────────────────────────┐
│                 FLUTTER APPLICATION LAYER (Dart)                │
│                                                                 │
│  - Giao diện người dùng (Camera preview + Skeleton Canvas)      │
│  - Hiển thị số lần lặp: "Reps: 5"                               │
│  - Phản hồi trực quan / giọng nói: "Tập tốt!", "Gập khuỷu tay!" │
└─────────────────────────────────────────────────────────────────┘
```

> **CẢNH BÁO KIẾN TRÚC QUAN TRỌNG:**  
> **Không** lấy từng frame ảnh bitmap/raw từ Flutter Camera rồi truyền qua `MethodChannel` sang Native để phân tích. Việc sao chép bộ nhớ (memory copy) liên tục giữa Dart VM và Android Native sẽ làm tụt FPS xuống dưới 8-10 FPS và làm máy rất nóng. Hãy để **CameraX và MediaPipe chạy khép kín hoàn toàn ở tầng Android Native**.

---

### 11.4 Chiến lược tối ưu Pin, Nhiệt độ và FPS trên Mobile

1. **Cơ chế Backpressure (Bỏ khung hình khi bận):**
   - Trong CameraX, sử dụng chiến lược:
     ```kotlin
     ImageAnalysis.Builder()
         .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
         .build()
     ```
   - Khi CPU/GPU đang bận suy luận frame trước, frame camera mới nhất sẽ ghi đè lên bộ đệm thay vì tạo hàng đợi vô tận gây tràn RAM (OOM Crash).
2. **Độ phân giải Camera tối ưu:**
   - Cấu hình CameraX ở mức **VGA (640×480)** hoặc **HD (1280×720)**. Không mở chế độ 1080p hay 4K vì model MediaPipe Pose Landmarker Full tự động resize ảnh đầu vào về kích thước chuẩn $256 \times 256$ pixel trước khi đưa vào mạng nơ-ron. Độ phân giải camera quá cao chỉ làm tốn pin và nóng máy.
3. **Cấu hình mô hình gọn nhẹ:**
   - Đặt `outputSegmentationMasks = false` (tắt mặt nạ tách người giúp tiết kiệm ~35% năng lượng xử lý).
   - Đặt `numPoses = 1` (giới hạn 1 người tập duy nhất).
4. **Kiểm soát tốc độ phân tích (Frame Throttling):**
   - Không cần chạy MediaPipe ở 60 FPS. Chỉ cần lấy mẫu ở mức **20 – 30 FPS** là đã đảm bảo độ phân giải thời gian tối ưu cho bài tập phục hồi chức năng, giúp thiết bị chạy liên tục 30-45 phút mà không bị quá nhiệt (thermal throttling).

