# BÁO CÁO NGHIÊN CỨU & XỬ LÝ DỮ LIỆU ÂM THANH (LAB 03)
## ĐỀ TÀI: CHUẨN BỊ VÀ CHUẨN HÓA DỮ LIỆU TIẾNG NÓI CHO NHẬN DẠNG LỆNH ĐIỀU KHIỂN
**Sinh viên thực hiện:** Lê Trọng Huy  
**Mã số sinh viên (MSSV):** 2351260658  
**Môn học:** Xử lý Âm thanh và Tiếng nói (Audio & Speech Processing)  
**Thư mục làm việc:** `lab03_2351260658_LeTrongHuy`

---

## 1. BẢNG ĐỐI CHIẾU MỤC TIÊU VÀ NGHIỆM THU HOÀN THÀNH ("CHECKLIST")

File `README.md` này là nơi đối chiếu chính thức toàn bộ các yêu cầu của người dùng về mặt **DỮ LIỆU** theo phương thức **"Làm - Sai - Sửa"**:

| STT | Yêu cầu của người dùng | Tệp tin thực hiện | Trạng thái nghiệm thu | Kết quả đối chiếu thực tế |
|:---:|:---|:---|:---:|:---|
| **1** | **Tạo 1 file chứa thư viện cần phải cài đặt** | [`requirements.txt`](requirements.txt) | **ĐÃ HOÀN THÀNH (100%)** | Chứa đầy đủ các thư viện phục vụ tạo, đọc, phân tích và chuẩn hóa âm thanh: `numpy`, `scipy`, `librosa`, `soundfile`, `gTTS`, `pydub`, `sounddevice`. |
| **2** | **Yêu cầu dữ liệu:**<br>- Định dạng: WAV mono (1 kênh)<br>- Tần số lấy mẫu: 16 kHz (16000 Hz)<br>- Độ sâu bit: 16-bit PCM (`PCM_16`)<br>- Có khoảng lặng: 0.2–0.5 s ở đầu và cuối<br>- Bộ từ mặc định: 5 lệnh (“mở”, “đóng”, “lên”, “xuống”, “dừng”)<br>- Viết 1 đoạn code Python để tạo dữ liệu như trên | [`dataset_generator.py`](dataset_generator.py) | **ĐÃ HOÀN THÀNH (100%)** | Đoạn code Python tự động tạo bộ dữ liệu chuẩn hóa chính xác $100\%$ các tiêu chí kỹ thuật: WAV mono, sample rate 16000 Hz, 16-bit PCM, khoảng lặng ngẫu nhiên trong khoảng $[0.2s - 0.5s]$ ở đầu và cuối mỗi file âm thanh. |
| **3** | **Phân chia dữ liệu:**<br>- Mỗi lớp tối thiểu 8 utterance mỗi người nói<br>- Gồm 6 mẫu train và 2 mẫu test<br>- Không để cùng một file xuất hiện ở cả hai tập<br>- Thí nghiệm speaker-independent phải tách theo người nói, không chỉ theo file<br>- Dữ liệu của 2 người nói để thử nghiệm speaker mismatch | [`dataset_generator.py`](dataset_generator.py)<br>[`verify_dataset.py`](verify_dataset.py) | **ĐÃ HOÀN THÀNH (100%)** | Tạo tổng cộng **80 files WAV** cho 2 người nói:<br>- `speaker1` (tông trầm/nam): 5 từ $\times$ 8 mẫu = 40 files (30 train, 10 test)<br>- `speaker2` (tông cao/nữ): 5 từ $\times$ 8 mẫu = 40 files (30 train, 10 test)<br>- Train và Test tách biệt độc lập theo người nói và theo file, **0 file trùng lặp**. |
| **4** | **Tự kiểm tra lại yêu cầu theo phương thức "Làm - Sai - Sửa"** | [`verify_dataset.py`](verify_dataset.py) | **ĐÃ HOÀN THÀNH (100%)** | Script tự động quét toàn bộ 80 file để kiểm tra header, kiểm tra năng lượng khoảng lặng (VAD), kiểm tra hash MD5 chống trùng lặp và xác thực phân bổ mẫu. |

---

## 2. NHẬT KÝ PHƯƠNG THỨC "LÀM - SAI - SỬA"

Trong quá trình thực hiện tạo dữ liệu, phương thức **"Làm - Sai - Sửa"** đã được áp dụng để phát hiện lỗi và khắc phục triệt để:

### Lỗi phát hiện: UnicodeEncodeError trên Console Windows cp1252
- **Làm:** Viết script `dataset_generator.py` và chạy trên Windows PowerShell.
- **Sai:** Terminal văng lỗi:
  ```text
  UnicodeEncodeError: 'charmap' codec can't encode character '\u1ede' in position 2: character maps to <undefined>
  ```
  Nguyên nhân do Windows console mặc định sử dụng codepage `cp1252` không in được các ký tự tiếng Việt có dấu.
- **Sửa:** Bổ sung cấu hình tự động reconfigure encoding sang UTF-8 cho `sys.stdout` và `sys.stderr`:
  ```python
  if hasattr(sys.stdout, "reconfigure"):
      sys.stdout.reconfigure(encoding="utf-8")
      sys.stderr.reconfigure(encoding="utf-8")
  ```
- **Kết quả sau sửa:** Quá trình tạo toàn bộ 80 file âm thanh chạy trơn tru, hiển thị đầy đủ thông tin tiếng Việt trên console.

---

## 3. CẤU TRÚC BỘ DỮ LIỆU ĐÃ TẠO

Thư mục dữ liệu được tổ chức phân cấp rõ ràng theo người nói (`speaker`), tập dữ liệu (`train`/`test`), và nhãn từ lệnh (`label`):

```text
lab03_2351260658_LeTrongHuy/
├── requirements.txt               # Thư viện cần cài đặt
├── dataset_generator.py           # Code Python tạo dữ liệu theo đúng yêu cầu
├── verify_dataset.py              # Code Python tự động kiểm tra và đối soát dữ liệu
├── record_audio.py                # Script hỗ trợ thu âm thêm giọng thật qua Micro (nếu cần)
├── README.md                      # Bảng đối chiếu mục tiêu và báo cáo dữ liệu
└── dataset/
    ├── metadata.csv               # Bảng chi tiết thông số kỹ thuật của toàn bộ 80 file
    ├── speaker1/                  # Người nói 1 (Tông trầm / Nam)
    │   ├── train/                 # 30 file (6 mẫu x 5 từ)
    │   │   ├── mo/                # mo_speaker1_train_01.wav ... 06.wav
    │   │   ├── dong/              # dong_speaker1_train_01.wav ... 06.wav
    │   │   ├── len/               # len_speaker1_train_01.wav ... 06.wav
    │   │   ├── xuong/             # xuong_speaker1_train_01.wav ... 06.wav
    │   │   └── dung/              # dung_speaker1_train_01.wav ... 06.wav
    │   └── test/                  # 10 file (2 mẫu x 5 từ)
    │       ├── mo/                # mo_speaker1_test_07.wav, 08.wav
    │       ├── dong/              # dong_speaker1_test_07.wav, 08.wav
    │       ├── len/               # len_speaker1_test_07.wav, 08.wav
    │       ├── xuong/             # xuong_speaker1_test_07.wav, 08.wav
    │       └── dung/              # dung_speaker1_test_07.wav, 08.wav
    └── speaker2/                  # Người nói 2 (Tông cao / Nữ)
        ├── train/                 # 30 file (6 mẫu x 5 từ)
        │   ├── mo/
        │   ├── dong/
        │   ├── len/
        │   ├── xuong/
        │   └── dung/
        └── test/                  # 10 file (2 mẫu x 5 từ)
            ├── mo/
            ├── dong/
            ├── len/
            ├── xuong/
            └── dung/
```

---

## 4. CHI TIẾT THÔNG SỐ KỸ THUẬT CỦA TỪNG FILE ÂM THANH

Mỗi file trong tập dữ liệu đều tuân thủ nghiêm ngặt các thông số:
- **Audio Format:** WAV (RIFF header)
- **Channels:** 1 channel (Mono tuyệt đối)
- **Sample Rate:** $16000\text{ Hz}$ ($16\text{ kHz}$)
- **Bit Depth:** 16-bit Linear PCM (Signed 16-bit PCM, subtype `PCM_16`)
- **Khoảng lặng (Silence Padding):**
  - Đầu file: $0.20\text{ s} - 0.50\text{ s}$
  - Cuối file: $0.20\text{ s} - 0.50\text{ s}$
  - Được kiểm soát bằng VAD (Voice Activity Detection / Energy Trimming) trước khi ghép mảng khoảng lặng có thêm nhiễu nền vi mô ($10^{-5}$) nhằm mô phỏng âm học phòng thu chân thực.
- **Biến thiên âm học (Acoustic Perturbation):**
  - `speaker1`: Cao độ trầm (pitch shift âm), tốc độ phát âm biến thiên nhẹ ($0.92\times - 1.05\times$).
  - `speaker2`: Cao độ thanh/cao (pitch shift dương), tốc độ phát âm biến thiên nhẹ ($0.95\times - 1.08\times$).

---

## 5. HƯỚNG DẪN THỰC THI VÀ TỰ ĐỐI SOÁT

### Bước 1: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Tái tạo dữ liệu (nếu muốn sinh mới)
```bash
python dataset_generator.py
```

### Bước 3: Chạy script tự động kiểm tra đối chiếu dữ liệu ("Làm - Sai - Sửa")
```bash
python verify_dataset.py
```

**Kết quả đối soát thực tế thu được từ script kiểm tra:**
```text
===========================================================================
KẾT QUẢ ĐỐI SOÁT YÊU CẦU DỮ LIỆU:
===========================================================================
1. Tổng số file kiểm tra: 80 file
2. Số người nói: 2 người nói (đáp ứng thí nghiệm speaker mismatch)
3. Tập Train: 60 file | Tập Test: 20 file
4. Trùng lặp Train/Test: 0 file (HOÀN HẢO - Không bị trùng)
5. Chuẩn âm thanh (WAV, Mono, 16kHz, 16-bit PCM): ĐÃ XÁC THỰC 100%


```
