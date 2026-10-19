"""
dataset_generator.py
Tự động tạo bộ dữ liệu tiếng nói chuẩn cho bài toán Nhận dạng tiếng nói nâng cao với HMM.
Sinh viên: Lê Trọng Huy - MSSV: 2351260658

Yêu cầu kỹ thuật đạt được:
- Định dạng âm thanh: WAV mono, 16 kHz, 16-bit PCM.
- Khoảng lặng (silence padding): 0.2 - 0.5 giây ở đầu và cuối mỗi file.
- Bộ từ vựng (5 lệnh): "mở", "đóng", "lên", "xuống", "dừng".
- Số lượng người nói: 2 người (Speaker 1 - tông trầm/nam, Speaker 2 - tông cao/nữ).
- Phân chia: Mỗi người nói có 8 mẫu / từ (6 train, 2 test).
- Tổng cộng: 2 speakers * 5 từ * 8 mẫu = 80 files WAV riêng biệt, không trùng lặp.
"""

import os
import sys
import io
import csv
import random
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS
import warnings

# Đảm bảo UTF-8 cho console Windows tránh UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

warnings.filterwarnings("ignore")

# Cấu hình tham số chuẩn
TARGET_SR = 16000           # 16 kHz
TARGET_CHANNELS = 1         # Mono
BIT_DEPTH = "PCM_16"        # 16-bit PCM
MIN_SILENCE_SEC = 0.20      # Khoảng lặng tối thiểu đầu/cuối: 0.2s
MAX_SILENCE_SEC = 0.50      # Khoảng lặng tối đa đầu/cuối: 0.5s

# 5 lệnh tiếng Việt bắt buộc
VOCABULARY = [
    {"label": "mo",    "text": "mở"},
    {"label": "dong",  "text": "đóng"},
    {"label": "len",   "text": "lên"},
    {"label": "xuong", "text": "xuống"},
    {"label": "dung",  "text": "dừng"}
]

# 2 Người nói với đặc trưng âm vực khác nhau
SPEAKERS = {
    "speaker1": {
        "desc": "Người nói 1 (Tông trầm / Nam)",
        "pitch_shift_range": (-3.0, -1.0),
        "speed_range": (0.92, 1.05),
        "train_samples": 6,
        "test_samples": 2
    },
    "speaker2": {
        "desc": "Người nói 2 (Tông cao / Nữ)",
        "pitch_shift_range": (2.0, 4.0),
        "speed_range": (0.95, 1.08),
        "train_samples": 6,
        "test_samples": 2
    }
}

def generate_base_audio_gtts(text: str, sr: int = TARGET_SR) -> np.ndarray:
    """Tạo âm thanh cơ bản từ gTTS tiếng Việt và chuyển sang mảng numpy float32 tại sample rate yêu cầu."""
    tts = gTTS(text=text, lang="vi", slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    
    # Load qua librosa resample chính xác về sr=16000, mono=True
    y, _ = librosa.load(fp, sr=sr, mono=True)
    return y

def trim_silence(y: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    """Loại bỏ khoảng lặng ban đầu của tín hiệu TTS để kiểm soát chính xác khoảng lặng mới."""
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(trimmed) < 100:
        return y
    return trimmed

def add_acoustic_variations(y: np.ndarray, sr: int, pitch_shift: float, speed_factor: float) -> np.ndarray:
    """
    Tạo biến thể phát âm tự nhiên:
    - Pitch shift: thay đổi cao độ giọng nói
    - Time stretch: thay đổi tốc độ phát âm
    - Thêm nhiễu nền vi mô (Gaussian noise ~ 45dB SNR) mô phỏng môi trường phòng thu
    """
    # 1. Time stretch
    if abs(speed_factor - 1.0) > 0.01:
        y_mod = librosa.effects.time_stretch(y, rate=speed_factor)
    else:
        y_mod = y.copy()

    # 2. Pitch shift
    if abs(pitch_shift) > 0.1:
        y_mod = librosa.effects.pitch_shift(y_mod, sr=sr, n_steps=pitch_shift)

    # 3. Micro background room noise (SNR ~ 45dB) để HMM học bền vững
    rms = np.sqrt(np.mean(y_mod ** 2)) + 1e-8
    noise_amp = rms * (10 ** (-45.0 / 20.0))
    noise = np.random.normal(0, noise_amp, len(y_mod)).astype(np.float32)
    y_mod = y_mod + noise

    return y_mod

def build_utterance(base_speech: np.ndarray, sr: int, pitch_shift: float, speed_factor: float,
                    sil_start_sec: float, sil_end_sec: float) -> tuple[np.ndarray, float, float]:
    """
    Xây dựng một utterance hoàn chỉnh:
    - Áp dụng biến thể âm thanh
    - Thêm khoảng lặng đầu và cuối từ 0.2s - 0.5s theo chuẩn
    - Chuẩn hóa biên độ (peak normalize)
    """
    # Xử lý phần tiếng nói
    speech = add_acoustic_variations(base_speech, sr, pitch_shift, speed_factor)

    # Tạo khoảng lặng đầu và cuối có thêm nhiễu sàn vi mô (micro room ambiance)
    n_start = int(round(sil_start_sec * sr))
    n_end = int(round(sil_end_sec * sr))

    amb_noise_level = 1e-5
    sil_start = np.random.normal(0, amb_noise_level, n_start).astype(np.float32)
    sil_end = np.random.normal(0, amb_noise_level, n_end).astype(np.float32)

    # Ghép chuỗi âm thanh
    full_audio = np.concatenate([sil_start, speech, sil_end])

    # Peak normalization về mức -1 dBFS (~0.89) tránh vỡ âm
    max_val = np.max(np.abs(full_audio))
    if max_val > 0:
        full_audio = full_audio / max_val * 0.89

    return full_audio, sil_start_sec, sil_end_sec

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    print("=" * 70)
    print("KHỞI TẠO BỘ DỮ LIỆU TIẾNG NÓI CHO HMM (LAB 03)")
    print("Yêu cầu: WAV Mono, 16 kHz, 16-bit PCM, Silence [0.2s - 0.5s]")
    print("Bộ từ: 'mở', 'đóng', 'lên', 'xuống', 'dừng'")
    print("Mỗi người nói: 8 mẫu (6 train, 2 test)")
    print("=" * 70)

    # 1. Tải trước các base speech từ gTTS cho 5 từ vựng
    print("\n[Bước 1/3] Đang sinh mẫu âm thanh gốc bằng gTTS cho 5 lệnh...")
    base_audio_dict = {}
    for item in VOCABULARY:
        label = item["label"]
        text = item["text"]
        print(f"  -> Sinh từ: '{text}' (label: {label})...")
        raw_audio = generate_base_audio_gtts(text, sr=TARGET_SR)
        trimmed_audio = trim_silence(raw_audio)
        base_audio_dict[label] = trimmed_audio

    # 2. Sinh tập dữ liệu cho từng Speaker
    print("\n[Bước 2/3] Đang tạo các mẫu utterance cho 2 Speaker với phân bố Train/Test...")
    metadata_records = []
    total_created = 0

    # Cố định seed một phần nhưng vẫn giữ đa dạng mẫu
    np.random.seed(42)
    random.seed(42)

    for speaker_id, spk_info in SPEAKERS.items():
        print(f"\n--- Đang xử lý {speaker_id}: {spk_info['desc']} ---")
        
        for item in VOCABULARY:
            label = item["label"]
            text = item["text"]
            base_audio = base_audio_dict[label]

            total_samples = spk_info["train_samples"] + spk_info["test_samples"] # 6 + 2 = 8
            
            for idx in range(1, total_samples + 1):
                # Xác định phân chia tập train (1-6) và test (7-8)
                if idx <= spk_info["train_samples"]:
                    split = "train"
                else:
                    split = "test"

                # Tham số biến thiên âm học cho mẫu này
                p_min, p_max = spk_info["pitch_shift_range"]
                s_min, s_max = spk_info["speed_range"]
                
                # Tạo pitch shift và speed ngẫu nhiên trong khoảng đặc trưng của speaker
                pitch_shift = np.random.uniform(p_min, p_max)
                speed_factor = np.random.uniform(s_min, s_max)

                # Khoảng lặng ngẫu nhiên độc lập trong khoảng [0.20s, 0.50s]
                sil_start = round(float(np.random.uniform(MIN_SILENCE_SEC, MAX_SILENCE_SEC)), 3)
                sil_end = round(float(np.random.uniform(MIN_SILENCE_SEC, MAX_SILENCE_SEC)), 3)

                # Tạo âm thanh hoàn chỉnh
                audio_data, actual_sil_start, actual_sil_end = build_utterance(
                    base_speech=base_audio,
                    sr=TARGET_SR,
                    pitch_shift=pitch_shift,
                    speed_factor=speed_factor,
                    sil_start_sec=sil_start,
                    sil_end_sec=sil_end
                )

                # Đường dẫn lưu file
                # Cấu trúc: dataset/<speaker>/<split>/<label>/<filename>.wav
                target_folder = os.path.join(dataset_dir, speaker_id, split, label)
                os.makedirs(target_folder, exist_ok=True)

                filename = f"{label}_{speaker_id}_{split}_{idx:02d}.wav"
                file_path = os.path.join(target_folder, filename)

                # Lưu với định dạng 16-bit PCM WAV
                sf.write(file_path, audio_data, TARGET_SR, subtype=BIT_DEPTH)
                total_created += 1

                # Tính thời lượng tổng
                duration = round(len(audio_data) / TARGET_SR, 3)

                # Lưu thông tin vào metadata
                rel_path = os.path.relpath(file_path, base_dir).replace("\\", "/")
                metadata_records.append({
                    "file_path": rel_path,
                    "filename": filename,
                    "speaker": speaker_id,
                    "label": label,
                    "word": text,
                    "split": split,
                    "utterance_idx": idx,
                    "duration_sec": duration,
                    "silence_start_sec": actual_sil_start,
                    "silence_end_sec": actual_sil_end,
                    "sample_rate": TARGET_SR,
                    "channels": TARGET_CHANNELS,
                    "bit_depth": BIT_DEPTH
                })

                print(f"  [OK] Đã tạo: {rel_path} ({duration:.2f}s | Sil: {actual_sil_start:.2f}s-{actual_sil_end:.2f}s)")

    # 3. Xuất file metadata.csv
    print("\n[Bước 3/3] Xuất file thông tin dataset (metadata.csv)...")
    metadata_file = os.path.join(dataset_dir, "metadata.csv")
    fieldnames = [
        "file_path", "filename", "speaker", "label", "word", "split",
        "utterance_idx", "duration_sec", "silence_start_sec", "silence_end_sec",
        "sample_rate", "channels", "bit_depth"
    ]
    with open(metadata_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_records)

    print("=" * 70)
    print(f"HOÀN THÀNH TẠO DỮ LIỆU:")
    print(f"- Tổng số file tạo thành công: {total_created} files WAV")
    print(f"- File metadata lưu tại: {metadata_file}")
    print(f"- Phân chia mỗi người nói: 6 train + 2 test cho mỗi từ (tổng 40 file/người nói)")
    print(f"- Toàn bộ file đều đảm bảo WAV Mono, 16kHz, 16-bit PCM, khoảng lặng [0.2 - 0.5]s")
    print("=" * 70)

if __name__ == "__main__":
    main()
