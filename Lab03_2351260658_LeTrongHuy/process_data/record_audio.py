"""
record_audio.py
Script hỗ trợ người dùng ghi âm trực tiếp tiếng nói qua Microphone theo chuẩn của đề bài:
- Format: WAV mono (1 channel)
- Sample Rate: 16000 Hz
- Độ sâu bit: 16-bit PCM
- Tự động hướng dẫn người dùng để khoảng lặng 0.2 - 0.5s ở đầu và cuối.
Sinh viên: Lê Trọng Huy - MSSV: 2351260658
"""

import os
import sys
import time
import wave
import numpy as np

# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def record_with_sounddevice(output_path: str, duration_sec: float = 2.0, sr: int = 16000):
    try:
        import sounddevice as sd
    except ImportError:
        print("[THÔNG BÁO] Thư viện 'sounddevice' chưa được cài đặt.")
        print("Để ghi âm trực tiếp bằng Microphone, bạn hãy chạy: pip install sounddevice")
        return False

    print(f"\n[CHUẨN BỊ GHI ÂM]: Thời lượng {duration_sec}s")
    print("QUY TẮC BẮT BUỘC:")
    print("1. Sau khi bắt đầu, HÃY GIỮ IM LẶNG khoảng 0.3s.")
    print("2. Phát âm rõ ràng một trong 5 từ: 'mở', 'đóng', 'lên', 'xuống', 'dừng'.")
    print("3. Tiếp tục GIỮ IM LẶNG khoảng 0.3s trước khi kết thúc.")
    print("\nChuẩn bị trong 3... 2... 1...")
    time.sleep(1)
    print(">>> ĐANG GHI ÂM... NÓI NGAY!")
    
    audio_data = sd.rec(int(duration_sec * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    print(">>> GHI ÂM HOÀN TẤT!")

    # Lưu file WAV PCM 16-bit
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio_data.tobytes())

    print(f"[THÀNH CÔNG] File âm thanh đã lưu tại: {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        out_file = sys.argv[1]
    else:
        out_file = "my_voice_sample.wav"
    record_with_sounddevice(out_file)
