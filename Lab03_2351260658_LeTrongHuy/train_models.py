"""
train_models.py
Script chuyên biệt để huấn luyện và LƯU TOÀN BỘ CÁC FILE MODEL HMM ra đĩa.
Sinh viên: Lê Trọng Huy - MSSV: 2351260658 (hoặc 2351263658)

Khi chạy script này, các file model sẽ được xuất vào thư mục 'models/':
1. models/hmm_words/ (các model riêng cho từng từ: mo, dong, len, xuong, dung)
2. models/hmm_speaker1.joblib (mô hình của người nói 1)
3. models/hmm_speaker2.joblib (mô hình của người nói 2)
4. models/hmm_speaker_independent.joblib (mô hình gộp đa người nói)
"""

import os
import sys
import glob
import joblib
import numpy as np
import soundfile as sf
import librosa
from hmmlearn import hmm

# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
WORDS_DIR = os.path.join(MODELS_DIR, "hmm_words")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(WORDS_DIR, exist_ok=True)

VOCAB = ["mo", "dong", "len", "xuong", "dung"]
VOCAB_VI = {"mo": "Mở", "dong": "Đóng", "len": "Lên", "xuong": "Xuống", "dung": "Dừng"}

def pre_emphasis(signal, alpha=0.97):
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def extract_39d(audio_path, sr=16000):
    y, orig_sr = sf.read(audio_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trim) < 400:
        y_trim = y
    y_emp = pre_emphasis(y_trim, alpha=0.97)
    mfcc = librosa.feature.mfcc(y=y_emp, sr=sr, n_mfcc=13, n_fft=512, win_length=400, hop_length=160, n_mels=26, window="hamming")
    delta1 = librosa.feature.delta(mfcc, order=1, width=5)
    delta2 = librosa.feature.delta(mfcc, order=2, width=5)
    feat_39 = np.concatenate([mfcc, delta1, delta2], axis=0).T
    if feat_39.shape[0] > 1:
        mean = np.mean(feat_39, axis=0, keepdims=True)
        std = np.std(feat_39, axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        feat_39 = (feat_39 - mean) / std
    return feat_39

def train_single_hmm(features_list, n_states=5, self_loop=0.7):
    X_concat = np.concatenate(features_list, axis=0)
    lengths = [len(f) for f in features_list]
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=60,
        min_covar=1e-3,
        random_state=42,
        init_params="stmc",
        params="stmc"
    )
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == n_states - 1:
            transmat[i, i] = 1.0
        else:
            transmat[i, i] = self_loop
            transmat[i, i + 1] = 1.0 - self_loop
    transmat = (transmat + 1e-6)
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.fit(X_concat, lengths=lengths)
    return model

def main():
    print("=" * 75)
    print("BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN VÀ XUẤT CÁC FILE MODEL HMM RA ĐĨA")
    print("=" * 75)

    # 1. Nạp dữ liệu train
    print("\n[Bước 1/3] Đang nạp và trích xuất đặc trưng 39D cho tập train...")
    spk1_train = {w: [] for w in VOCAB}
    spk2_train = {w: [] for w in VOCAB}
    multi_train = {w: [] for w in VOCAB}

    for w in VOCAB:
        for f in glob.glob(os.path.join(DATASET_DIR, "speaker1", "train", w, "*.wav")):
            feat = extract_39d(f)
            spk1_train[w].append(feat)
            multi_train[w].append(feat)
            
        for f in glob.glob(os.path.join(DATASET_DIR, "speaker2", "train", w, "*.wav")):
            feat = extract_39d(f)
            spk2_train[w].append(feat)
            multi_train[w].append(feat)

    # 2. Huấn luyện và lưu từng từ
    print("\n[Bước 2/3] Huấn luyện và lưu model từng từ vựng vào 'models/hmm_words/'...")
    words_models = {}
    for w in VOCAB:
        model = train_single_hmm(spk1_train[w], n_states=5)
        words_models[w] = model
        model_file = os.path.join(WORDS_DIR, f"{w}_model.joblib")
        joblib.dump(model, model_file)
        print(f"  [ĐÃ LƯU]: {model_file} (Từ: '{VOCAB_VI[w]}')")

    # 3. Huấn luyện và lưu các bộ nhận dạng tổng hợp
    print("\n[Bước 3/3] Huấn luyện và lưu các bộ mô hình theo người nói...")
    # Speaker 1
    spk1_models = {w: train_single_hmm(spk1_train[w], n_states=5) for w in VOCAB}
    spk1_file = os.path.join(MODELS_DIR, "hmm_speaker1.joblib")
    joblib.dump(spk1_models, spk1_file)
    print(f"  [ĐÃ LƯU]: {spk1_file} (Mô hình Speaker 1 - 5 từ)")

    # Speaker 2
    spk2_models = {w: train_single_hmm(spk2_train[w], n_states=5) for w in VOCAB}
    spk2_file = os.path.join(MODELS_DIR, "hmm_speaker2.joblib")
    joblib.dump(spk2_models, spk2_file)
    print(f"  [ĐÃ LƯU]: {spk2_file} (Mô hình Speaker 2 - 5 từ)")

    # Speaker-Independent (Multi-speaker)
    si_models = {w: train_single_hmm(multi_train[w], n_states=5) for w in VOCAB}
    si_file = os.path.join(MODELS_DIR, "hmm_speaker_independent.joblib")
    joblib.dump(si_models, si_file)
    print(f"  [ĐÃ LƯU]: {si_file} (Mô hình Speaker-Independent - 5 từ)")

    print("\n" + "=" * 75)
    print(f"HOÀN THÀNH! Tất cả model đã được lưu tại: {MODELS_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
