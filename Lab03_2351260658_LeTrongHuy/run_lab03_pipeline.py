"""
run_lab03_pipeline.py
Kịch bản trung tâm thực hiện toàn bộ 8 quy trình A-H, 6 thí nghiệm E1-E6
và tự động sinh toàn bộ Jupyter Notebook (.ipynb) kèm hình ảnh, báo cáo.
Sinh viên: Lê Trọng Huy - MSSV: 2351260658 (hoặc 2351263658)
"""

import os
import sys
import glob
import time
import json
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nbformat as nbf

# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FIG_DIR = os.path.join(BASE_DIR, "figures")
EXP_DIR = os.path.join(BASE_DIR, "ThiNghiem_Va_LyThuyet")
EXP_FIG_DIR = os.path.join(EXP_DIR, "figures")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EXP_FIG_DIR, exist_ok=True)

VOCAB = ["mo", "dong", "len", "xuong", "dung"]
VOCAB_VI = {"mo": "Mở", "dong": "Đóng", "len": "Lên", "xuong": "Xuống", "dung": "Dừng"}

# =============================================================================
# 1. CÁC HÀM XỬ LÝ ÂM THANH & TRÍCH XUẤT ĐẶC TRƯNG (YÊU CẦU A)
# =============================================================================
def pre_emphasis(signal: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """Tiền nhấn tín hiệu tiếng nói: y[n] = x[n] - alpha * x[n-1]"""
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def extract_39d_features(audio_path: str, apply_cmvn: bool = True, sr: int = 16000) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Quy trình A:
    1. Đọc WAV 16kHz mono
    2. Cắt silence qua librosa.effects.trim
    3. Tiền nhấn (pre-emphasis alpha=0.97)
    4. Trích xuất 13 MFCC (win_length=400 ~25ms, hop_length=160 ~10ms, n_mels=26)
    5. Tính Delta và Delta-Delta
    6. Ghép thành ma trận T x 39
    7. Áp dụng CMVN theo utterance
    """
    y, orig_sr = sf.read(audio_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    # 1. Cắt silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < 400:
        y_trimmed = y

    # 2. Tiền nhấn
    y_emp = pre_emphasis(y_trimmed, alpha=0.97)

    # 3. Trích xuất 13 MFCC cơ sở
    mfcc = librosa.feature.mfcc(
        y=y_emp,
        sr=sr,
        n_mfcc=13,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=26,
        window="hamming"
    ) # shape: (13, T)

    # 4. Tính Delta và Delta-Delta
    delta1 = librosa.feature.delta(mfcc, order=1, width=5)
    delta2 = librosa.feature.delta(mfcc, order=2, width=5)

    # 5. Ghép nối tạo vector 39 chiều
    features_39 = np.concatenate([mfcc, delta1, delta2], axis=0).T # (T, 39)

    # 6. CMVN (Cepstral Mean and Variance Normalization)
    if apply_cmvn and features_39.shape[0] > 1:
        mean = np.mean(features_39, axis=0, keepdims=True)
        std = np.std(features_39, axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        features_39_norm = (features_39 - mean) / std
    else:
        features_39_norm = features_39

    return features_39_norm, mfcc.T, delta1.T, delta2.T

def plot_feature_heatmaps():
    """Vẽ Heatmap MFCC / Delta / Delta2 của từ 'mở' và 'đóng' theo Yêu cầu A"""
    print("[Quy trình A] Vẽ Heatmaps đặc trưng âm học cho từ 'mở' và 'đóng'...")
    file_mo = glob.glob(os.path.join(DATASET_DIR, "speaker1", "train", "mo", "*.wav"))[0]
    file_dong = glob.glob(os.path.join(DATASET_DIR, "speaker1", "train", "dong", "*.wav"))[0]

    for fpath, label_vi, fname in [(file_mo, "Mở", "heatmap_features_mo.png"), (file_dong, "Đóng", "heatmap_features_dong.png")]:
        feat_39, mfcc, d1, d2 = extract_39d_features(fpath, apply_cmvn=True)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # 13 MFCC
        im0 = axes[0].imshow(mfcc.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"13 MFCCs cơ sở - Lệnh '{label_vi}'", fontsize=12, fontweight='bold')
        axes[0].set_ylabel("Hệ số MFCC (0-12)")
        fig.colorbar(im0, ax=axes[0])

        # 13 Delta
        im1 = axes[1].imshow(d1.T, aspect='auto', origin='lower', cmap='plasma')
        axes[1].set_title(f"13 Delta (Vận tốc biến thiên phổ) - Lệnh '{label_vi}'", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Hệ số Delta (0-12)")
        fig.colorbar(im1, ax=axes[1])

        # 13 Delta-Delta
        im2 = axes[2].imshow(d2.T, aspect='auto', origin='lower', cmap='magma')
        axes[2].set_title(f"13 Delta-Delta (Gia tốc biến thiên phổ) - Lệnh '{label_vi}'", fontsize=12, fontweight='bold')
        axes[2].set_ylabel("Hệ số Delta-Delta (0-12)")
        axes[2].set_xlabel("Chỉ số khung thời gian (Frames, 10ms/hop)")
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        out_path = os.path.join(FIG_DIR, fname)
        plt.savefig(out_path, dpi=200)
        plt.savefig(os.path.join(EXP_FIG_DIR, fname), dpi=200)
        plt.close()
        print(f"  -> Đã lưu: {out_path}")

# =============================================================================
# 2. KHỞI TẠO & HUẤN LUYỆN HMM TRÁI-SANG-PHẢI (YÊU CẦU B & C)
# =============================================================================
def init_left_to_right_hmm(n_states: int = 5, self_loop_prob: float = 0.7, covar_type: str = "diag") -> hmm.GaussianHMM:
    """
    Quy trình B: Khởi tạo HMM trái-sang-phải:
    - pi = [1, 0, ..., 0]
    - a_ii ~ 0.7; a_i,i+1 = 1 - a_ii
    - a_nn = 1.0
    - Không chuyển ngược (a_ij = 0 với j < i), không nhảy cóc
    """
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=covar_type,
        n_iter=60,
        min_covar=1e-3,
        random_state=42,
        init_params="stmc",
        params="stmc"
    )
    
    # 1. Khởi tạo xác suất trạng thái ban đầu pi
    startprob = np.zeros(n_states)
    startprob[0] = 1.0

    # 2. Khởi tạo ma trận chuyển trạng thái Left-to-Right
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i == n_states - 1:
            transmat[i, i] = 1.0
        else:
            transmat[i, i] = self_loop_prob
            transmat[i, i + 1] = 1.0 - self_loop_prob

    # Thêm pseudo-count cực nhỏ để tránh lỗi zero-sum row
    transmat = transmat + 1e-6
    transmat = transmat / transmat.sum(axis=1, keepdims=True)

    model.startprob_ = startprob
    model.transmat_ = transmat
    return model

def train_hmms_for_all_words(train_data: dict[str, list[np.ndarray]], n_states: int = 5, covar_type: str = "diag") -> dict[str, hmm.GaussianHMM]:
    """Quy trình C: Nối các utterance train thành X_concat, truyền lengths vào fit()"""
    models = {}
    for word in VOCAB:
        feats = train_data[word]
        X_concat = np.concatenate(feats, axis=0)
        lengths = [f.shape[0] for f in feats]

        model = init_left_to_right_hmm(n_states=n_states, self_loop_prob=0.7, covar_type=covar_type)
        model.fit(X_concat, lengths=lengths)
        models[word] = model

    return models

# =============================================================================
# 3. THUẬT TOÁN DTW BASELINE (YÊU CẦU E)
# =============================================================================
def dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Tính khoảng cách DTW (Dynamic Time Warping) giữa hai chuỗi vector đặc trưng.
    Khoảng cách cơ sở: Khoảng cách Euclidean giữa các frames.
    Quy hoạch động: D[i, j] = dist + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
    """
    n, m = len(x), len(y)
    # Ma trận chi phí Euclidean
    cost_matrix = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=2)

    # Khởi tạo ma trận quy hoạch động
    dp = np.full((n + 1, m + 1), float('inf'))
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Chuẩn hóa khoảng cách theo tổng độ dài path
    return dp[n, m] / (n + m)

def predict_dtw(test_feat: np.ndarray, train_templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    """Phân loại 1-NN sử dụng khoảng cách DTW."""
    best_dist = float('inf')
    best_word = None
    for word, templates in train_templates.items():
        for tpl in templates:
            dist = dtw_distance(test_feat, tpl)
            if dist < best_dist:
                best_dist = dist
                best_word = word
    return best_word, best_dist

# =============================================================================
# 4. BIGRAM LANGUAGE MODEL & MAP RE-RANKING (YÊU CẦU G)
# =============================================================================
class BigramLanguageModel:
    def __init__(self, corpus_path: str, add_k: float = 0.5):
        self.add_k = add_k
        self.unigrams = {}
        self.bigrams = {}
        self.vocab = set()
        self._train(corpus_path)

    def _train(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip().lower() for line in f if line.strip()]

        for line in lines:
            words = line.split()
            for w in words:
                self.vocab.add(w)
                self.unigrams[w] = self.unigrams.get(w, 0) + 1

            for w1, w2 in zip(words[:-1], words[1:]):
                bg = (w1, w2)
                self.bigrams[bg] = self.bigrams.get(bg, 0) + 1

    def log_prob(self, prev_word: str, word: str) -> float:
        """P(w | prev_w) với Add-k Smoothing: (C(w_prev, w) + k) / (C(w_prev) + k * |V|)"""
        prev_w = prev_word.lower()
        w = word.lower()
        V_size = len(self.vocab)
        count_prev = self.unigrams.get(prev_w, 0)
        count_bg = self.bigrams.get((prev_w, w), 0)

        prob = (count_bg + self.add_k) / (count_prev + self.add_k * V_size)
        return float(np.log(prob))

# =============================================================================
# 5. CHẠY TOÀN BỘ CÁC THỰC NGHIỆM VÀ XUẤT ĐỒ THỊ
# =============================================================================
def run_all_experiments():
    print("=" * 80)
    print("BẮT ĐẦU CHẠY TOÀN BỘ CÁC BƯỚC THÍ NGHIỆM LAB 03 (LÀM - SAI - SỬA)")
    print("=" * 80)

    # 1. Trích xuất đặc trưng cho toàn bộ tập dữ liệu
    print("\n[Bước 1] Đang nạp tập dữ liệu và trích xuất đặc trưng 39D...")
    data_train = {w: [] for w in VOCAB}
    data_test = {w: [] for w in VOCAB}
    data_test_mismatch = {w: [] for w in VOCAB} # Speaker 2 dùng làm mismatch test

    # Speaker 1 train
    for w in VOCAB:
        for f in glob.glob(os.path.join(DATASET_DIR, "speaker1", "train", w, "*.wav")):
            feat, _, _, _ = extract_39d_features(f, apply_cmvn=True)
            data_train[w].append(feat)

    # Speaker 1 test (clean test)
    for w in VOCAB:
        for f in glob.glob(os.path.join(DATASET_DIR, "speaker1", "test", w, "*.wav")):
            feat, _, _, _ = extract_39d_features(f, apply_cmvn=True)
            data_test[w].append(feat)

    # Speaker 2 test (mismatch test)
    for w in VOCAB:
        for f in glob.glob(os.path.join(DATASET_DIR, "speaker2", "test", w, "*.wav")):
            feat, _, _, _ = extract_39d_features(f, apply_cmvn=True)
            data_test_mismatch[w].append(feat)

    print(f"  -> Đã nạp: Train (Speaker 1) = {sum(len(v) for v in data_train.values())} mẫu")
    print(f"  -> Đã nạp: Clean Test (Speaker 1) = {sum(len(v) for v in data_test.values())} mẫu")
    print(f"  -> Đã nạp: Mismatch Test (Speaker 2) = {sum(len(v) for v in data_test_mismatch.values())} mẫu")

    # 2. Vẽ Heatmaps (Yêu cầu A)
    plot_feature_heatmaps()

    # 3. Huấn luyện HMM (Yêu cầu B & C)
    print("\n[Bước 2] Huấn luyện mô hình Left-to-Right HMM (n_states=5)...")
    hmm_models = train_hmms_for_all_words(data_train, n_states=5, covar_type="diag")
    for w in VOCAB:
        tm = hmm_models[w].transmat_
        print(f"  - Ma trận transmat_ của từ '{w}':")
        print(np.round(tm, 3))

    # 4. Viterbi Alignment & Step Plot (Yêu cầu D)
    print("\n[Bước 3] Thực hiện Viterbi Alignment và vẽ đồ thị State Sequence...")
    for target_word in ["mo", "dong"]:
        sample_feat = data_test[target_word][0]
        model = hmm_models[target_word]
        viterbi_logprob, state_path = model.decode(sample_feat, algorithm="viterbi")
        
        plt.figure(figsize=(9, 4.5))
        frames = np.arange(len(state_path))
        plt.step(frames, state_path, where='post', color='#1f77b4', linewidth=2.5, label='Trạng thái ẩn (State)')
        plt.scatter(frames, state_path, color='#d62728', s=25, zorder=3)
        plt.title(f"Viterbi Alignment Path - Lệnh '{VOCAB_VI[target_word]}' (Log-Likelihood: {viterbi_logprob:.1f})", fontsize=13, fontweight='bold')
        plt.xlabel("Chỉ số khung thời gian (Frame Index, 10ms/frame)", fontsize=11)
        plt.ylabel("Trạng thái HMM (States: 0 - 4)", fontsize=11)
        plt.yticks(np.arange(5), [f"S{i}" for i in range(5)])
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()

        out_viterbi = os.path.join(FIG_DIR, f"viterbi_path_{target_word}.png")
        plt.savefig(out_viterbi, dpi=200)
        plt.savefig(os.path.join(EXP_FIG_DIR, f"viterbi_path_{target_word}.png"), dpi=200)
        plt.close()
        print(f"  -> Đã lưu Viterbi path: {out_viterbi}")

    # Bảng điểm Likelihood của một mẫu kiểm thử
    print("\n[Bảng điểm Likelihood minh họa quyết định nhận dạng của test utterance 'mo_test_01']:")
    test_sample = data_test["mo"][0]
    scores_dict = {w: hmm_models[w].score(test_sample) for w in VOCAB}
    print(f"{'Mô hình HMM':<15} | {'Log-Likelihood':<18} | {'Normalized (LL/T)':<18} | {'Quyết định'}")
    print("-" * 65)
    best_w = max(scores_dict, key=scores_dict.get)
    for w in VOCAB:
        s = scores_dict[w]
        sn = s / len(test_sample)
        marker = " <== [CHỌN]" if w == best_w else ""
        print(f"{VOCAB_VI[w]:<15} | {s:>18.2f} | {sn:>18.2f} |{marker}")

    # 5. Đánh giá HMM trên Clean Test và Mismatch Test (Yêu cầu H)
    print("\n[Bước 4] Đánh giá HMM trên Clean Test và Mismatch Test...")
    y_true_clean, y_pred_clean = [], []
    for w in VOCAB:
        for f in data_test[w]:
            scores = {l: hmm_models[l].score(f) for l in VOCAB}
            pred = max(scores, key=scores.get)
            y_true_clean.append(w)
            y_pred_clean.append(pred)

    acc_clean = accuracy_score(y_true_clean, y_pred_clean)
    cm_clean = confusion_matrix(y_true_clean, y_pred_clean, labels=VOCAB)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_clean, annot=True, fmt="d", cmap="Blues", xticklabels=[VOCAB_VI[w] for w in VOCAB], yticklabels=[VOCAB_VI[w] for w in VOCAB])
    plt.title(f"HMM Confusion Matrix - Clean Test (Acc: {acc_clean*100:.1f}%)", fontweight="bold")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cm_clean_hmm.png"), dpi=200)
    plt.savefig(os.path.join(EXP_FIG_DIR, "cm_clean_hmm.png"), dpi=200)
    plt.close()

    # Mismatch Test
    y_true_mis, y_pred_mis = [], []
    for w in VOCAB:
        for f in data_test_mismatch[w]:
            scores = {l: hmm_models[l].score(f) for l in VOCAB}
            pred = max(scores, key=scores.get)
            y_true_mis.append(w)
            y_pred_mis.append(pred)

    acc_mis = accuracy_score(y_true_mis, y_pred_mis)
    cm_mis = confusion_matrix(y_true_mis, y_pred_mis, labels=VOCAB)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_mis, annot=True, fmt="d", cmap="Reds", xticklabels=[VOCAB_VI[w] for w in VOCAB], yticklabels=[VOCAB_VI[w] for w in VOCAB])
    plt.title(f"HMM Confusion Matrix - Mismatch Test (Acc: {acc_mis*100:.1f}%)", fontweight="bold")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cm_mismatch_hmm.png"), dpi=200)
    plt.savefig(os.path.join(EXP_FIG_DIR, "cm_mismatch_hmm.png"), dpi=200)
    plt.close()

    print(f"  -> Accuracy Clean Test: {acc_clean*100:.2f}%")
    print(f"  -> Accuracy Mismatch Test (Speaker 2): {acc_mis*100:.2f}%")

    # 6. So sánh DTW Baseline (Yêu cầu E & E4)
    print("\n[Bước 5] Đánh giá DTW Baseline và đo thời gian thực thi...")
    start_dtw = time.time()
    y_true_dtw, y_pred_dtw = [], []
    for w in VOCAB:
        for f in data_test[w]:
            pred_w, _ = predict_dtw(f, data_train)
            y_true_dtw.append(w)
            y_pred_dtw.append(pred_w)
    time_dtw = time.time() - start_dtw
    acc_dtw = accuracy_score(y_true_dtw, y_pred_dtw)
    cm_dtw = confusion_matrix(y_true_dtw, y_pred_dtw, labels=VOCAB)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_dtw, annot=True, fmt="d", cmap="Greens", xticklabels=[VOCAB_VI[w] for w in VOCAB], yticklabels=[VOCAB_VI[w] for w in VOCAB])
    plt.title(f"DTW Baseline Confusion Matrix (Acc: {acc_dtw*100:.1f}%)", fontweight="bold")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "cm_dtw.png"), dpi=200)
    plt.savefig(os.path.join(EXP_FIG_DIR, "cm_dtw.png"), dpi=200)
    plt.close()

    # Đo thời gian inference HMM
    start_hmm = time.time()
    for w in VOCAB:
        for f in data_test[w]:
            _ = {l: hmm_models[l].score(f) for l in VOCAB}
    time_hmm = time.time() - start_hmm

    # Đồ thị so sánh DTW vs HMM
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    methods = ['DTW Baseline', 'HMM Left-to-Right']
    accs = [acc_dtw * 100, acc_clean * 100]
    times = [time_dtw * 1000 / len(y_true_dtw), time_hmm * 1000 / len(y_true_clean)] # ms per sample

    x_pos = np.arange(len(methods))
    bar1 = ax1.bar(x_pos - 0.2, accs, width=0.4, color='#2ca02c', label='Độ chính xác (%)')
    ax1.set_ylabel('Độ chính xác (%)', color='#2ca02c', fontsize=11)
    ax1.set_ylim(0, 115)

    ax2 = ax1.twinx()
    bar2 = ax2.bar(x_pos + 0.2, times, width=0.4, color='#1f77b4', label='Thời gian / mẫu (ms)')
    ax2.set_ylabel('Thời gian suy luận / mẫu (ms)', color='#1f77b4', fontsize=11)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, fontsize=11, fontweight='bold')
    plt.title("So sánh Hiệu năng: DTW Baseline vs. HMM", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "dtw_vs_hmm_comparison.png"), dpi=200)
    plt.savefig(os.path.join(EXP_FIG_DIR, "dtw_vs_hmm_comparison.png"), dpi=200)
    plt.close()

    print(f"  -> DTW Accuracy: {acc_dtw*100:.2f}% | Thời gian: {time_dtw*1000/len(y_true_dtw):.2f} ms/mẫu")
    print(f"  -> HMM Accuracy: {acc_clean*100:.2f}% | Thời gian: {time_hmm*1000/len(y_true_clean):.2f} ms/mẫu")

    # 7. Thí nghiệm E1: n_states in {3, 5, 7}
    print("\n[Bước 6] Thí nghiệm E1: So sánh n_states = 3, 5, 7...")
    n_states_list = [3, 5, 7]
    acc_by_states = []
    for ns in n_states_list:
        mdls = train_hmms_for_all_words(data_train, n_states=ns, covar_type="diag")
        preds = []
        for w in VOCAB:
            for f in data_test[w]:
                sc = {l: mdls[l].score(f) for l in VOCAB}
                preds.append(max(sc, key=sc.get))
        a = accuracy_score(y_true_clean, preds)
        acc_by_states.append(a * 100)
        print(f"  - n_states = {ns}: Accuracy = {a*100:.1f}%")

    plt.figure(figsize=(6, 4))
    plt.plot(n_states_list, acc_by_states, marker='o', linewidth=2, color='#e377c2', markersize=8)
    plt.title("Thí nghiệm E1: Ảnh hưởng của số trạng thái n_states đến Accuracy", fontweight="bold")
    plt.xlabel("Số lượng trạng thái ẩn (n_states)")
    plt.ylabel("Độ chính xác (%)")
    plt.ylim(0, 110)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_FIG_DIR, "e1_nstates_comparison.png"), dpi=200)
    plt.close()

    # 8. Thí nghiệm E2: 13 MFCC tĩnh vs 39 MFCC động
    print("\n[Bước 7] Thí nghiệm E2: So sánh 13 MFCC tĩnh vs 39 MFCC động...")
    # Trích xuất 13 MFCC tĩnh
    data_train_13 = {w: [f[:, :13] for f in data_train[w]] for w in VOCAB}
    data_test_13 = {w: [f[:, :13] for f in data_test[w]] for w in VOCAB}
    models_13 = train_hmms_for_all_words(data_train_13, n_states=5, covar_type="diag")

    preds_13 = []
    for w in VOCAB:
        for f in data_test_13[w]:
            sc = {l: models_13[l].score(f) for l in VOCAB}
            preds_13.append(max(sc, key=sc.get))
    acc_13 = accuracy_score(y_true_clean, preds_13)

    plt.figure(figsize=(6, 4))
    plt.bar(["13 MFCC tĩnh", "39 MFCC động (+Δ, Δ²)"], [acc_13 * 100, acc_clean * 100], color=['#ff7f0e', '#1f77b4'], width=0.5)
    plt.title("Thí nghiệm E2: 13 MFCC tĩnh vs. 39 MFCC động", fontweight="bold")
    plt.ylabel("Độ chính xác (%)")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_FIG_DIR, "e2_13static_vs_39dynamic.png"), dpi=200)
    plt.close()
    print(f"  -> Accuracy 13 MFCC tĩnh: {acc_13*100:.2f}% | 39 MFCC động: {acc_clean*100:.2f}%")

    # 9. Thí nghiệm E3: Diag vs Full Covariance
    print("\n[Bước 8] Thí nghiệm E3: Diag vs Full Covariance...")
    try:
        models_full = train_hmms_for_all_words(data_train, n_states=5, covar_type="full")
        preds_full = []
        for w in VOCAB:
            for f in data_test[w]:
                sc = {l: models_full[l].score(f) for l in VOCAB}
                preds_full.append(max(sc, key=sc.get))
        acc_full = accuracy_score(y_true_clean, preds_full)
        status_full = f"Thành công - Accuracy = {acc_full*100:.1f}%"
    except Exception as e:
        status_full = f"Suy biến ma trận (Singular Covariance Matrix): {str(e)[:60]}"
    print(f"  -> Kết quả Full Covariance: {status_full}")

    # 10. Thí nghiệm G & E6: Bigram MAP Re-ranking
    print("\n[Bước 9] Thí nghiệm G & E6: Bigram Language Model và MAP Re-ranking...")
    lm = BigramLanguageModel(os.path.join(BASE_DIR, "corpus_commands.txt"), add_k=0.5)
    # Giả lập ngữ cảnh: từ đứng trước là "cửa", ta muốn phân biệt giữa "mở" và "đóng"
    # Hoặc từ đứng trước là "hãy" -> P(mở | hãy) vs P(dừng | hãy)
    lambda_list = [0.0, 0.1, 0.3, 1.0]
    rerank_demo = []
    # Chọn 1 mẫu có acoustic score gần nhau
    test_sample_dong = data_test["dong"][0]
    T = len(test_sample_dong)
    prev_word = "cửa" # trong ngữ cảnh "đóng cửa" rất phổ biến

    for lm_weight in lambda_list:
        scores_map = {}
        for w in VOCAB:
            ac_score = hmm_models[w].score(test_sample_dong) / T
            lm_score = lm.log_prob(prev_word, VOCAB_VI[w])
            total = ac_score + lm_weight * lm_score
            scores_map[w] = total
        best_decision = max(scores_map, key=scores_map.get)
        rerank_demo.append((lm_weight, best_decision, scores_map))
        print(f"  - lambda_LM = {lm_weight:<4} -> Dự đoán: {VOCAB_VI[best_decision]} (Top score: {scores_map[best_decision]:.2f})")

    # 11. Thí nghiệm H: Reject Threshold cho từ ngoài từ vựng (OOV)
    print("\n[Bước 10] Thí nghiệm H: Phân tích phân bố Score và Reject Threshold OOV...")
    in_vocab_scores = []
    for w in VOCAB:
        for f in data_test[w]:
            s_norm = hmm_models[w].score(f) / len(f)
            in_vocab_scores.append(s_norm)

    # Sinh mẫu giả lập Out-of-Vocabulary (nhiễu hoặc âm thanh lạ như từ 'bật', 'tắt')
    np.random.seed(42)
    oov_scores = []
    for _ in range(20):
        # Nhiễu âm thanh ngẫu nhiên mô phỏng OOV
        fake_oov = np.random.normal(0, 1.5, (60, 39))
        best_oov_score = max(hmm_models[w].score(fake_oov) / len(fake_oov) for w in VOCAB)
        oov_scores.append(best_oov_score)

    plt.figure(figsize=(7, 4.5))
    plt.hist(in_vocab_scores, bins=10, alpha=0.7, color='green', label='Trong từ vựng (In-Vocab Đúng Lớp)')
    plt.hist(oov_scores, bins=10, alpha=0.7, color='red', label='Ngoài từ vựng (OOV / Sai Lớp)')
    
    # Ngưỡng đề xuất threshold
    threshold = (np.mean(in_vocab_scores) + np.mean(oov_scores)) / 2.0
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Ngưỡng từ chối (Threshold = {threshold:.1f})')
    plt.title("Phân bố Log-Likelihood / Frame: In-Vocabulary vs. OOV", fontweight="bold")
    plt.xlabel("Normalized Log-Likelihood (LL / T)")
    plt.ylabel("Tần suất (Số mẫu)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(EXP_FIG_DIR, "oov_score_distribution.png"), dpi=200)
    plt.close()
    print(f"  -> Ngưỡng đề xuất từ chối OOV (Reject Threshold): {threshold:.2f}")

    print("\n" + "=" * 80)
    print("HOÀN TẤT THỰC THI TOÀN BỘ CÁC THỰC NGHIỆM LAB 03!")
    print("=" * 80)
    return {
        "acc_clean": acc_clean,
        "acc_mis": acc_mis,
        "acc_dtw": acc_dtw,
        "time_dtw": time_dtw * 1000 / len(y_true_dtw),
        "time_hmm": time_hmm * 1000 / len(y_true_clean),
        "acc_13": acc_13,
        "acc_by_states": acc_by_states,
        "threshold": threshold
    }

if __name__ == "__main__":
    results = run_all_experiments()
