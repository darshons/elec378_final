"""
ELEC 378 Final Project — Non-Neural-Network Model (v2)
Pipeline: SPM BoVW (1000 words) + HSV Color + PCA + RBF Kernel SVM

Validation accuracy: 65.96%

Over v1, this version adds Spatial Pyramid Matching (1x1, 2x2,
4x4 = 21 regions) to capture spatial layout of visual words.
Everything else stays the same: 1000-word vocabulary, HSV color,
PCA (95% variance), RBF kernel SVM.

The improvement was marginal, suggesting the bottleneck is not
just spatial encoding — vocabulary size and PCA compression may
also need tuning.

Run:  python pipeline_v2.py
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "butterfly_data")
TRAIN_CSV     = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test_images", "test_images")
SAMPLE_SUB    = os.path.join(BASE_DIR, "sample_submission.csv")

OUTPUT_DIR    = os.path.join(BASE_DIR, "v2_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────
IMG_SIZE       = 256
SIFT_MAX_KP    = 500
VOCAB_SIZE     = 1000       # same vocabulary size as v1
SAMPLE_PER_IMG = 50
PCA_VARIANCE   = 0.95       # same PCA strategy as v1
RANDOM_STATE   = 42

SPM_LEVELS = [1, 2, 4]                          # NEW: spatial pyramid
N_REGIONS  = sum(l * l for l in SPM_LEVELS)      # 21

H_BINS, S_BINS, V_BINS = 32, 32, 32
COLOR_DIM = H_BINS + S_BINS + V_BINS  # 96


# ── Helpers ───────────────────────────────────────────────────

def load_and_resize(path, max_side=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def extract_sift(gray, sift, max_kp=SIFT_MAX_KP):
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        return [], np.empty((0, 128), dtype=np.float32)
    if len(kp) > max_kp:
        indices = np.argsort([-k.response for k in kp])[:max_kp]
        kp = [kp[i] for i in indices]
        des = des[indices]
    return kp, des.astype(np.float32)


def spatial_pyramid_histogram(keypoints, descriptors, kmeans,
                               vocab_size, img_h, img_w, levels):
    """SPM histogram with level-based weighting."""
    n_levels = len(levels)
    total_bins = vocab_size * sum(l * l for l in levels)
    if len(descriptors) == 0:
        return np.zeros(total_bins, dtype=np.float32)

    words = kmeans.predict(descriptors)
    kp_x = np.array([k.pt[0] for k in keypoints])
    kp_y = np.array([k.pt[1] for k in keypoints])

    all_hists = []
    for li, level in enumerate(levels):
        if li == 0:
            weight = 1.0 / (2 ** (n_levels - 1))
        else:
            weight = 1.0 / (2 ** (n_levels - li))
        cell_w = img_w / level
        cell_h = img_h / level
        for row in range(level):
            for col in range(level):
                mask = ((kp_x >= col * cell_w) & (kp_x < (col + 1) * cell_w) &
                        (kp_y >= row * cell_h) & (kp_y < (row + 1) * cell_h))
                cell_words = words[mask]
                hist = np.zeros(vocab_size, dtype=np.float32)
                if len(cell_words) > 0:
                    for w_idx in cell_words:
                        hist[w_idx] += 1
                    hist /= hist.sum()
                all_hists.append(hist * weight)
    return np.concatenate(all_hists)


def extract_hsv_histogram(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [H_BINS], [0, 180]).flatten().astype(np.float32)
    s_hist = cv2.calcHist([hsv], [1], None, [S_BINS], [0, 256]).flatten().astype(np.float32)
    v_hist = cv2.calcHist([hsv], [2], None, [V_BINS], [0, 256]).flatten().astype(np.float32)
    for h in [h_hist, s_hist, v_hist]:
        s = h.sum()
        if s > 0:
            h /= s
    return np.concatenate([h_hist, s_hist, v_hist])


def encode_images(file_list, img_dir, sift, kmeans, kp_data_list=None,
                  desc_label="Encode"):
    """Encode images as SPM BoVW + color feature vectors."""
    total_dim = VOCAB_SIZE * N_REGIONS + COLOR_DIM
    features = []
    for i, fname in enumerate(tqdm(file_list, desc=desc_label)):
        path = os.path.join(img_dir, fname)
        if kp_data_list is not None:
            kp, des, img_h, img_w = kp_data_list[i]
            bgr = cv2.imread(path)
            if bgr is not None:
                h0, w0 = bgr.shape[:2]
                sc = IMG_SIZE / max(h0, w0)
                bgr = cv2.resize(bgr, (int(w0*sc), int(h0*sc)),
                                 interpolation=cv2.INTER_AREA)
        else:
            bgr, gray = load_and_resize(path)
            if gray is None:
                features.append(np.zeros(total_dim, dtype=np.float32))
                continue
            kp, des = extract_sift(gray, sift)
            img_h, img_w = gray.shape[:2]

        spm = spatial_pyramid_histogram(
            kp, des, kmeans, VOCAB_SIZE, img_h, img_w, SPM_LEVELS)
        color = (extract_hsv_histogram(bgr) if bgr is not None
                 else np.zeros(COLOR_DIM, dtype=np.float32))
        features.append(np.concatenate([spm, color]))
    return np.array(features)


# ── Main ──────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 65)
    print("PIPELINE v2: SPM BoVW (1000) + Color + PCA + RBF SVM")
    print("=" * 65)

    # 1. Load metadata and split
    df = pd.read_csv(TRAIN_CSV)
    df.columns = [c.strip() for c in df.columns]
    print(f"\nLoaded {len(df)} entries, {df['TARGET'].nunique()} classes")

    train_files, val_files, train_labels, val_labels = train_test_split(
        df["file_name"].values, df["TARGET"].values,
        test_size=0.2, stratify=df["TARGET"].values, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # 2. SIFT extraction
    print(f"\n--- Phase 1: SIFT extraction ---")
    sift = cv2.SIFT_create(nfeatures=SIFT_MAX_KP)

    train_kp_data = []
    all_sampled_desc = []

    for fname in tqdm(train_files, desc="SIFT (train)"):
        path = os.path.join(TRAIN_IMG_DIR, fname)
        bgr, gray = load_and_resize(path)
        if gray is None:
            train_kp_data.append(([], np.empty((0, 128), dtype=np.float32), 0, 0))
            continue
        kp, des = extract_sift(gray, sift)
        train_kp_data.append((kp, des, gray.shape[0], gray.shape[1]))
        if len(des) > SAMPLE_PER_IMG:
            idx = np.random.RandomState(RANDOM_STATE).choice(
                len(des), SAMPLE_PER_IMG, replace=False)
            all_sampled_desc.append(des[idx])
        elif len(des) > 0:
            all_sampled_desc.append(des)

    sampled = np.vstack(all_sampled_desc)
    print(f"  Sampled descriptors: {sampled.shape}")

    # 3. K-Means vocabulary
    print(f"\n--- Phase 2: K-Means vocabulary (K={VOCAB_SIZE}) ---")
    kmeans = MiniBatchKMeans(
        n_clusters=VOCAB_SIZE, batch_size=4096, random_state=RANDOM_STATE,
        n_init=3, max_iter=300,
    )
    kmeans.fit(sampled)
    print(f"  Inertia: {kmeans.inertia_:.0f}")

    # 4. Feature encoding
    spm_dim = VOCAB_SIZE * N_REGIONS
    total_dim = spm_dim + COLOR_DIM
    print(f"\n--- Phase 3: Feature encoding ({spm_dim} SPM + {COLOR_DIM} color = {total_dim}) ---")

    X_train = encode_images(train_files, TRAIN_IMG_DIR, sift, kmeans,
                            train_kp_data, "Encode (train)")
    X_val   = encode_images(val_files, TRAIN_IMG_DIR, sift, kmeans,
                            desc_label="Encode (val)")
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    # 5. Label encoding
    le = LabelEncoder()
    le.fit(np.concatenate([train_labels, val_labels]))
    y_train = le.transform(train_labels)
    y_val   = le.transform(val_labels)

    # 6. Standardize + PCA
    print(f"\n--- Phase 4: Standardize + PCA ({PCA_VARIANCE*100:.0f}% variance) ---")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_sc)
    X_val_pca   = pca.transform(X_val_sc)
    n_comp = pca.n_components_
    explained = pca.explained_variance_ratio_.sum()
    print(f"  {total_dim} -> {n_comp} dims (explains {explained*100:.2f}% variance)")

    # 7. SVM hyperparameter search
    print(f"\n--- Phase 5: RBF SVM hyperparameter search ---")

    param_grid = [
        {'C': 0.1,  'gamma': 'scale'},
        {'C': 1,    'gamma': 'scale'},
        {'C': 10,   'gamma': 'scale'},
        {'C': 100,  'gamma': 'scale'},
        {'C': 0.1,  'gamma': 'auto'},
        {'C': 1,    'gamma': 'auto'},
        {'C': 10,   'gamma': 'auto'},
        {'C': 100,  'gamma': 'auto'},
    ]

    best_score = 0
    best_params = {}
    results = []

    for params in param_grid:
        C, gamma = params['C'], params['gamma']
        print(f"  C={C:>5}, gamma={str(gamma):>8} ...", end=" ", flush=True)
        svm = SVC(kernel='rbf', C=C, gamma=gamma, decision_function_shape='ovr',
                  random_state=RANDOM_STATE, cache_size=1000)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(svm, X_train_pca, y_train, cv=cv, scoring='accuracy')
        mean_sc, std_sc = scores.mean(), scores.std()
        results.append((C, gamma, mean_sc, std_sc))
        print(f"CV acc = {mean_sc:.4f} +/- {std_sc:.4f}")
        if mean_sc > best_score:
            best_score = mean_sc
            best_params = {'C': C, 'gamma': gamma}

    # Fine-tune around best C
    best_C = best_params['C']
    best_gamma = best_params['gamma']
    fine_C_values = [best_C * m for m in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]]
    print(f"\n  Fine-tuning C around {best_C}...")
    for C in fine_C_values:
        print(f"    C={C:.1f} ...", end=" ", flush=True)
        svm = SVC(kernel='rbf', C=C, gamma=best_gamma,
                  decision_function_shape='ovr', random_state=RANDOM_STATE, cache_size=1000)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(svm, X_train_pca, y_train, cv=cv, scoring='accuracy')
        mean_sc, std_sc = scores.mean(), scores.std()
        results.append((C, best_gamma, mean_sc, std_sc))
        print(f"CV acc = {mean_sc:.4f} +/- {std_sc:.4f}")
        if mean_sc > best_score:
            best_score = mean_sc
            best_params = {'C': C, 'gamma': best_gamma}

    print(f"\n  Best: C={best_params['C']}, gamma={best_params['gamma']}, CV={best_score:.4f}")

    # 8. Train final model
    print(f"\n--- Phase 6: Train final SVM ---")
    final_svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'],
                    decision_function_shape='ovr', random_state=RANDOM_STATE, cache_size=1000)
    final_svm.fit(X_train_pca, y_train)
    sv_count = final_svm.n_support_.sum()
    print(f"  Support vectors: {sv_count} / {len(X_train_pca)} ({sv_count/len(X_train_pca)*100:.1f}%)")

    # 9. Evaluate
    print(f"\n--- Phase 7: Evaluate ---")
    y_val_pred = final_svm.predict(X_val_pca)
    val_acc = accuracy_score(y_val, y_val_pred)
    y_train_pred = final_svm.predict(X_train_pca)
    train_acc = accuracy_score(y_train, y_train_pred)

    print(f"  Train accuracy:      {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Gap:                 {(train_acc - val_acc)*100:.2f}pp")

    report = classification_report(y_val, y_val_pred, output_dict=True)
    per_class = np.array([report[str(c)]['recall'] for c in range(len(le.classes_))])
    worst_5 = np.argsort(per_class)[:5]
    best_5  = np.argsort(per_class)[-5:]

    # 10. Save pipeline
    print(f"\n--- Saving pipeline to {OUTPUT_DIR} ---")
    pipeline = {
        'kmeans': kmeans, 'scaler': scaler, 'pca': pca, 'svm': final_svm,
        'label_encoder': le, 'best_params': best_params,
        'spm_levels': SPM_LEVELS, 'vocab_size': VOCAB_SIZE,
        'val_accuracy': val_acc,
    }
    with open(os.path.join(OUTPUT_DIR, "pipeline_v2.pkl"), "wb") as f:
        pickle.dump(pipeline, f)

    # ══════════════════════════════════════════════════════════
    # 11. GENERATE KAGGLE SUBMISSION
    # ══════════════════════════════════════════════════════════
    print(f"\n--- Phase 8: Generate Kaggle submission ---")

    sub_df = pd.read_csv(SAMPLE_SUB)
    sub_df.columns = [c.strip() for c in sub_df.columns]
    test_ids = sub_df['ID'].values
    print(f"  Test images to predict: {len(test_ids)}")

    test_fnames = [tid + ".jpg" for tid in test_ids]
    X_test = encode_images(test_fnames, TEST_IMG_DIR, sift, kmeans,
                           desc_label="Encode (test)")
    print(f"  X_test: {X_test.shape}")

    X_test_sc  = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_sc)
    print(f"  X_test after PCA: {X_test_pca.shape}")

    y_test_pred = final_svm.predict(X_test_pca)
    test_labels = le.inverse_transform(y_test_pred)

    submission = pd.DataFrame({'ID': test_ids, 'TARGET': test_labels})
    out_path = os.path.join(BASE_DIR, "submission_v2.csv")
    submission.to_csv(out_path, index=False)
    print(f"  Submission saved: {out_path}")
    print(f"  {len(submission)} rows, {len(set(test_labels))} unique classes predicted")

    # ══════════════════════════════════════════════════════════
    # 12. Diagnostics
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("DIAGNOSTICS")
    print("=" * 65)
    print(f"  Feature dims:         {total_dim} ({spm_dim} SPM + {COLOR_DIM} color)")
    print(f"  PCA components:       {n_comp} (explains {explained*100:.2f}%)")
    print(f"  Kernel:               RBF")
    print(f"  Best params:          C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"  Best CV accuracy:     {best_score:.4f}")
    print(f"  Train accuracy:       {train_acc:.4f}")
    print(f"  Validation accuracy:  {val_acc:.4f}")
    print(f"  Train-Val gap:        {(train_acc - val_acc)*100:.2f}pp")
    print(f"  Support vectors:      {sv_count} / {len(X_train_pca)} ({sv_count/len(X_train_pca)*100:.1f}%)")
    print(f"  Per-class acc range:  [{per_class.min():.4f}, {per_class.max():.4f}]")
    print(f"  Per-class acc mean:   {per_class.mean():.4f}")
    print(f"\n  5 worst classes:")
    for i in worst_5:
        print(f"    {le.classes_[i]:35s} acc={per_class[i]:.4f}")
    print(f"  5 best classes:")
    for i in best_5:
        print(f"    {le.classes_[i]:35s} acc={per_class[i]:.4f}")
    print(f"\n  Grid search (top 10):")
    for C, gamma, mean, std in sorted(results, key=lambda x: -x[2])[:10]:
        print(f"    C={C:>5}, gamma={str(gamma):<8}  ->  {mean:.4f} +/- {std:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Submission file: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()