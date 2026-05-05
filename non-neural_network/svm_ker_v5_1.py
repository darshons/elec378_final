"""
378 final proj non-nn model 5
spm bovw, color thru hsv, chi-squared kernel svm

0.788 on kaggle B)

apparently chi-squared is better on raw histogram data, so no more pca and standardscaler

overview:
1. sift features from training images
2. k-means for visual feature vocab
3. spatial pyramid/bag of visual words
4. analyze color with hsv since sift is grayscale
5. chi-squared kernel svm
6. encode test images and apply prediction
7. csv generation
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import cv2
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import chi2_kernel
from tqdm import tqdm

# paths
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "butterfly_data")
TRAIN_CSV     = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images", "train_images")
TEST_IMG_DIR  = os.path.join(DATA_DIR, "test_images", "test_images")
SAMPLE_SUB    = os.path.join(BASE_DIR, "sample_submission.csv")

OUTPUT_DIR    = os.path.join(BASE_DIR, "v5_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# feature extraction parameters
IMG_SIZE       = 256 # to standardize, aspect ratios should stay same
SIFT_MAX_KP    = 500 # keeping 500 most important sift keypoints
VOCAB_SIZE     = 500 # 500 clusters for k means
SAMPLE_PER_IMG = 50 # 50 descriptors per image * ~10000 images = 500000 for k means
RANDOM_STATE   = 42

SPM_LEVELS = [1, 2, 4] # spm will have 3 levels: 1x1, 2x2, 4x4. each box has own bovw.
N_REGIONS  = sum(l * l for l in SPM_LEVELS)  # 1 + 4 + 16 = 21

H_BINS, S_BINS, V_BINS = 32, 32, 32
COLOR_DIM = H_BINS + S_BINS + V_BINS  # 96 dim from hsv color analysis



# load/resize imgs
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

# sift for 500 keypoints
def extract_sift(gray, sift, max_kp=SIFT_MAX_KP):
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        return [], np.empty((0, 128), dtype=np.float32)
    if len(kp) > max_kp:
        indices = np.argsort([-k.response for k in kp])[:max_kp]
        kp = [kp[i] for i in indices]
        des = des[indices]
    return kp, des.astype(np.float32)

# spm, k means, bovw
def spatial_pyramid_histogram(keypoints, descriptors, kmeans,
                               vocab_size, img_h, img_w, levels):
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

# hsv colors
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

# encode image list into feature vectors.  
def encode_images(file_list, img_dir, sift, kmeans, kp_data_list=None):
    """Encode a list of images into feature vectors.
    If kp_data_list is provided, reuses pre-extracted SIFT data (for training set).
    Otherwise extracts SIFT on the fly (for val/test sets).
    """
    total_dim = VOCAB_SIZE * N_REGIONS + COLOR_DIM
    features = []
    for i, fname in enumerate(tqdm(file_list, desc=f"Encode")):
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


# main
def main():
    t0 = time.time()
    print("=" * 65)
    print("PIPELINE v5: SPM BoVW + Color + Chi-Squared Kernel SVM")
    print("=" * 65)

    # load metadata + split
    df = pd.read_csv(TRAIN_CSV)
    df.columns = [c.strip() for c in df.columns]
    print(f"\nLoaded {len(df)} entries, {df['TARGET'].nunique()} classes")

    train_files, val_files, train_labels, val_labels = train_test_split(
        df["file_name"].values, df["TARGET"].values,
        test_size=0.2, stratify=df["TARGET"].values, random_state=RANDOM_STATE,
    )
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # sift extraction on train data 
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

    # k-means vocab
    print(f"\n--- Phase 2: K-Means vocabulary (K={VOCAB_SIZE}) ---")
    kmeans = MiniBatchKMeans(
        n_clusters=VOCAB_SIZE, batch_size=4096, random_state=RANDOM_STATE,
        n_init=3, max_iter=300,
    )
    kmeans.fit(sampled)
    print(f"  Inertia: {kmeans.inertia_:.0f}")

    # feature encoding on raw histograms
    spm_dim = VOCAB_SIZE * N_REGIONS
    total_dim = spm_dim + COLOR_DIM
    print(f"\n--- Phase 3: Feature encoding ({spm_dim} SPM + {COLOR_DIM} color = {total_dim}) ---")

    X_train = encode_images(train_files, TRAIN_IMG_DIR, sift, kmeans, train_kp_data)
    X_val   = encode_images(val_files, TRAIN_IMG_DIR, sift, kmeans)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

    # label encoding
    le = LabelEncoder()
    le.fit(np.concatenate([train_labels, val_labels]))
    y_train = le.transform(train_labels)
    y_val   = le.transform(val_labels)

    # chi-squared kernel svm search on raw features
    gamma_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    C_values     = [1, 10, 50, 100]

    print(f"\n--- Phase 4: Chi-squared kernel SVM search ---")
    best_score = 0
    best_params = {}
    results = []

    for gamma in gamma_values:
        print(f"\n  gamma={gamma}: computing kernel...", end=" ", flush=True)
        t_k = time.time()
        K_train = chi2_kernel(X_train, X_train, gamma=gamma)
        print(f"done ({time.time()-t_k:.1f}s)")

        for C in C_values:
            print(f"    C={C:>5}, gamma={gamma}  ->  ", end="", flush=True)
            svm = SVC(kernel='precomputed', C=C, decision_function_shape='ovr',
                      random_state=RANDOM_STATE, cache_size=1000)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scores = cross_val_score(svm, K_train, y_train, cv=cv, scoring='accuracy')
            mean_sc, std_sc = scores.mean(), scores.std()
            results.append((C, gamma, mean_sc, std_sc))
            print(f"CV acc = {mean_sc:.4f} +/- {std_sc:.4f}")
            if mean_sc > best_score:
                best_score = mean_sc
                best_params = {'C': C, 'gamma': gamma}

    print(f"\n  Best: C={best_params['C']}, gamma={best_params['gamma']}, CV={best_score:.4f}")

    # train final model
    print(f"\n--- Phase 5: Train final SVM ---")
    K_train_final = chi2_kernel(X_train, X_train, gamma=best_params['gamma'])
    final_svm = SVC(kernel='precomputed', C=best_params['C'],
                    decision_function_shape='ovr', random_state=RANDOM_STATE,
                    cache_size=1000)
    final_svm.fit(K_train_final, y_train)
    sv_count = final_svm.n_support_.sum()
    print(f"  Support vectors: {sv_count} / {len(X_train)} ({sv_count/len(X_train)*100:.1f}%)")

    # evaluate on validation set
    print(f"\n--- Phase 6: Evaluate ---")
    K_val = chi2_kernel(X_val, X_train, gamma=best_params['gamma'])
    y_val_pred = final_svm.predict(K_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    y_train_pred = final_svm.predict(K_train_final)
    train_acc = accuracy_score(y_train, y_train_pred)

    print(f"  Train accuracy:      {train_acc:.4f}")
    print(f"  Validation accuracy: {val_acc:.4f}")
    print(f"  Gap:                 {(train_acc - val_acc)*100:.2f}pp")

    report = classification_report(y_val, y_val_pred, output_dict=True)
    per_class = np.array([report[str(c)]['recall'] for c in range(len(le.classes_))])
    worst_5 = np.argsort(per_class)[:5]
    best_5  = np.argsort(per_class)[-5:]

    # save pipeline
    print(f"\n--- Saving pipeline to {OUTPUT_DIR} ---")
    pipeline = {
        'kmeans': kmeans, 'svm': final_svm, 'label_encoder': le,
        'best_params': best_params, 'spm_levels': SPM_LEVELS,
        'vocab_size': VOCAB_SIZE, 'val_accuracy': val_acc,
    }
    with open(os.path.join(OUTPUT_DIR, "pipeline_v5.pkl"), "wb") as f:
        pickle.dump(pipeline, f)
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)

    # csv
    print(f"\n--- Phase 7: Generate CSV ---")

    sub_df = pd.read_csv(SAMPLE_SUB)
    sub_df.columns = [c.strip() for c in sub_df.columns]
    test_ids = sub_df['ID'].values
    print(f"  Test images to predict: {len(test_ids)}")

    # encode test images using same pipeline
    test_fnames = [tid + ".jpg" for tid in test_ids]
    X_test = encode_images(test_fnames, TEST_IMG_DIR, sift, kmeans)
    print(f"  X_test: {X_test.shape}")

    # compute chi-squared kernel between test and training features
    print(f"  Computing chi² kernel (test × train)...", end=" ", flush=True)
    t_k = time.time()
    K_test = chi2_kernel(X_test, X_train, gamma=best_params['gamma'])
    print(f"done ({time.time()-t_k:.1f}s)")

    # predict and decode labels
    y_test_pred = final_svm.predict(K_test)
    test_labels = le.inverse_transform(y_test_pred)

    # write submission CSV
    submission = pd.DataFrame({'ID': test_ids, 'TARGET': test_labels})
    out_path = os.path.join(BASE_DIR, "submission_v5.csv")
    submission.to_csv(out_path, index=False)
    print(f"  Submission saved: {out_path}")
    print(f"  {len(submission)} rows, {len(set(test_labels))} unique classes predicted")

    # diagnostics
    
    print("\n" + "=" * 65)
    print("DIAGNOSTICS")
    print("=" * 65)
    print(f"  Feature dims:         {total_dim} ({spm_dim} SPM + {COLOR_DIM} color)")
    print(f"  Kernel:               Chi-squared (precomputed)")
    print(f"  Best params:          C={best_params['C']}, gamma={best_params['gamma']}")
    print(f"  Best CV accuracy:     {best_score:.4f}")
    print(f"  Train accuracy:       {train_acc:.4f}")
    print(f"  Validation accuracy:  {val_acc:.4f}")
    print(f"  Train-Val gap:        {(train_acc - val_acc)*100:.2f}pp")
    print(f"  Support vectors:      {sv_count} / {len(X_train)} ({sv_count/len(X_train)*100:.1f}%)")
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