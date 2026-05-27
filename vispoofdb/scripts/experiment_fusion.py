"""
experiment_fusion.py

Chạy tuần tự các thử nghiệm theo kế hoạch:
 1) Baseline (đọc các mô hình đã lưu nếu có)
 2) Swap-features: train SVM/XGBoost/MLP trên từng feature set
 3) Late-fusion: soft-voting + logistic meta (oof)
 4) Early-fusion: concat + PCA + SVM/XGBoost
 5) Stacking: out-of-fold probs -> meta-classifier

Kết quả sẽ in ra và lưu vào `vispoofdb/experiments/results_summary.csv`.

Ghi chú: Script cố gắng tìm các file feature trong các đường dẫn thông dụng của repo.
"""

import os
import joblib
import numpy as np
import sys
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import StratifiedKFold

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / 'vispoofdb' / 'data'
MODELS_DIR = BASE / 'vispoofdb' / 'models_saved'
OUT_DIR = BASE / 'vispoofdb' / 'experiments'
OUT_DIR.mkdir(exist_ok=True)

# Load metadata paths if available
METADATA_PATH = BASE / 'vispoofdb' / 'data' / 'clean_data' / 'metadata.csv'
METADATA_PATHS = None
METADATA_LABELS = None
METADATA_SPLITS = None
if METADATA_PATH.exists():
    try:
        import csv
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            METADATA_PATHS = [row['file_path'] for row in reader]
            METADATA_LABELS = [0 if row['label'].lower() == 'real' else 1 for row in reader]
            METADATA_SPLITS = [row['split'] for row in reader]
        print(f"Loaded metadata with {len(METADATA_PATHS)} entries")
    except Exception:
        METADATA_PATHS = METADATA_LABELS = METADATA_SPLITS = None

# Map feature keys to likely file locations
FEATURE_PATHS = {
    'lfcc': DATA_DIR / 'features_lfcc' / 'X_lfcc.npy',
    'mfcc40': DATA_DIR / 'features_mfcc' / 'X_data.npy',
    'mfcc480': DATA_DIR / 'features_model' / 'svm' / 'X_all.npy',
    'tone': DATA_DIR / 'features_model' / 'tone' / 'X_tone.npy',
    'wav2vec': DATA_DIR / 'features_wav2vec' / 'X_wav2vec.npy',
}
LABEL_PATHS = {
    'lfcc': DATA_DIR / 'features_lfcc' / 'y_lfcc.npy',
    'mfcc40': DATA_DIR / 'features_mfcc' / 'y_label.npy',
    'mfcc480': DATA_DIR / 'features_model' / 'svm' / 'y_all.npy',
    'tone': DATA_DIR / 'features_model' / 'tone' / 'y_tone.npy',
    'wav2vec': DATA_DIR / 'features_wav2vec' / 'y_wav2vec.npy',
}
SPLIT_PATHS = {
    'lfcc': DATA_DIR / 'features_lfcc' / 'splits_lfcc.npy',
    'mfcc40': DATA_DIR / 'features_mfcc' / 'splits.npy',
    'mfcc480': DATA_DIR / 'features_model' / 'svm' / 'splits_svm.npy',
    'tone': DATA_DIR / 'features_model' / 'tone' / 'splits_tone.npy',
    'wav2vec': DATA_DIR / 'features_wav2vec' / 'splits_wav2vec.npy',
}

RESULTS = []

# Utilities

def load_feature(key):
    x_path = FEATURE_PATHS.get(key)
    y_path = LABEL_PATHS.get(key)
    s_path = SPLIT_PATHS.get(key)
    if not x_path or not x_path.exists():
        print(f"[WARN] feature {key} not found at {x_path}")
        return None
    X = np.load(x_path)
    y = np.load(y_path) if y_path and y_path.exists() else None
    splits = np.load(s_path, allow_pickle=True) if s_path and s_path.exists() else None
    # try to find a 'paths' file in same directory
    paths = None
    try:
        parent = x_path.parent
        # look for any file starting with 'paths'
        candidates = list(parent.glob('paths*.npy'))
        if candidates:
            paths = np.load(candidates[0], allow_pickle=True)
    except Exception:
        paths = None
    # fallback: if metadata available and lengths match, use metadata paths/splits/labels
    if X is not None and METADATA_PATHS is not None:
        try:
            if len(X) == len(METADATA_PATHS):
                if paths is None:
                    paths = np.array(METADATA_PATHS)
                if splits is None and METADATA_SPLITS is not None:
                    splits = np.array(METADATA_SPLITS)
                if y is None and METADATA_LABELS is not None:
                    y = np.array(METADATA_LABELS)
        except Exception:
            pass
    print(f"Loaded {key}: X={X.shape}, y={None if y is None else y.shape}, splits={'present' if splits is not None else 'missing'}, paths={'present' if paths is not None else 'missing'})")
    return X, y, splits, paths


def compute_eer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return fpr[idx]


def eval_preds(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    eer = compute_eer(y_true, y_score)
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, eer=eer)


# Split helper
def masks_from_splits(splits):
    mask_train = splits == 'train'
    mask_seen = splits == 'test_seen'
    mask_unseen = splits == 'test_unseen'
    return mask_train, mask_seen, mask_unseen


# 1) Baseline: try to load saved models and evaluate
print("\n== Baseline evaluation from saved models ==\n")
# Attempt to evaluate a few saved models
saved_model_info = [
    ('svm_lfcc', 'svm_lfcc_model.pkl', 'scaler_lfcc.pkl', 'lfcc'),
    ('svm_tone', 'svm_tone_model.pkl', 'scaler_tone.pkl', 'tone'),
    ('svm_voice', 'svm_voice_model.pkl', 'scaler_final.pkl', 'mfcc40'),
    ('xgboost_tone', 'xgboost_tone_model.pkl', None, 'tone'),
    ('best_xgboost', 'best_xgboost.pkl', None, 'mfcc480'),
    ('mlp_wav2vec', 'mlp_wav2vec_model.pkl', 'scaler_wav2vec.pkl', 'wav2vec'),
]

for name, model_file, scaler_file, feat_key in saved_model_info:
    model_path = MODELS_DIR / model_file
    scaler_path = MODELS_DIR / scaler_file if scaler_file else None
    try:
        model = joblib.load(model_path) if model_path.exists() else None
        scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None
        data = load_feature(feat_key)
        if model is None or data is None:
            print(f"Skipping {name} (missing model or data)")
            continue
        X, y, splits, paths = data
        mask_train, mask_seen, mask_unseen = masks_from_splits(splits)
        X_seen = scaler.transform(X[mask_seen]) if scaler is not None else X[mask_seen]
        X_unseen = scaler.transform(X[mask_unseen]) if scaler is not None else X[mask_unseen]
        y_seen = y[mask_seen]
        y_unseen = y[mask_unseen]
        # preds
        y_pred_seen = model.predict(X_seen)
        y_score_seen = model.predict_proba(X_seen)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_seen)
        res_seen = eval_preds(y_seen, y_pred_seen, y_score_seen)
        y_pred_un = model.predict(X_unseen)
        y_score_un = model.predict_proba(X_unseen)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_unseen)
        res_un = eval_preds(y_unseen, y_pred_un, y_score_un)
        print(f"Model {name} -- test_seen acc={res_seen['accuracy']*100:.2f}% eer={res_seen['eer']*100:.2f}% | test_unseen acc={res_un['accuracy']*100:.2f}% eer={res_un['eer']*100:.2f}%")
        RESULTS.append((name, 'baseline', res_seen, res_un))
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

# Evaluate AASIST if available
print("\n== Evaluating AASIST (Deep Learning) ==\n")
try:
    aasist_model_path = MODELS_DIR / 'aasist_best_model.pth'
    metadata_path = BASE / 'vispoofdb' / 'data' / 'clean_data' / 'metadata.csv'
    
    if aasist_model_path.exists() and metadata_path.exists():
        from pathlib import Path as P
        aasist_root = BASE / 'AASIST'
        sys.path.insert(0, str(aasist_root))
        
        from dataset import AudioDataset
        from models.baseline import Full_AASIST_Model
        from torch.utils.data import DataLoader
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aasist_model = Full_AASIST_Model().to(device)
        aasist_model.load_state_dict(torch.load(str(aasist_model_path), map_location=device))
        aasist_model.eval()
        
        # Load test_seen and test_unseen data
        for split_name in ['test_seen', 'test_unseen']:
            dataset = AudioDataset(str(metadata_path), split=split_name)
            loader = DataLoader(dataset, batch_size=16, shuffle=False)
            
            all_preds = []
            all_probs = []
            all_labels = []
            
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    outputs = aasist_model(x)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    all_probs.extend(probs)
                    all_preds.extend(preds)
                    all_labels.extend(y.numpy())
            
            all_probs = np.array(all_probs)
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            if split_name == 'test_seen':
                res_seen = eval_preds(all_labels, all_preds, all_probs)
            else:
                res_un = eval_preds(all_labels, all_preds, all_probs)
        
        print(f"Model AASIST -- test_seen acc={res_seen['accuracy']*100:.2f}% eer={res_seen['eer']*100:.2f}% | test_unseen acc={res_un['accuracy']*100:.2f}% eer={res_un['eer']*100:.2f}%")
        RESULTS.append(('AASIST', 'baseline', res_seen, res_un))
    else:
        if not aasist_model_path.exists():
            print(f"AASIST model not found: {aasist_model_path}")
        if not metadata_path.exists():
            print(f"Metadata not found: {metadata_path}")

except Exception as e:
    print(f"Error evaluating AASIST: {e}")
    import traceback
    traceback.print_exc()


# Helper: quick trainer for common models

def train_and_evaluate_quick(model_type, X, y, splits, scaler=None, pca=None, save_name=None):
    mask_train, mask_seen, mask_unseen = masks_from_splits(splits)
    X_train, X_seen, X_un = X[mask_train], X[mask_seen], X[mask_unseen]
    y_train, y_seen, y_un = y[mask_train], y[mask_seen], y[mask_unseen]
    if scaler is None:
        scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_seen_sc = scaler.transform(X_seen)
    X_un_sc = scaler.transform(X_un)
    if pca is not None:
        pca = PCA(n_components=pca)
        X_train_sc = pca.fit_transform(X_train_sc)
        X_seen_sc = pca.transform(X_seen_sc)
        X_un_sc = pca.transform(X_un_sc)
    if model_type == 'svm':
        model = SVC(kernel='rbf', probability=True, C=10.0, random_state=42)
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)
    elif model_type == 'xgb' and XGBClassifier is not None:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    else:
        raise ValueError('unknown model')
    model.fit(X_train_sc, y_train)
    y_pred_seen = model.predict(X_seen_sc)
    y_score_seen = model.predict_proba(X_seen_sc)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_seen_sc)
    res_seen = eval_preds(y_seen, y_pred_seen, y_score_seen)
    y_pred_un = model.predict(X_un_sc)
    y_score_un = model.predict_proba(X_un_sc)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_un_sc)
    res_un = eval_preds(y_un, y_pred_un, y_score_un)
    if save_name:
        joblib.dump(model, OUT_DIR / f"{save_name}.pkl")
        joblib.dump(scaler, OUT_DIR / f"{save_name}_scaler.pkl")
    return model, scaler, res_seen, res_un


# 2) Swap-features: train SVM on each feature quickly
print("\n== Swap-features: training SVM on available features (quick) ==\n")
for feat in ['lfcc', 'tone', 'mfcc40', 'wav2vec']:
    data = load_feature(feat)
    if data is None:
        continue
    X, y, splits, paths = data
    try:
        model, scaler, res_seen, res_un = train_and_evaluate_quick('svm', X, y, splits, save_name=f"svm_on_{feat}")
        print(f"SVM on {feat}: test_unseen EER={res_un['eer']*100:.2f}% acc={res_un['accuracy']*100:.2f}%")
        RESULTS.append((f"svm_on_{feat}", 'swap', res_seen, res_un))
    except Exception as e:
        print(f"Error training SVM on {feat}: {e}")

# 3) Late-fusion (soft-voting) using saved models probabilities (average)
print("\n== Late-fusion (soft-voting average) ==\n")
# Attempt to load probabilities from models we evaluated earlier; reuse those predictions
# For simplicity, use the saved models we could load above and average their probs per test set
available_probs_seen = []
available_probs_un = []
labels_seen = None
labels_un = None
for name, model_file, scaler_file, feat_key in saved_model_info:
    model_path = MODELS_DIR / model_file
    scaler_path = MODELS_DIR / scaler_file if scaler_file else None
    if not model_path.exists():
        continue
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None
        data = load_feature(feat_key)
        if data is None:
            continue
        X, y, splits, paths = data
        mask_train, mask_seen, mask_unseen = masks_from_splits(splits)
        X_seen = scaler.transform(X[mask_seen]) if scaler is not None else X[mask_seen]
        X_un = scaler.transform(X[mask_unseen]) if scaler is not None else X[mask_unseen]
        y_seen = y[mask_seen]
        y_un = y[mask_unseen]
        p_seen = model.predict_proba(X_seen)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_seen)
        p_un = model.predict_proba(X_un)[:,1] if hasattr(model, 'predict_proba') else model.decision_function(X_un)
        available_probs_seen.append(p_seen)
        available_probs_un.append(p_un)
        labels_seen = y_seen
        labels_un = y_un
        print(f"Included {name} in fusion (feat={feat_key})")
    except Exception as e:
        print(f"Skipping {name} for fusion: {e}")

if available_probs_seen:
    avg_p_seen = np.mean(np.vstack(available_probs_seen), axis=0)
    avg_p_un = np.mean(np.vstack(available_probs_un), axis=0)
    pred_seen = (avg_p_seen >= 0.5).astype(int)
    pred_un = (avg_p_un >= 0.5).astype(int)
    res_seen = eval_preds(labels_seen, pred_seen, avg_p_seen)
    res_un = eval_preds(labels_un, pred_un, avg_p_un)
    print(f"Late-fusion AVG -- test_unseen EER={res_un['eer']*100:.2f}% acc={res_un['accuracy']*100:.2f}%")
    RESULTS.append(("late_fusion_avg", 'fusion', res_seen, res_un))

# 4) Early-fusion: concat MFCC40 + Tone + LFCC (if available)
print("\n== Early-fusion: concat MFCC40 + Tone (+LFCC if present) -> PCA(95%) -> SVM ==\n")
parts = []
labels = None
splits = None
for key in ['mfcc40', 'tone', 'lfcc']:
    d = load_feature(key)
    if d is None:
        continue
    Xk, yk, sk, pk = d
    # If feature is frame-level (3D: N x D x T), pool to fixed-size vector
    if Xk is not None and Xk.ndim == 3:
        # mean + std over time axis -> shape (N, 2*D)
        mean_feat = np.mean(Xk, axis=2)
        std_feat = np.std(Xk, axis=2)
        Xk = np.concatenate([mean_feat, std_feat], axis=1)
        print(f"Pooled 3D feature {key} -> {Xk.shape}")
    parts.append((key, Xk, yk, sk, pk))
    # only set labels/splits if present (avoid overwriting with missing ones)
    if labels is None and yk is not None:
        labels = yk
    if splits is None and sk is not None:
        splits = sk

if len(parts) >= 1:
    # align by index assumptions (all feature files have same ordering)
    # Align by paths if available across parts
    # collect path lists
    paths_list = [p for (_, _, _, _, p) in parts if p is not None]
    if paths_list:
        # find common paths
        common = set(paths_list[0])
        for p in paths_list[1:]:
            common = common.intersection(set(p))
        common = sorted(common)
        if not common:
            print("No common samples found across features for early-fusion")
            X_concat = None
        else:
            aligned_arrays = []
            for (k, Xk, yk, sk, pk) in parts:
                if pk is None:
                    # cannot align this part, skip
                    print(f"Skipping {k} in early-fusion because no paths available to align")
                    X_concat = None
                    break
                idx_map = {path: i for i, path in enumerate(pk)}
                idxs = [idx_map[p] for p in common if p in idx_map]
                aligned_arrays.append(Xk[idxs])
            else:
                try:
                    X_concat = np.hstack(aligned_arrays)
                    # set labels/splits from one of the aligned parts if possible
                    labels = None
                    splits = None
                    for (k, Xk, yk, sk, pk) in parts:
                        if pk is not None and yk is not None and sk is not None:
                            idx_map = {path: i for i, path in enumerate(pk)}
                            idxs = [idx_map[p] for p in common if p in idx_map]
                            labels = yk[idxs]
                            splits = sk[idxs]
                            break
                except Exception as e:
                    print(f"Error concatenating aligned features: {e}")
                    X_concat = None
    else:
        try:
            X_concat = np.hstack([Xk for (_, Xk, _, _, _) in parts])
        except Exception as e:
            print(f"Error concatenating features: {e}")
            X_concat = None
    if X_concat is not None:
        try:
            model, scaler, res_seen, res_un = train_and_evaluate_quick('svm', X_concat, labels, splits, pca=0.95, save_name='svm_early_fusion')
            print(f"Early-fusion SVM -- test_unseen EER={res_un['eer']*100:.2f}% acc={res_un['accuracy']*100:.2f}%")
            RESULTS.append(("svm_early_fusion", 'early_fusion', res_seen, res_un))
        except Exception as e:
            print(f"Early-fusion failed: {e}")

# 5) Stacking: train base SVMs on different features with OOF then meta logistic
print("\n== Stacking (OOF probs -> Logistic meta) ==\n")
base_features = ['lfcc', 'tone', 'mfcc40']
base_oof_train = None
base_oof_seen = []
base_oof_un = []
base_labels_train = None
base_splits = None
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Load all base features and align by paths if possible
base_data = {}
for key in base_features:
    d = load_feature(key)
    if d is None:
        continue
    Xk, yk, sk, pk = d
    # if frame-level, pool
    if Xk is not None and Xk.ndim == 3:
        mean_feat = np.mean(Xk, axis=2)
        std_feat = np.std(Xk, axis=2)
        Xk = np.concatenate([mean_feat, std_feat], axis=1)
    base_data[key] = (Xk, yk, sk, pk)

# Check if all have paths
all_have_paths = all([v[3] is not None for v in base_data.values()]) if base_data else False
if not base_data:
    print("No base features available for stacking")
else:
    if not all_have_paths:
        print("Skipping stacking: not all base features provide 'paths' for reliable alignment")
    else:
        # compute common paths
        path_lists = [list(v[3]) for v in base_data.values()]
        common = set(path_lists[0])
        for p in path_lists[1:]:
            common = common.intersection(set(p))
        common = sorted(common)
        if not common:
            print("No common samples for stacking")
        else:
            # align datasets
            aligned = {}
            for k, (Xk, yk, sk, pk) in base_data.items():
                idx_map = {path: i for i, path in enumerate(pk)}
                idxs = [idx_map[p] for p in common if p in idx_map]
                aligned[k] = (Xk[idxs], yk[idxs], sk[idxs])
            # proceed OOF per aligned feature
            for key, (Xk, yk, sk) in aligned.items():
                mask_train, mask_seen, mask_unseen = masks_from_splits(sk)
                X_train_k = Xk[mask_train]
                y_train_k = yk[mask_train]
                X_seen_k = Xk[mask_seen]
                X_un_k = Xk[mask_unseen]
                oof = np.zeros(len(X_train_k))
                for train_idx, val_idx in skf.split(X_train_k, y_train_k):
                    Xtr = X_train_k[train_idx]
                    ytr = y_train_k[train_idx]
                    Xval = X_train_k[val_idx]
                    model = SVC(kernel='rbf', probability=True, C=10.0)
                    scaler = StandardScaler()
                    Xtr_sc = scaler.fit_transform(Xtr)
                    Xval_sc = scaler.transform(Xval)
                    model.fit(Xtr_sc, ytr)
                    oof[val_idx] = model.predict_proba(Xval_sc)[:,1]
                scaler_full = StandardScaler()
                Xtrain_full_sc = scaler_full.fit_transform(X_train_k)
                model_full = SVC(kernel='rbf', probability=True, C=10.0)
                model_full.fit(Xtrain_full_sc, y_train_k)
                X_seen_sc = scaler_full.transform(X_seen_k)
                X_un_sc = scaler_full.transform(X_un_k)
                p_seen = model_full.predict_proba(X_seen_sc)[:,1]
                p_un = model_full.predict_proba(X_un_sc)[:,1]
                base_oof_train = np.column_stack([base_oof_train, oof]) if base_oof_train is not None else oof.reshape(-1,1)
                base_oof_seen.append(p_seen)
                base_oof_un.append(p_un)
                base_labels_train = y_train_k
                base_splits = sk
                print(f"Built OOF for base feature {key}")

if base_oof_train is not None:
    # meta train
    meta_clf = LogisticRegression(max_iter=200)
    meta_clf.fit(base_oof_train, base_labels_train)
    # meta preds on seen/un
    meta_seen_p = meta_clf.predict_proba(np.vstack(base_oof_seen).T)[:,1]
    meta_un_p = meta_clf.predict_proba(np.vstack(base_oof_un).T)[:,1]
    meta_seen_pred = (meta_seen_p >= 0.5).astype(int)
    meta_un_pred = (meta_un_p >= 0.5).astype(int)
    # labels for seen/un (from aligned datasets used for stacking)
    try:
        first_key = list(aligned.keys())[0]
        _, y_ref, splits_ref = aligned[first_key]
        mask_train_r, mask_seen_r, mask_unseen_r = masks_from_splits(splits_ref)
        y_seen = y_ref[mask_seen_r]
        y_un = y_ref[mask_unseen_r]
        res_seen = eval_preds(y_seen, meta_seen_pred, meta_seen_p)
        res_un = eval_preds(y_un, meta_un_pred, meta_un_p)
        print(f"Stacking meta -- test_unseen EER={res_un['eer']*100:.2f}% acc={res_un['accuracy']*100:.2f}%")
        RESULTS.append(("stacking_logreg", 'stacking', res_seen, res_un))
    except Exception as e:
        print(f"Could not evaluate stacking meta: {e}")

# Save summary
import csv
out_csv = OUT_DIR / 'results_summary.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['experiment','type','set','accuracy','precision','recall','f1','eer'])
    for name, typ, seen, un in RESULTS:
        writer.writerow([name, typ, 'test_seen', f"{seen['accuracy']:.4f}", f"{seen['precision']:.4f}", f"{seen['recall']:.4f}", f"{seen['f1']:.4f}", f"{seen['eer']:.4f}"])
        writer.writerow([name, typ, 'test_unseen', f"{un['accuracy']:.4f}", f"{un['precision']:.4f}", f"{un['recall']:.4f}", f"{un['f1']:.4f}", f"{un['eer']:.4f}"])

print(f"\nSaved results to {out_csv}")
print("Done.")
