"""
plot_results.py

Generate visualizations (ROC, DET-like FNR vs FPR, confusion matrices)
for available saved models and fusion experiments. Saves PNGs in vispoofdb/figures/.
"""
from pathlib import Path
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / 'vispoofdb' / 'data'
MODELS_DIR = BASE / 'vispoofdb' / 'models_saved'
EXPERIMENT_DIR = BASE / 'vispoofdb' / 'experiments'
FIG_DIR = BASE / 'vispoofdb' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Similar loader to experiment_fusion
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

# Simple loader
def load_feature(key):
    x_path = FEATURE_PATHS.get(key)
    y_path = LABEL_PATHS.get(key)
    s_path = SPLIT_PATHS.get(key)
    if not x_path or not x_path.exists():
        return None, None, None
    X = np.load(x_path)
    y = np.load(y_path) if y_path and y_path.exists() else None
    splits = np.load(s_path, allow_pickle=True) if s_path and s_path.exists() else None
    # pool if 3D
    if X is not None and X.ndim == 3:
        mean_feat = np.mean(X, axis=2)
        std_feat = np.std(X, axis=2)
        X = np.concatenate([mean_feat, std_feat], axis=1)
    return X, y, splits

models_to_plot = [
    ('svm_lfcc', MODELS_DIR / 'svm_lfcc_model.pkl', MODELS_DIR / 'scaler_lfcc.pkl', 'lfcc'),
    ('svm_tone', MODELS_DIR / 'svm_tone_model.pkl', MODELS_DIR / 'scaler_tone.pkl', 'tone'),
    ('xgboost_tone', MODELS_DIR / 'xgboost_tone_model.pkl', None, 'tone'),
    ('svm_wav2vec', EXPERIMENT_DIR / 'svm_on_wav2vec.pkl', EXPERIMENT_DIR / 'svm_on_wav2vec_scaler.pkl', 'wav2vec'),
]

roc_data = []
# Evaluate each model
for name, model_path, scaler_path, feat_key in models_to_plot:
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        continue
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None
    X, y, splits = load_feature(feat_key)
    if X is None or y is None or splits is None:
        print(f"Missing data for {feat_key}, skipping {name}")
        continue
    mask_train = splits == 'train'
    mask_unseen = splits == 'test_unseen'
    X_un = X[mask_unseen]
    y_un = y[mask_unseen]
    X_un_sc = scaler.transform(X_un) if scaler is not None else X_un
    # get scores
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X_un_sc)[:,1]
    else:
        try:
            scores = model.decision_function(X_un_sc)
        except Exception:
            print(f"Model {name} cannot produce scores")
            continue
    fpr, tpr, _ = roc_curve(y_un, scores)
    roc_auc = auc(fpr, tpr)
    roc_data.append((name, fpr, tpr, roc_auc))
    # save confusion matrix at threshold 0.5
    preds = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_un, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['real','fake'])
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax)
    fig.suptitle(f"Confusion matrix {name} (test_unseen)")
    fig.savefig(FIG_DIR / f"cm_{name}_unseen.png", bbox_inches='tight')
    plt.close(fig)
    # save ROC curve per model
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],'k--',linewidth=0.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'ROC {name} (test_unseen)')
    ax.legend()
    fig.savefig(FIG_DIR / f"roc_{name}_unseen.png", bbox_inches='tight')
    plt.close(fig)
    # DET-like: plot FNR vs FPR
    fnr = 1 - tpr
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(fpr, fnr, label=name)
    ax.set_xlabel('FPR')
    ax.set_ylabel('FNR')
    ax.set_title(f'DET-like (FNR vs FPR) {name} (test_unseen)')
    ax.legend()
    fig.savefig(FIG_DIR / f"det_{name}_unseen.png", bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plots for {name}")

# Combined ROC plot
if roc_data:
    fig, ax = plt.subplots(figsize=(6,6))
    for name, fpr, tpr, roc_auc in roc_data:
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0,1],[0,1],'k--',linewidth=0.5)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC comparison (test_unseen)')
    ax.legend()
    fig.savefig(FIG_DIR / 'roc_comparison_unseen.png', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved combined ROC to {FIG_DIR / 'roc_comparison_unseen.png'}")

print('Done plotting.')
