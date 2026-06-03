"""
plot_results.py

Generate visualizations (ROC, DET-like FNR vs FPR, confusion matrices)
for all 9 models: 8 sklearn models + 1 AASIST deep learning model.
Saves PNGs in vispoofdb/figures/.
"""
from pathlib import Path
import numpy as np
import joblib
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

BASE = Path(__file__).resolve().parents[2]
DATA_DIR = BASE / 'vispoofdb' / 'data'
MODELS_DIR = BASE / 'vispoofdb' / 'models_saved'
EXPERIMENT_DIR = BASE / 'vispoofdb' / 'experiments'
FIG_DIR = BASE / 'vispoofdb' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)
AASIST_DIR = BASE / 'vispoofdb' / 'models' / 'aasist'

# Fix encoding for Windows terminal
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Feature paths
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


def load_feature(key):
    """Load feature, labels, and splits for a given key."""
    x_path = FEATURE_PATHS.get(key)
    y_path = LABEL_PATHS.get(key)
    s_path = SPLIT_PATHS.get(key)
    if not x_path or not x_path.exists():
        return None, None, None
    X = np.load(x_path)
    y = np.load(y_path) if y_path and y_path.exists() else None
    splits = np.load(s_path, allow_pickle=True) if s_path and s_path.exists() else None
    # pool if 3D (mean + std over time axis)
    if X is not None and X.ndim == 3:
        mean_feat = np.mean(X, axis=2)
        std_feat = np.std(X, axis=2)
        X = np.concatenate([mean_feat, std_feat], axis=1)
    return X, y, splits

def evaluate_sklearn_model(name, model_path, scaler_path, feat_key):
    """Evaluate sklearn model on test_unseen."""
    if not model_path.exists():
        print(f"[SKIP] {name}: model not found")
        return None
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None
        X, y, splits = load_feature(feat_key)
        
        if X is None or y is None or splits is None:
            print(f"[SKIP] {name}: missing features")
            return None
        
        mask_unseen = splits == 'test_unseen'
        X_un = X[mask_unseen]
        y_un = y[mask_unseen]
        
        if scaler is not None:
            X_un = scaler.transform(X_un)
        
        # Get scores
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_un)[:, 1]
        else:
            try:
                scores = model.decision_function(X_un)
            except:
                print(f"[SKIP] {name}: cannot produce scores")
                return None
        
        preds = (scores >= 0.5).astype(int)
        acc = accuracy_score(y_un, preds)
        
        return {
            'name': name,
            'y_true': y_un,
            'y_score': scores,
            'y_pred': preds,
            'accuracy': acc,
        }
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return None

def evaluate_aasist_model():
    """Evaluate AASIST model on test_unseen."""
    try:
        sys.path.insert(0, str(AASIST_DIR))
        from dataset import AudioDataset
        from models.baseline import Full_AASIST_Model
        
        metadata_path = BASE / 'vispoofdb' / 'data' / 'clean_data' / 'metadata.csv'
        if not metadata_path.exists():
            print("[SKIP] AASIST: metadata not found")
            return None
        
        model_path = MODELS_DIR / 'aasist_best_model.pth'
        if not model_path.exists():
            print("[SKIP] AASIST: model not found")
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = Full_AASIST_Model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Load test_unseen data
        dataset = AudioDataset(str(metadata_path), split='test_unseen')
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                outputs = model(x)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        all_probs = np.array(all_probs)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        acc = accuracy_score(all_labels, all_preds)
        
        return {
            'name': 'AASIST',
            'y_true': all_labels,
            'y_score': all_probs,
            'y_pred': all_preds,
            'accuracy': acc,
        }
    except Exception as e:
        print(f"[ERROR] AASIST: {e}")
        return None

# All 9 models to evaluate
models_to_plot = [
    # 8 Sklearn models
    ('svm_lfcc', MODELS_DIR / 'svm_lfcc_model.pkl', MODELS_DIR / 'scaler_lfcc.pkl', 'lfcc', 'sklearn'),
    ('svm_mfcc', MODELS_DIR / 'svm_voice_model.pkl', MODELS_DIR / 'scaler_final.pkl', 'mfcc40', 'sklearn'),
    ('mlp_mfcc', MODELS_DIR / 'best_mlp.pkl', MODELS_DIR / 'scaler_mlp.pkl', 'mfcc480', 'sklearn'),
    ('xgboost_mfcc', MODELS_DIR / 'best_xgboost.pkl', None, 'mfcc480', 'sklearn'),
    ('mlp_wav2vec', MODELS_DIR / 'mlp_wav2vec_model.pkl', MODELS_DIR / 'scaler_wav2vec.pkl', 'wav2vec', 'sklearn'),
    ('svm_tone', MODELS_DIR / 'svm_tone_model.pkl', MODELS_DIR / 'scaler_tone.pkl', 'tone', 'sklearn'),
    ('xgboost_tone', MODELS_DIR / 'xgboost_tone_model.pkl', None, 'tone', 'sklearn'),
    ('svm_fusion', MODELS_DIR / 'svm_tone_fusion_model.pkl', MODELS_DIR / 'scaler_fusion.pkl', 'tone', 'sklearn'),
    # 1 Deep learning model
    ('aasist', None, None, None, 'aasist'),
]

print("\n" + "="*70)
print("  EVALUATING ALL 9 MODELS ON TEST_UNSEEN")
print("="*70 + "\n")

results = []

# Evaluate all models
for item in models_to_plot:
    if len(item) == 5:
        name, model_path, scaler_path, feat_key, model_type = item
        if model_type == 'sklearn':
            result = evaluate_sklearn_model(name, model_path, scaler_path, feat_key)
            if result:
                results.append(result)
        elif model_type == 'aasist':
            result = evaluate_aasist_model()
            if result:
                results.append(result)

if not results:
    print("\n[ERROR] No models could be evaluated.")
    sys.exit(1)

print(f"\n✓ Successfully evaluated {len(results)} models\n")

# Generate visualizations
roc_data = []

print("="*70)
print("  GENERATING PLOTS")
print("="*70 + "\n")

for result in results:
    name = result['name']
    y_true = result['y_true']
    y_score = result['y_score']
    y_pred = result['y_pred']
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    roc_data.append((name, fpr, tpr, roc_auc))
    
    # Individual ROC plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve: {name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / f"roc_{name}_unseen.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'AI/Fake'])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues')
    fig.suptitle(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
    fig.savefig(FIG_DIR / f"cm_{name}_unseen.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # DET curve
    fnr = 1 - tpr
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, fnr, linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('False Negative Rate', fontsize=11)
    ax.set_title(f'DET Curve: {name}', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / f"det_{name}_unseen.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ {name:20} | AUC={roc_auc:.4f} | Acc={result['accuracy']*100:6.2f}%")

# Combined ROC plot
if roc_data:
    fig, ax = plt.subplots(figsize=(8, 8))
    for name, fpr, tpr, roc_auc in sorted(roc_data, key=lambda x: x[3], reverse=True):
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Comparison: All Models', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / 'roc_comparison_all_unseen.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Saved combined ROC plot to {FIG_DIR / 'roc_comparison_all_unseen.png'}")

print(f"\n✓ All plots saved to: {FIG_DIR}\n")
print("="*70)
