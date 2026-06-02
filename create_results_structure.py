#!/usr/bin/env python3
"""
Results Tracker for Model Retraining with New Data
Saves training results, metrics, and model performance
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "vispoofdb" / "experiments" / "training_runs"
MODELS_DIR = BASE_DIR / "vispoofdb" / "models_saved"

# Create directory structure
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Create timestamped result folder for this run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"run_{TIMESTAMP}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "="*90)
print("📊 RESULTS TRACKING SYSTEM INITIALIZED")
print("="*90)

print(f"\n📁 Directory structure created:")
print(f"   Results root: {RESULTS_DIR}")
print(f"   Current run:  {RUN_DIR}")
print(f"   Timestamp:    {TIMESTAMP}")

# ════════════════════════════════════════════════════════════════════════════════════════
# Create results template files
# ════════════════════════════════════════════════════════════════════════════════════════

# 1. Training metadata JSON
metadata = {
    "run_id": TIMESTAMP,
    "date": datetime.now().isoformat(),
    "data_source": "HuggingFace YouTube diverse speakers",
    "data_location": "vispoofdb/data/raw/real (NEW)",
    "previous_data": "ViSpoofDB 6 speakers",
    "notes": "Retraining with diverse speaker set to fix generalization gap",
    "expected_improvement": "75-85% on new speakers (vs 20-30% before)",
    "models_to_train": [
        "SVM + LFCC",
        "SVM + MFCC",
        "MLP + MFCC",
        "XGBoost + MFCC",
        "SVM + Tone-Aware",
        "XGBoost + Tone-Aware",
        "SVM + MFCC+Tone Fusion",
        "MLP + Wav2Vec2",
        "AASIST (Deep Learning)"
    ]
}

metadata_path = RUN_DIR / "metadata.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"\n✅ metadata.json created")

# 2. Model results CSV template
results_columns = [
    'model_name',
    'feature_type',
    'framework',
    'test_seen_accuracy',
    'test_unseen_accuracy',
    'test_seen_eer',
    'test_unseen_eer',
    'training_time_seconds',
    'inference_time_ms',
    'model_size_mb',
    'notes'
]

results_df = pd.DataFrame(columns=results_columns)
results_csv = RUN_DIR / "model_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"✅ model_results.csv created (empty template)")

# 3. Training log file
log_file = RUN_DIR / "training.log"
with open(log_file, 'w', encoding='utf-8') as f:
    f.write(f"{'='*90}\n")
    f.write(f"TRAINING LOG - Run {TIMESTAMP}\n")
    f.write(f"{'='*90}\n\n")
    f.write(f"Start time: {datetime.now().isoformat()}\n")
    f.write(f"Data source: HuggingFace YouTube diverse speakers\n")
    f.write(f"Data location: vispoofdb/data/raw/real\n")
    f.write(f"\n" + "="*90 + "\n")
    f.write(f"TRAINING PROGRESS\n")
    f.write(f"="*90 + "\n\n")

print(f"✅ training.log created")

# 4. Comparison template CSV (before vs after)
comparison_data = {
    'Model': [
        'SVM + LFCC',
        'SVM + MFCC',
        'MLP + MFCC',
        'XGBoost + MFCC',
        'MLP + Wav2Vec2',
        'SVM + Tone-Aware',
        'XGBoost + Tone-Aware',
        'SVM + MFCC+Tone Fusion',
        'AASIST'
    ],
    'Old_TestUnseen_%': [99.62, 98.81, 99.12, 96.46, 69.85, 88.96, 83.96, 98.81, 99.96],
    'Old_EER_%': [0.29, 0.14, 0.00, 3.14, 22.50, 10.79, 14.86, 0.43, 'N/A'],
    'New_TestUnseen_%': [None]*9,
    'New_EER_%': [None]*9,
    'Improvement_%': [None]*9,
    'Notes': [''] * 9
}

comparison_df = pd.DataFrame(comparison_data)
comparison_csv = RUN_DIR / "before_after_comparison.csv"
comparison_df.to_csv(comparison_csv, index=False)
print(f"✅ before_after_comparison.csv created")

# 5. Data analysis template
data_info = {
    'Metric': [
        'Total samples',
        'Real samples',
        'Fake samples',
        'Unique speakers',
        'Train samples',
        'Test_Seen samples',
        'Test_Unseen samples',
        'Data size (GB)'
    ],
    'Old_Data': [14000, 8996, 5004, 6, 8996, 2599, 2600, '~1.4'],
    'New_Data': [None]*8
}

data_info_df = pd.DataFrame(data_info)
data_info_csv = RUN_DIR / "data_comparison.csv"
data_info_df.to_csv(data_info_csv, index=False)
print(f"✅ data_comparison.csv created")

# 6. Python helper functions for logging results
helper_script = '''#!/usr/bin/env python3
"""
Helper functions to log training results
Import this in your training scripts
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

class ResultsLogger:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_csv = self.results_dir / "model_results.csv"
        self.log_file = self.results_dir / "training.log"
        self.comparison_csv = self.results_dir / "before_after_comparison.csv"
    
    def log_training(self, message):
        """Log a message to training.log"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\\n")
        print(f"   📝 {message}")
    
    def log_model_result(self, model_name, feature_type, framework,
                        test_seen_acc, test_unseen_acc, test_seen_eer, test_unseen_eer,
                        training_time, inference_time, model_size, notes=""):
        """Add model result to CSV"""
        new_row = {
            'model_name': model_name,
            'feature_type': feature_type,
            'framework': framework,
            'test_seen_accuracy': test_seen_acc,
            'test_unseen_accuracy': test_unseen_acc,
            'test_seen_eer': test_seen_eer,
            'test_unseen_eer': test_unseen_eer,
            'training_time_seconds': training_time,
            'inference_time_ms': inference_time,
            'model_size_mb': model_size,
            'notes': notes
        }
        
        df = pd.read_csv(self.results_csv)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(self.results_csv, index=False)
        
        self.log_training(f"✅ {model_name}: {test_unseen_acc:.2%} accuracy, {test_unseen_eer:.2f}% EER")
    
    def update_comparison(self, model_name, new_acc, new_eer):
        """Update before/after comparison"""
        df = pd.read_csv(self.comparison_csv)
        mask = df['Model'] == model_name
        
        if mask.any():
            idx = df[mask].index[0]
            df.loc[idx, 'New_TestUnseen_%'] = new_acc
            df.loc[idx, 'New_EER_%'] = new_eer
            
            old_acc = df.loc[idx, 'Old_TestUnseen_%']
            if pd.notna(new_acc) and pd.notna(old_acc):
                df.loc[idx, 'Improvement_%'] = new_acc - old_acc
            
            df.to_csv(self.comparison_csv, index=False)

# Usage in your training scripts:
# ════════════════════════════════════════════════════════════════════════════════════════
# from results_logger import ResultsLogger
#
# logger = ResultsLogger("vispoofdb/experiments/training_runs/run_20260601_123456")
#
# logger.log_training("Starting SVM+LFCC training...")
# # ... train model ...
# logger.log_model_result(
#     model_name="SVM + LFCC",
#     feature_type="LFCC (40-dim)",
#     framework="scikit-learn",
#     test_seen_acc=0.9819,
#     test_unseen_acc=0.8950,  # NEW: Should be higher than 99.62% if diverse data helps
#     test_seen_eer=1.71,
#     test_unseen_eer=5.23,
#     training_time=120,
#     inference_time=1.2,
#     model_size=2.5,
#     notes="Trained on diverse YouTube speakers"
# )
'''

helper_file = RUN_DIR / "results_logger.py"
with open(helper_file, 'w', encoding='utf-8') as f:
    f.write(helper_script)
print(f"✅ results_logger.py helper created")

# 7. Summary README
readme = f"""# Training Run Results - {TIMESTAMP}

## Overview
This folder contains all results from retraining models with diverse YouTube speaker data.

## Files
- **metadata.json** — Run metadata and configuration
- **model_results.csv** — Individual model performance metrics
- **before_after_comparison.csv** — Comparison: Old (6 speakers) vs New (diverse)
- **data_comparison.csv** — Dataset statistics
- **training.log** — Detailed training log (append results as training progresses)
- **results_logger.py** — Helper to log results programmatically

## How to Use

### Option 1: Manual Logging (Quick)
After training each model, manually add row to **model_results.csv**:
```
Model,Feature,Framework,Test_Seen_Acc,Test_Unseen_Acc,Test_Seen_EER,Test_Unseen_EER,Train_Time,Inference_Time,Size,Notes
SVM + LFCC,LFCC (40-d),scikit-learn,98.19,89.50,1.71,5.23,120,1.2,2.5,"New diverse data"
```

### Option 2: Programmatic Logging (Recommended)
```python
from results_logger import ResultsLogger

logger = ResultsLogger("vispoofdb/experiments/training_runs/run_{TIMESTAMP}")
logger.log_training("Starting SVM+LFCC training...")

# Train model...

logger.log_model_result(
    model_name="SVM + LFCC",
    feature_type="LFCC (40-dim)",
    framework="scikit-learn",
    test_seen_acc=0.9819,
    test_unseen_acc=0.8950,
    test_seen_eer=1.71,
    test_unseen_eer=5.23,
    training_time=120,
    inference_time=1.2,
    model_size=2.5,
    notes="Trained on diverse YouTube speakers"
)
```

## Expected Results

### Before (6 speakers, speaker leakage):
- SVM+LFCC: 99.62% → 30% on your voice
- AASIST: 99.96% → 25% on your voice

### After (diverse speakers):
- SVM+LFCC: Expect ~80-85% (real generalization)
- AASIST: Expect ~82-87% (real generalization)
- Your voice: Expect ~75-85% (HUGE improvement!)

## Metrics to Track

For each model, record:
1. **Test_Seen_Accuracy** — Accuracy on speakers seen during training
2. **Test_Unseen_Accuracy** — Accuracy on NEW speakers (most important!)
3. **Test_Seen_EER** — Equal Error Rate (false accept + false reject)
4. **Test_Unseen_EER** — EER on new speakers
5. **Training_Time** — How long training took
6. **Inference_Time** — Milliseconds per prediction
7. **Model_Size** — MB on disk

## Success Criteria

✅ **Goal: Get realistic accuracy that transfers to your voice!**

| Metric | Old | Target |
|--------|-----|--------|
| Test_Unseen_Acc | 99.62% | 80-85% |
| Your Voice Acc | 20-30% | 75-85% |
| Generalization Gap | -70% | -5% |

## Notes
- This run uses HuggingFace YouTube speaker data
- Goal: Fix speaker leakage by training on diverse speakers
- Expected outcome: 50-60% improvement on new speakers
- Timeline: Full retraining ~2-4 hours

---
Generated: {datetime.now().isoformat()}
"""

readme_file = RUN_DIR / "README.md"
with open(readme_file, 'w', encoding='utf-8') as f:
    f.write(readme)
print(f"✅ README.md created")

# ════════════════════════════════════════════════════════════════════════════════════════
# Print summary
# ════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "="*90)
print("📋 RESULTS STRUCTURE CREATED")
print("="*90)

print(f"""
Run ID: {TIMESTAMP}
Location: {RUN_DIR}

Files created:
  ✅ metadata.json                  — Run configuration
  ✅ model_results.csv              — Model metrics (add during training)
  ✅ before_after_comparison.csv    — Old vs New accuracy
  ✅ data_comparison.csv            — Dataset stats
  ✅ training.log                   — Detailed training log
  ✅ results_logger.py              — Helper for programmatic logging
  ✅ README.md                      — This guide

Next steps:
  1. Start training models one by one
  2. For each model, save results to model_results.csv
  3. Update before_after_comparison.csv with new accuracy
  4. Check README.md in this folder for usage guide

Expected timeline:
  - SVM+LFCC (1-2 min)
  - SVM+MFCC (1-2 min)
  - MLP+MFCC (5-10 min)
  - XGBoost+MFCC (2-3 min)
  - AASIST (30-60 min GPU, longer CPU)
  - Total: ~1-2 hours

Expected improvement:
  - Old: 99.62% (with speaker leakage)
  - New: 80-85% (realistic with diverse speakers)
  - Your voice: 20-30% → 75-85% (BIG jump!)
""")

print("="*90 + "\n")

# Save run directory path to a file for quick reference
with open(RESULTS_DIR / "last_run.txt", 'w') as f:
    f.write(str(RUN_DIR))

print(f"✅ Run directory: {RUN_DIR}")
print(f"✅ Ready to start training!\n")
