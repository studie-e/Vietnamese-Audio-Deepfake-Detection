# Training Run Results - 20260601_211827

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

logger = ResultsLogger("vispoofdb/experiments/training_runs/run_20260601_211827")
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
Generated: 2026-06-01T21:18:27.473031
