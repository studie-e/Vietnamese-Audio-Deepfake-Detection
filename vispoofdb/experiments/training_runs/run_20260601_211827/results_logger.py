#!/usr/bin/env python3
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
            f.write(f"[{timestamp}] {message}\n")
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
