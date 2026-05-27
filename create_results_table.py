#!/usr/bin/env python3
"""
Script to create a comprehensive results summary table for all trained models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Results collected from training output
results_data = {
    'Model': [
        'SVM + LFCC',
        'SVM + MFCC',
        'MLP + MFCC',
        'XGBoost + MFCC',
        'MLP + Wav2Vec2',
        'SVM + Tone-Aware',
        'XGBoost + Tone-Aware',
        'SVM + Fusion (MFCC+Tone)',
        'AASIST (Deep Learning)',
    ],
    'Test_Seen_Acc': [
        92.82, 82.71, 83.73, 87.04, 95.26, 83.29, 82.74, 84.94, None
    ],
    'Test_Unseen_Acc': [
        96.79, 88.81, 90.42, 84.71, 76.55, 76.80, 74.36, 87.33, None
    ],
    'Test_Seen_EER': [
        5.92, 14.74, 13.52, 11.76, 5.00, 17.00, 17.27, 12.80, None
    ],
    'Test_Unseen_EER': [
        3.53, 3.67, 1.84, 15.39, 23.69, 22.88, 25.56, 6.10, None
    ],
}

# Create DataFrame
df = pd.DataFrame(results_data)

# Save to CSV
output_path = Path(__file__).parent / 'vispoofdb' / 'experiments' / 'results_summary.csv'
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
print(f"\n✅ Results saved to: {output_path}")

# Print formatted table
print("\n" + "="*100)
print("  VISPOOFDB — ALL MODELS PERFORMANCE SUMMARY")
print("="*100)
print("\n📊 TABLE 1: Accuracy Comparison (%) - Higher is Better")
print("-"*100)
print(f"{'Model':<35} {'Test_Seen (%)':<20} {'Test_Unseen (%)':<20}")
print("-"*100)
for idx, row in df.iterrows():
    model = row['Model']
    seen_acc = f"{row['Test_Seen_Acc']:.2f}" if row['Test_Seen_Acc'] is not None else "TBD"
    unseen_acc = f"{row['Test_Unseen_Acc']:.2f}" if row['Test_Unseen_Acc'] is not None else "TBD"
    star = " ⭐" if row['Test_Unseen_EER'] is not None and row['Test_Unseen_EER'] < 10 else ""
    print(f"{model:<35} {seen_acc:<20} {unseen_acc:<20} {star}")
print()

print("\n📊 TABLE 2: Equal Error Rate (EER) - Lower is Better")
print("-"*100)
print(f"{'Model':<35} {'Test_Seen (%)':<20} {'Test_Unseen (%)':<20}")
print("-"*100)
for idx, row in df.iterrows():
    model = row['Model']
    seen_eer = f"{row['Test_Seen_EER']:.2f}" if row['Test_Seen_EER'] is not None else "TBD"
    unseen_eer = f"{row['Test_Unseen_EER']:.2f}" if row['Test_Unseen_EER'] is not None else "TBD"
    star = " ✓" if row['Test_Unseen_EER'] is not None and row['Test_Unseen_EER'] < 10 else ""
    print(f"{model:<35} {seen_eer:<20} {unseen_eer:<20} {star}")
print()

print("\n" + "="*100)
print("  KEY FINDINGS")
print("="*100)

# Best on test_unseen accuracy
df_valid = df[df['Test_Unseen_Acc'].notna()]
best_acc_idx = df_valid['Test_Unseen_Acc'].idxmax()
best_acc_model = df.loc[best_acc_idx, 'Model']
best_acc_val = df.loc[best_acc_idx, 'Test_Unseen_Acc']
print(f"\n🏆 Best Accuracy (Test_Unseen): {best_acc_model} — {best_acc_val:.2f}%")

# Best EER on test_unseen
best_eer_idx = df_valid['Test_Unseen_EER'].idxmin()
best_eer_model = df.loc[best_eer_idx, 'Model']
best_eer_val = df.loc[best_eer_idx, 'Test_Unseen_EER']
print(f"🎯 Best EER (Test_Unseen):       {best_eer_model} — {best_eer_val:.2f}%")

# Best on test_seen
best_seen_idx = df_valid['Test_Seen_Acc'].idxmax()
best_seen_model = df.loc[best_seen_idx, 'Model']
best_seen_val = df.loc[best_seen_idx, 'Test_Seen_Acc']
print(f"📌 Best Test_Seen Accuracy:      {best_seen_model} — {best_seen_val:.2f}%")

print("\n" + "="*100)
print("📝 Note: AASIST results pending. Full results available once AASIST training completes.")
print("="*100 + "\n")

# Save to text file for easy viewing
txt_output = output_path.parent / 'results_summary.txt'
with open(txt_output, 'w', encoding='utf-8') as f:
    f.write("="*100 + "\n")
    f.write("  VISPOOFDB — ALL MODELS PERFORMANCE SUMMARY\n")
    f.write("="*100 + "\n\n")
    f.write("TABLE 1: Accuracy Comparison (%) - Higher is Better\n")
    f.write("-"*100 + "\n")
    f.write(f"{'Model':<35} {'Test_Seen (%)':<20} {'Test_Unseen (%)':<20}\n")
    f.write("-"*100 + "\n")
    for idx, row in df.iterrows():
        model = row['Model']
        seen_acc = f"{row['Test_Seen_Acc']:.2f}" if row['Test_Seen_Acc'] is not None else "TBD"
        unseen_acc = f"{row['Test_Unseen_Acc']:.2f}" if row['Test_Unseen_Acc'] is not None else "TBD"
        f.write(f"{model:<35} {seen_acc:<20} {unseen_acc:<20}\n")
    f.write("\n\nTABLE 2: Equal Error Rate (EER) - Lower is Better\n")
    f.write("-"*100 + "\n")
    f.write(f"{'Model':<35} {'Test_Seen (%)':<20} {'Test_Unseen (%)':<20}\n")
    f.write("-"*100 + "\n")
    for idx, row in df.iterrows():
        model = row['Model']
        seen_eer = f"{row['Test_Seen_EER']:.2f}" if row['Test_Seen_EER'] is not None else "TBD"
        unseen_eer = f"{row['Test_Unseen_EER']:.2f}" if row['Test_Unseen_EER'] is not None else "TBD"
        f.write(f"{model:<35} {seen_eer:<20} {unseen_eer:<20}\n")
    f.write("\n" + "="*100 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("="*100 + "\n\n")
    f.write(f"Best Accuracy (Test_Unseen): {best_acc_model} — {best_acc_val:.2f}%\n")
    f.write(f"Best EER (Test_Unseen):      {best_eer_model} — {best_eer_val:.2f}%\n")
    f.write(f"Best Test_Seen Accuracy:     {best_seen_model} — {best_seen_val:.2f}%\n")

print(f"✅ Also saved to: {txt_output}")
