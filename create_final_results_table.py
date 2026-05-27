"""
Create final results table for all trained models
"""

import pandas as pd
from pathlib import Path

# Results from training output
results = {
    'Model': [
        'SVM + LFCC',
        'SVM + MFCC',
        'MLP + MFCC',
        'XGBoost + MFCC',
        'MLP + Wav2Vec2',
        'SVM + Tone-Aware',
        'XGBoost + Tone-Aware',
        'SVM + MFCC+Tone Fusion',
        'AASIST (Deep Learning)',
    ],
    'Test_Seen_Acc': [92.82, 82.71, 83.73, 87.04, 95.26, 83.29, 82.74, 84.94, 82.0],  # AASIST Epoch 2 value
    'Test_Unseen_Acc': [96.79, 88.81, 90.42, 84.71, 76.55, 76.80, 74.36, 87.33, 81.98],  # AASIST Epoch 2 value
    'Test_Seen_EER': [5.92, 14.74, 13.52, 11.76, 5.00, 17.00, 17.27, 12.80, None],
    'Test_Unseen_EER': [3.53, 3.67, 1.84, 15.39, 23.69, 22.88, 25.56, 6.10, None],
}

df = pd.DataFrame(results)

# Save to CSV
output_dir = Path('vispoofdb/experiments')
output_dir.mkdir(parents=True, exist_ok=True)

csv_path = output_dir / 'results_summary_final.csv'
df.to_csv(csv_path, index=False)
print(f"✅ CSV saved: {csv_path}")

# Save to TXT
txt_path = output_dir / 'results_summary_final.txt'
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("=" * 90 + "\n")
    f.write("VISPOOFDB — ALL MODELS PERFORMANCE SUMMARY (FINAL)\n")
    f.write("=" * 90 + "\n\n")
    
    f.write("TABLE 1: Accuracy Comparison (%) - Higher is Better\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Model':<40} {'Test_Seen (%)':<20} {'Test_Unseen (%)':<20}\n")
    f.write("-" * 90 + "\n")
    
    for idx, row in df.iterrows():
        f.write(f"{row['Model']:<40} {row['Test_Seen_Acc']:<20.2f} {row['Test_Unseen_Acc']:<20.2f}\n")
    
    f.write("\n\n")
    f.write("TABLE 2: Equal Error Rate (EER) - Lower is Better\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Model':<40} {'Test_Seen (%)':<20} {'Test_Unseen (%)':<20}\n")
    f.write("-" * 90 + "\n")
    
    for idx, row in df.iterrows():
        seen_eer = f"{row['Test_Seen_EER']:.2f}" if pd.notna(row['Test_Seen_EER']) else "N/A"
        unseen_eer = f"{row['Test_Unseen_EER']:.2f}" if pd.notna(row['Test_Unseen_EER']) else "N/A"
        f.write(f"{row['Model']:<40} {seen_eer:<20} {unseen_eer:<20}\n")
    
    f.write("\n" + "=" * 90 + "\n")
    f.write("KEY FINDINGS\n")
    f.write("=" * 90 + "\n\n")
    
    # Best performers
    best_unseen_acc_idx = df['Test_Unseen_Acc'].idxmax()
    best_unseen_eer_idx = df[df['Test_Unseen_EER'].notna()]['Test_Unseen_EER'].idxmin()
    best_seen_acc_idx = df['Test_Seen_Acc'].idxmax()
    
    f.write(f"✅ Best Accuracy (Test_Unseen): {df.loc[best_unseen_acc_idx, 'Model']} — {df.loc[best_unseen_acc_idx, 'Test_Unseen_Acc']:.2f}%\n")
    f.write(f"✅ Best EER (Test_Unseen):      {df.loc[best_unseen_eer_idx, 'Model']} — {df.loc[best_unseen_eer_idx, 'Test_Unseen_EER']:.2f}%\n")
    f.write(f"✅ Best Test_Seen Accuracy:    {df.loc[best_seen_acc_idx, 'Model']} — {df.loc[best_seen_acc_idx, 'Test_Seen_Acc']:.2f}%\n\n")
    
    f.write("📝 Notes:\n")
    f.write("  - AASIST: Epoch 2/20 results (training interrupted)\n")
    f.write("  - Top performer: SVM + LFCC (96.79% unseen accuracy)\n")
    f.write("  - Most reliable (lowest EER): MLP + MFCC (1.84% unseen EER)\n")
    f.write("  - Best fusion: SVM + MFCC+Tone Fusion (87.33% accuracy, 6.10% EER)\n")

print(f"✅ TXT saved: {txt_path}")

# Display
print("\n" + "=" * 90)
print("FINAL RESULTS")
print("=" * 90)
print(df.to_string(index=False))
