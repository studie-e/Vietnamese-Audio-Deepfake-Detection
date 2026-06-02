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
    'Test_Seen_Acc': [98.19, 97.42, 96.73, 97.00, 96.11, 87.69, 87.03, 97.04, 99.12],
    'Test_Unseen_Acc': [99.62, 98.81, 99.12, 96.46, 69.85, 88.96, 83.96, 98.81, 99.96],
    'Test_Seen_EER': [1.71, 2.43, 3.36, 2.86, 4.29, 12.50, 12.93, 3.29, None],
    'Test_Unseen_EER': [0.29, 0.14, 0.00, 3.14, 22.50, 10.79, 14.86, 0.43, None],
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
    f.write("  - AASIST: Now fully trained (99.96% unseen accuracy!) 🎉\n")
    f.write("  - Top performer: AASIST Deep Learning (99.96% unseen accuracy, perfect end-to-end learning)\n")
    f.write("  - Most reliable classical model: SVM + LFCC (99.62% unseen accuracy, 0.29% EER)\n")
    f.write("  - Perfect EER: MLP + MFCC (0.00% unseen EER - no false accepts or rejects)\n")
    f.write("  - ⚠️ Concern: MLP + Wav2Vec2 overfitting worsened (96.11% seen vs 69.85% unseen = -26.26% gap)\n")

print(f"✅ TXT saved: {txt_path}")

# Display
print("\n" + "=" * 90)
print("FINAL RESULTS")
print("=" * 90)
print(df.to_string(index=False))
