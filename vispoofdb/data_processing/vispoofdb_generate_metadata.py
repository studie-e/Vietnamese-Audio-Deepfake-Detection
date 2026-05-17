import os
import pandas as pd
from pathlib import Path

# ════════════════════════════════════════════════════════════════
#  Tạo lại metadata.csv cho vispoofdb_processed/fake/
#  dựa trên cấu trúc file hiện tại
# ════════════════════════════════════════════════════════════════

# ── Mapping giữa source và technique (từ notebook)
TECHNIQUE_MAP = {
    "vivos":      "natural",
    "vlsp":       "natural",
    "mmstts":     "neural_tts",
    "edgetts":    "neural_tts_commercial",
    "fptai":      "neural_tts_commercial",
    "viettel":    "neural_tts_commercial",
    "elevenlabs": "voice_clone",
    "minimax":    "neural_tts_multilingual",
    "zaloai":     "neural_tts_commercial",
    "gtts":       "concatenative_tts",
}

# ── Các source là UNSEEN (chỉ dùng test)
UNSEEN_SOURCES = {"gtts"}

def generate_metadata_for_vispoofdb():
    """
    Tạo metadata.csv cho thư mục vispoofdb_processed/fake/
    Quét tất cả file audio từ các thư mục con
    """
    
    # Thư mục fake
    fake_base = Path(r"D:\hien\Dai hoc\Nam 2\ki2\seminar\Vietnamese-Audio-Deepfake-Detection\vispoofdb_processed\fake")
    metadata_file = fake_base.parent / "metadata.csv"
    
    if not fake_base.exists():
        print(f"❌ Thư mục không tồn tại: {fake_base}")
        return
    
    print(f"🔄 Đang quét thư mục: {fake_base}")
    print(f"{'='*70}")
    
    rows = []
    total_files = 0
    
    # Duyệt tất cả thư mục con (edgetts, elevenlabs, fptai, gtts, mmstts, zaloai)
    for source_dir in sorted(fake_base.iterdir()):
        if not source_dir.is_dir():
            continue
        
        source_name = source_dir.name
        
        # Tìm tất cả file audio (.wav, .mp3)
        audio_files = sorted(list(source_dir.glob("*.wav")) + list(source_dir.glob("*.mp3")))
        
        if not audio_files:
            print(f"⚠️  {source_name}: không có file audio")
            continue
        
        print(f"\n📂 {source_name}")
        print(f"   Số file: {len(audio_files)}")
        
        # Xác định split (test_unseen nếu là gtts, còn lại là pending -> sẽ chia train/test sau)
        split = "test_unseen" if source_name in UNSEEN_SOURCES else "pending"
        technique = TECHNIQUE_MAP.get(source_name, "unknown")
        
        print(f"   Split: {split}")
        print(f"   Technique: {technique}")
        
        # Tạo row cho mỗi file
        for i, audio_file in enumerate(audio_files):
            # Tính đường dẫn tương đối từ project root
            relative_path = audio_file.relative_to(fake_base.parent.parent)
            relative_path_str = str(relative_path).replace("\\", "/")
            
            rows.append({
                "file_id": audio_file.stem,  # Tên file không extension (e.g., edgetts_0000)
                "file_path": relative_path_str,  # Đường dẫn tương đối (e.g., vispoofdb_processed/fake/edgetts/edgetts_0000.wav)
                "label": "fake",
                "source": source_name,
                "technique": technique,
                "split": split,
            })
        
        total_files += len(audio_files)
    
    if not rows:
        print("\n❌ Không tìm thấy file audio nào!")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 Tổng: {total_files} file")
    
    # Tạo DataFrame
    df = pd.DataFrame(rows)
    
    # Chia train/test cho các source không phải UNSEEN
    seen_rows = df[df["split"] == "pending"]
    if len(seen_rows) > 0:
        # Shuffle và chia 80/20
        seen_shuffled = seen_rows.sample(frac=1, random_state=42)
        cut = int(len(seen_shuffled) * 0.8)
        
        train_indices = seen_shuffled.index[:cut]
        test_indices = seen_shuffled.index[cut:]
        
        df.loc[train_indices, "split"] = "train"
        df.loc[test_indices, "split"] = "test_seen"
        
        print(f"\n📊 Chia dữ liệu seen sources:")
        print(f"   Train: {len(train_indices)} file")
        print(f"   Test (seen): {len(test_indices)} file")
    
    print(f"   Test (unseen): {len(df[df['split'] == 'test_unseen'])} file")
    
    # Lưu metadata
    df.to_csv(metadata_file, index=False, encoding="utf-8")
    print(f"\n✅ Lưu metadata.csv: {metadata_file}")
    
    # Hiển thị thống kê
    print(f"\n{'='*70}")
    print("📊 Thống kê theo source + label:")
    stats = df.groupby(["source", "label"])["file_id"].count().reset_index()
    stats.columns = ["source", "label", "count"]
    print(stats.to_string(index=False))
    
    print(f"\n📊 Thống kê theo split:")
    print(df["split"].value_counts().sort_index().to_string())
    
    print(f"\n{'='*70}")
    print(f"\n✅ Hoàn thành! metadata.csv tại: {metadata_file}")
    
    return df

if __name__ == "__main__":
    df = generate_metadata_for_vispoofdb()
