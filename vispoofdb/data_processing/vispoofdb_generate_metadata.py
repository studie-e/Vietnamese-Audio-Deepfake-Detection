import sys
import os
import pandas as pd
from pathlib import Path

# Fix encoding cho terminal Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

# ════════════════════════════════════════════════════════════════
#  Tao lai metadata.csv cho vispoofdb/data/clean_data/
#  Xu ly ca real (file truc tiep) va fake (co thu muc con)
# ════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parents[2]
CLEAN_DATA_DIR = BASE_DIR / "vispoofdb" / "data" / "clean_data"

# ── Mapping giua source va technique
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

# ── Cac source la UNSEEN (chi dung test)
UNSEEN_SOURCES = {"gtts"}

# ── Source mac dinh cho thu muc real (khong co thu muc con)
REAL_DEFAULT_SOURCE = "vivos"


def collect_real(real_dir: Path, clean_data_root: Path) -> list[dict]:
    """
    Thu muc real chua file .wav truc tiep (khong co thu muc con).
    Gan source mac dinh la REAL_DEFAULT_SOURCE.
    """
    if not real_dir.exists():
        print(f"[WARN] Thu muc real khong ton tai: {real_dir}")
        return []

    audio_files = sorted(real_dir.glob("*.wav"))
    if not audio_files:
        print(f"[WARN] Khong co file .wav trong: {real_dir}")
        return []

    print(f"\n  Nguon: {REAL_DEFAULT_SOURCE}")
    print(f"  So file: {len(audio_files)}")
    print(f"  Technique: {TECHNIQUE_MAP.get(REAL_DEFAULT_SOURCE, 'unknown')}")
    print(f"  Split: pending (se chia train/test_seen sau)")

    rows = []
    technique = TECHNIQUE_MAP.get(REAL_DEFAULT_SOURCE, "unknown")
    for audio_file in audio_files:
        rel = audio_file.relative_to(clean_data_root)
        rows.append({
            "file_id":   audio_file.stem,
            "file_path": str(rel).replace("\\", "/"),
            "label":     "real",
            "source":    REAL_DEFAULT_SOURCE,
            "technique": technique,
            "split":     "pending",
        })

    return rows


def collect_fake(fake_dir: Path, clean_data_root: Path) -> list[dict]:
    """
    Thu muc fake co cac thu muc con (edgetts, elevenlabs, ...).
    Moi thu muc con la mot source.
    """
    if not fake_dir.exists():
        print(f"[WARN] Thu muc fake khong ton tai: {fake_dir}")
        return []

    rows = []
    source_dirs = sorted([d for d in fake_dir.iterdir() if d.is_dir()])

    if not source_dirs:
        # Truong hop khong co thu muc con - lay file truc tiep
        audio_files = sorted(fake_dir.glob("*.wav"))
        if audio_files:
            print(f"\n  (fake khong co thu muc con, dung source 'unknown')")
            for audio_file in audio_files:
                rel = audio_file.relative_to(clean_data_root)
                rows.append({
                    "file_id":   audio_file.stem,
                    "file_path": str(rel).replace("\\", "/"),
                    "label":     "fake",
                    "source":    "unknown",
                    "technique": "unknown",
                    "split":     "pending",
                })
        return rows

    for source_dir in source_dirs:
        source_name = source_dir.name
        audio_files = sorted(list(source_dir.glob("*.wav")) + list(source_dir.glob("*.mp3")))

        if not audio_files:
            print(f"\n  [WARN] {source_name}: khong co file audio")
            continue

        technique = TECHNIQUE_MAP.get(source_name, "unknown")
        split_label = "test_unseen" if source_name in UNSEEN_SOURCES else "pending"

        print(f"\n  Nguon: {source_name}")
        print(f"  So file: {len(audio_files)}")
        print(f"  Technique: {technique}")
        print(f"  Split: {split_label}")

        for audio_file in audio_files:
            rel = audio_file.relative_to(clean_data_root)
            rows.append({
                "file_id":   audio_file.stem,
                "file_path": str(rel).replace("\\", "/"),
                "label":     "fake",
                "source":    source_name,
                "technique": technique,
                "split":     split_label,
            })

    return rows


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split strategy:
    - Real: 60% train, 20% test_seen, 20% test_unseen (different speakers)
    - Fake gtts: 100% test_unseen (completely new technique)
    - Fake others: 80% train, 20% test_seen
    """
    result_parts = []

    # ── REAL: 60/20/20 split
    if "real" in df["label"].unique():
        real_df = df[df["label"] == "real"].copy()
        shuffled = real_df.sample(frac=1, random_state=42)
        
        n = len(shuffled)
        train_cut = int(n * 0.60)
        test_seen_cut = train_cut + int(n * 0.20)
        
        real_df.loc[shuffled.index[:train_cut], "split"] = "train"
        real_df.loc[shuffled.index[train_cut:test_seen_cut], "split"] = "test_seen"
        real_df.loc[shuffled.index[test_seen_cut:], "split"] = "test_unseen"
        
        result_parts.append(real_df)

    # ── FAKE: Handle gtts vs others separately
    if "fake" in df["label"].unique():
        fake_df = df[df["label"] == "fake"].copy()
        
        # Gtts: 100% test_unseen (completely new technique)
        gtts_df = fake_df[fake_df["source"] == "gtts"].copy()
        gtts_df["split"] = "test_unseen"
        result_parts.append(gtts_df)
        
        # Others: 80% train, 20% test_seen
        others_df = fake_df[fake_df["source"] != "gtts"].copy()
        shuffled = others_df.sample(frac=1, random_state=42)
        cut = int(len(shuffled) * 0.80)
        
        others_df.loc[shuffled.index[:cut], "split"] = "train"
        others_df.loc[shuffled.index[cut:], "split"] = "test_seen"
        result_parts.append(others_df)

    return pd.concat(result_parts).sort_index()


def generate_metadata():
    """
    Tao metadata.csv tong hop cho ca real va fake tu vispoofdb/data/clean_data/
    """
    print("=== TAO METADATA.CSV CHO VISPOOFDB ===")
    print(f"  Nguon du lieu: {CLEAN_DATA_DIR}\n")

    if not CLEAN_DATA_DIR.exists():
        print(f"[ERROR] Khong tim thay thu muc: {CLEAN_DATA_DIR}")
        return

    all_rows = []

    # ── 1. Thu thap du lieu REAL tu clean_data/
    real_dir = CLEAN_DATA_DIR / "real"
    print("=" * 60)
    print("REAL")
    real_rows = collect_real(real_dir, CLEAN_DATA_DIR)
    all_rows.extend(real_rows)

    # ── 2. Thu thap du lieu FAKE
    fake_dir = CLEAN_DATA_DIR / "fake"
    print(f"\n{'=' * 60}")
    print("FAKE")
    fake_rows = collect_fake(fake_dir, CLEAN_DATA_DIR)
    all_rows.extend(fake_rows)

    if not all_rows:
        print("\n[ERROR] Khong tim thay file audio nao!")
        return

    df = pd.DataFrame(all_rows)

    # ── 3. Chia split train / test_seen / test_unseen
    df = assign_splits(df)

    # ── 4. Luu metadata.csv
    metadata_file = CLEAN_DATA_DIR / "metadata.csv"
    df.to_csv(metadata_file, index=False, encoding="utf-8")

    # ── 5. Hien thi thong ke
    print(f"\n{'=' * 60}")
    print("THONG KE TONG HOP")
    print(f"  Tong so file: {len(df)}")
    print(f"    - real: {len(df[df['label'] == 'real'])}")
    print(f"    - fake: {len(df[df['label'] == 'fake'])}")
    print(f"\n  Thong ke theo split:")
    split_stats = df.groupby(["label", "split"])["file_id"].count().reset_index()
    split_stats.columns = ["label", "split", "count"]
    print(split_stats.to_string(index=False))
    print(f"\n  Thong ke theo source + label:")
    src_stats = df.groupby(["source", "label"])["file_id"].count().reset_index()
    src_stats.columns = ["source", "label", "count"]
    print(src_stats.to_string(index=False))

    print(f"\n  Luu metadata: {metadata_file}")
    print(f"{'=' * 60}")
    print("HOAN THANH!")

    return df


if __name__ == "__main__":
    df = generate_metadata()
