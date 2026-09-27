"""
indexer.py - V2 local sample library indexer.

The schema is additive so an existing drums.db can be migrated without
throwing away the user's library. Feature vectors remain JSON for portability;
a future ANN index can be layered on top without changing the database contract.
"""

import argparse
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor

from audio_features import FEATURE_VERSION, extract_features


DB_NAME = "drums.db"

V2_COLUMNS = {
    "feature_version": "INTEGER DEFAULT 1",
    "is_loop": "INTEGER DEFAULT 0",
    "rms_db": "REAL DEFAULT 0",
    "band_profile_json": "TEXT DEFAULT '[]'",
    "rolloff": "REAL DEFAULT 0",
    "spectral_flatness": "REAL DEFAULT 0",
    "category_confidence": "REAL DEFAULT 0",
    "category_probs_json": "TEXT DEFAULT '{}'",
    "path_category": "TEXT DEFAULT 'other'",
    "attack_ratio": "REAL DEFAULT 0",
    "body_ratio": "REAL DEFAULT 0",
    "tail_ratio": "REAL DEFAULT 0",
    "transient_rms": "REAL DEFAULT 0",
    "body_rms": "REAL DEFAULT 0",
    "tail_rms": "REAL DEFAULT 0",
}


def init_db(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            filename TEXT,
            kit_name TEXT,
            category TEXT,
            onset_time_ms REAL,
            attack_time_ms REAL,
            lat REAL,
            crest_db REAL,
            f0 REAL,
            f_sub REAL,
            f_top REAL,
            pitch_drop_st REAL,
            f0_conf REAL,
            att_low REAL,
            att_mid REAL,
            att_high REAL,
            sus_low REAL,
            sus_mid REAL,
            sus_high REAL,
            mel24_json TEXT,
            centroid REAL,
            noise_ratio REAL,
            decay_ms REAL,
            duration_ms REAL,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("PRAGMA table_info(samples)")
    existing = {row[1] for row in cur.fetchall()}
    for name, definition in V2_COLUMNS.items():
        if name not in existing:
            cur.execute(f"ALTER TABLE samples ADD COLUMN {name} {definition}")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_category ON samples(category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_is_loop ON samples(is_loop)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_f_sub ON samples(f_sub)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_crest ON samples(crest_db)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_kit ON samples(kit_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_version ON samples(feature_version)")
    conn.commit()
    return conn


def get_existing_paths(conn, min_version=FEATURE_VERSION):
    cur = conn.cursor()
    cur.execute("SELECT path FROM samples WHERE COALESCE(feature_version, 1) >= ?", (min_version,))
    return {row[0] for row in cur.fetchall()}


def scan_directory(directory):
    supported = {".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".m4a"}
    files = []
    for root, _, names in os.walk(directory):
        for name in names:
            if os.path.splitext(name)[1].lower() in supported:
                files.append(os.path.join(root, name))
    return files


def resolve_kit_name(file_path, base_folder):
    rel = os.path.relpath(file_path, base_folder)
    parts = rel.split(os.sep)
    if len(parts) > 1:
        if os.path.basename(os.path.abspath(base_folder)).lower() == "user":
            return parts[0]
        return os.path.basename(os.path.abspath(base_folder))
    return os.path.basename(os.path.abspath(base_folder))


def worker_extract(item):
    file_path, base_folder = item
    try:
        feats = extract_features(file_path)
        feats["kit_name"] = resolve_kit_name(file_path, base_folder)
        return feats
    except Exception as exc:
        return {"error": str(exc), "path": file_path}


def _record(feats):
    return (
        feats["path"], feats["filename"], feats["kit_name"], feats["category"],
        feats["onset_time_ms"], feats["attack_time_ms"], feats["lat"],
        feats["crest_db"], feats["f0"], feats["f_sub"], feats["f_top"],
        feats["pitch_drop_st"], feats["f0_conf"], feats["att_low"],
        feats["att_mid"], feats["att_high"], feats["sus_low"],
        feats["sus_mid"], feats["sus_high"], feats["mel24_json"],
        feats["centroid"], feats["noise_ratio"], feats["decay_ms"],
        feats["duration_ms"], feats["feature_version"], feats["is_loop"],
        feats["rms_db"], feats["band_profile_json"], feats["rolloff"],
        feats["spectral_flatness"], feats["category_confidence"],
        feats["category_probs_json"], feats["path_category"],
        feats["attack_ratio"], feats["body_ratio"], feats["tail_ratio"],
        feats["transient_rms"], feats["body_rms"], feats["tail_rms"],
    )


def index_folder(folder_path, kit_name=None, db_path=DB_NAME, force=False, num_workers=8):
    if not os.path.isdir(folder_path):
        print(f"Error: Directory does not exist: {folder_path}")
        return 0

    conn = init_db(db_path)
    existing = get_existing_paths(conn) if not force else set()

    files = scan_directory(folder_path)
    to_process = [p for p in files if os.path.abspath(p) not in existing]

    print(f"\nScanning: {folder_path}")
    print(f"Found {len(files)} audio files; processing {len(to_process)}.")

    if not to_process:
        conn.close()
        return 0

    items = [(p, folder_path) for p in to_process]
    total = len(items)
    success = 0
    errors = 0
    batch = []
    start = time.time()

    cols = [
        "path", "filename", "kit_name", "category", "onset_time_ms", "attack_time_ms", "lat",
        "crest_db", "f0", "f_sub", "f_top", "pitch_drop_st", "f0_conf",
        "att_low", "att_mid", "att_high", "sus_low", "sus_mid", "sus_high",
        "mel24_json", "centroid", "noise_ratio", "decay_ms", "duration_ms",
        "feature_version", "is_loop", "rms_db", "band_profile_json", "rolloff",
        "spectral_flatness", "category_confidence", "category_probs_json",
        "path_category", "attack_ratio", "body_ratio", "tail_ratio",
        "transient_rms", "body_rms", "tail_rms",
    ]
    placeholders = ",".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO samples ({','.join(cols)}) VALUES ({placeholders})"

    with ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        for i, feats in enumerate(executor.map(worker_extract, items), 1):
            if "error" in feats:
                errors += 1
                print(f"[ERROR] {feats['path']}: {feats['error']}")
                continue

            batch.append(_record(feats))
            success += 1

            if len(batch) >= 100 or i == total:
                conn.executemany(sql, batch)
                conn.commit()
                batch.clear()

            if i % 100 == 0 or i == total:
                elapsed = max(time.time() - start, 0.1)
                print(f"[{i}/{total}] {i / elapsed:.1f} files/s")

    conn.close()
    elapsed = max(time.time() - start, 0.1)
    print(f"Done: {success} indexed, {errors} errors, {elapsed:.1f}s.")
    return success


def print_stats(db_path=DB_NAME):
    if not os.path.exists(db_path):
        print("Database not found.")
        return

    conn = init_db(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM samples")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT kit_name) FROM samples")
    kits = cur.fetchone()[0]
    cur.execute("SELECT category, COUNT(*) FROM samples GROUP BY category ORDER BY COUNT(*) DESC")
    cats = cur.fetchall()

    print(f"\nSamples: {total}")
    print(f"Kits: {kits}")
    for category, count in cats:
        print(f"  {category:8} {count}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a local drum/sample folder.")
    parser.add_argument("folder")
    parser.add_argument("--db", default=DB_NAME)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    index_folder(args.folder, db_path=args.db, force=args.force, num_workers=args.workers)
