"""
test_v2.py - lightweight regression tests for the producer-oriented matcher.
Run with: python -m pytest -q
"""

import math
import os
import sqlite3
import tempfile

import numpy as np

from audio_features import (
    TARGET_SR,
    classify_drum_role,
    compute_band_profile,
    compute_envelope_features,
    extract_features,
    is_loop_sample,
)
from indexer import init_db
from matcher import compute_similarity


def _tone(freq, seconds=0.18, sr=TARGET_SR):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t * 18)).astype(np.float32)


def test_schema_has_v2_columns():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "test.db")
        conn = init_db(db)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)").fetchall()}
        conn.close()
        assert "is_loop" in cols
        assert "category_probs_json" in cols
        assert "band_profile_json" in cols


def test_loop_classifier():
    assert is_loop_sample("", "kick.wav", 180, "kick") == 0
    assert is_loop_sample("", "drum_loop_90bpm.wav", 800, "kick") == 1
    assert is_loop_sample("", "snare.wav", 800, "snare") == 0


def test_extract_raw_slice_has_acoustic_role():
    audio = _tone(70, 0.20)
    feats = extract_features(audio, category=None)
    assert feats["category"] in {"kick", "808", "perc", "other"}
    assert 0.0 <= feats["category_confidence"] <= 1.0
    assert len(feats["mel24_json"]) > 2


def test_similarity_exposes_producer_dimensions():
    audio = _tone(90, 0.20)
    ref = extract_features(audio, category="kick")
    cand = extract_features(audio * 0.9, category="kick")
    score = compute_similarity(ref, cand, "kick")
    assert score["total"] > 0.85
    assert score["sim_attack"] >= 0.0
    assert score["sim_body"] >= 0.0
    assert score["sim_tail"] >= 0.0
    assert score["sim_mel24"] >= 0.0


def test_indexer_insert_and_migration():
    import soundfile as sf
    from indexer import index_folder

    with tempfile.TemporaryDirectory() as d:
        audio_dir = os.path.join(d, "samples")
        os.makedirs(audio_dir)
        wav_path = os.path.join(audio_dir, "test_kick.wav")
        sf.write(wav_path, _tone(60, 0.25), TARGET_SR)

        db_path = os.path.join(d, "drums.db")
        count = index_folder(audio_dir, db_path=db_path)
        assert count == 1

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM samples WHERE filename='test_kick.wav'").fetchone())
        conn.close()

        assert row["feature_version"] == 2
        assert row["is_loop"] == 0
        assert row["band_profile_json"] != "[]"
        assert row["attack_ratio"] > 0.0


def test_whole_vs_portion_matching():
    import soundfile as sf
    from matcher import match_sample

    with tempfile.TemporaryDirectory() as d:
        # Create a long 808-like tone: 0.8s long with gradual decay
        t = np.arange(int(0.80 * TARGET_SR), dtype=np.float32) / TARGET_SR
        audio_808 = (np.sin(2 * np.pi * 55.0 * t) * np.exp(-t * 3.0)).astype(np.float32)

        wav_path = os.path.join(d, "sample_808.wav")
        sf.write(wav_path, audio_808, TARGET_SR)

        db_path = os.path.join(d, "drums.db")
        from indexer import index_folder
        index_folder(d, db_path=db_path)

        # Match whole sample
        matches_whole, ref_whole = match_sample(wav_path, db_path=db_path)
        assert ref_whole["duration_ms"] >= 750.0
        assert ref_whole["decay_ms"] > 150.0
        assert len(matches_whole) >= 1
        assert matches_whole[0][0] >= 0.99  # Self-match is ~100%

        # Match short portion (first 100ms)
        matches_portion, ref_portion = match_sample(
            wav_path, db_path=db_path, time_range=(0.0, 0.10)
        )
        assert ref_portion["duration_ms"] <= 105.0
        assert ref_portion["decay_ms"] < ref_whole["decay_ms"]
