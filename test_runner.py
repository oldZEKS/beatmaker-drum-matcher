"""
test_runner.py - Automated tests for drum matcher pipeline.
"""

import os
import glob
import sqlite3
from audio_features import extract_features, load_audio
from matcher import compute_similarity, WEIGHT_PROFILES


def test_feature_extraction():
    snare_dir = r"C:\Users\ZEKSTK\Documents\Image-Line\User\Capital Drum Kit 4\Snare"
    files = glob.glob(os.path.join(snare_dir, "*.wav"))
    assert len(files) > 0, "No test snares found!"
    
    test_file = files[0]
    print(f"Testing feature extraction on: {os.path.basename(test_file)}")
    feats = extract_features(test_file)
    
    print("Extracted features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")

    assert feats['crest_db'] > 0, "Crest factor should be positive"
    assert feats['f0'] > 0, "f0 should be positive"
    assert feats['centroid'] > 100, "Centroid should be reasonable"
    assert 0.0 <= feats['noise_ratio'] <= 1.0, "Noise ratio must be between 0 and 1"
    assert feats['decay_ms'] > 0, "Decay time should be positive"
    print("Feature extraction assertion PASSED!\n")
    return feats


def test_self_match(feats):
    print("Testing self-match (matching sample against itself)...")
    weights = WEIGHT_PROFILES['snare']
    sims = compute_similarity(feats, feats, weights)
    
    print(f"Total Self-Match Score: {round(sims['total'] * 100, 2)}%")
    print(f"  Transient: {round(sims['transient'], 3)}")
    print(f"  Body f0  : {round(sims['body_f0'], 3)}")
    print(f"  Centroid : {round(sims['brightness'], 3)}")
    print(f"  Noise    : {round(sims['noise'], 3)}")
    print(f"  Decay    : {round(sims['decay'], 3)}")
    
    assert abs(sims['total'] - 1.0) < 1e-4, f"Self-match must be 1.0, got {sims['total']}"
    assert sims['tuning_cents'] == 0, "Self-match tuning must be 0 cents"
    print("Self-match assertion PASSED!\n")


if __name__ == '__main__':
    feats = test_feature_extraction()
    test_self_match(feats)
    print("ALL TESTS PASSED SUCCESSFULLY! Ready to index full libraries.")
