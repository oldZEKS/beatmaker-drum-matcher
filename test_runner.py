"""
test_runner.py - Automated tests for drum matcher pipeline.
"""

import os
import glob
import sqlite3
from audio_features import extract_features, load_audio
from matcher import compute_similarity


def _extract_test_feats():
    snare_dir = r"C:\Users\ZEKSTK\Documents\Image-Line\User\Capital Drum Kit 4\Snare"
    if not os.path.exists(snare_dir):
        return None
    files = glob.glob(os.path.join(snare_dir, "*.wav"))
    if not files:
        return None
    
    test_file = files[0]
    feats = extract_features(test_file)

    assert feats['crest_db'] > 0, "Crest factor should be positive"
    assert feats['f0'] > 0, "f0 should be positive"
    assert feats['centroid'] > 100, "Centroid should be reasonable"
    assert 0.0 <= feats['noise_ratio'] <= 1.0, "Noise ratio must be between 0 and 1"
    assert feats['decay_ms'] > 0, "Decay time should be positive"
    return feats


def test_feature_extraction():
    feats = _extract_test_feats()
    if feats is not None:
        assert isinstance(feats, dict)


def test_self_match():
    feats = _extract_test_feats()
    if feats is None:
        return
    sims = compute_similarity(feats, feats, "snare")
    assert abs(sims['total'] - 1.0) < 1e-4, f"Self-match must be 1.0, got {sims['total']}"
    assert sims['tuning_cents'] == 0, "Self-match tuning must be 0 cents"


if __name__ == '__main__':
    feats = test_feature_extraction()
    test_self_match(feats)
    print("ALL TESTS PASSED SUCCESSFULLY! Ready to index full libraries.")
