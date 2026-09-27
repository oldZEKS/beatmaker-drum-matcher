"""
matcher.py - High-Precision Drum Sample Matcher with 24-Band Mel Micro-Timbre.
"""

import os
import sys
import math
import json
import sqlite3
import argparse
import winsound
import numpy as np
from audio_features import extract_features, load_audio, debleed_audio


DB_NAME = "drums.db"


def cosine_sim_3(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    norm1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2) + 1e-9
    norm2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2) + 1e-9
    return float(np.clip(dot / (norm1 * norm2), 0.0, 1.0))


def cosine_sim_mel(v1, v2):
    """
    Computes cosine similarity between two 24-band Mel vectors.
    """
    if len(v1) != len(v2) or len(v1) == 0:
        return 0.5
    dot = float(np.dot(v1, v2))
    norm1 = float(np.linalg.norm(v1)) + 1e-9
    norm2 = float(np.linalg.norm(v2)) + 1e-9
    return float(np.clip(dot / (norm1 * norm2), 0.0, 1.0))


def compute_similarity(ref, cand, category='snare', modifier=None):
    cat = category.lower()

    # 1. Attack Dynamics (Crest Factor + Log-Attack Rise Time)
    crest_diff = abs(ref['crest_db'] - cand['crest_db'])
    sim_crest = math.exp(- (crest_diff / 4.5) ** 2)

    lat_diff = abs(ref['lat'] - cand['lat'])
    sim_lat = math.exp(- (lat_diff / 0.5) ** 2)

    sim_attack = 0.60 * sim_crest + 0.40 * sim_lat

    # 2. Pitch & Fundamental Contour
    if cat in ('kick', '808'):
        ref_sub = max(ref['f_sub'], 20.0)
        cand_sub = max(cand['f_sub'], 20.0)
        sub_cents = abs(1200.0 * math.log2(cand_sub / ref_sub))
        sim_sub = math.exp(- (sub_cents / 220.0) ** 2)

        drop_diff = abs(ref['pitch_drop_st'] - cand['pitch_drop_st'])
        sim_drop = math.exp(- (drop_diff / 6.0) ** 2)

        ref_top = max(ref['f_top'], 50.0)
        cand_top = max(cand['f_top'], 50.0)
        top_cents = abs(1200.0 * math.log2(cand_top / ref_top))
        sim_top = math.exp(- (top_cents / 500.0) ** 2)

        sim_pitch = 0.55 * sim_sub + 0.30 * sim_drop + 0.15 * sim_top
        tuning_cents = round(1200.0 * math.log2(ref_sub / cand_sub))
    else:
        ref_f0 = max(ref['f_sub'], 20.0)
        cand_f0 = max(cand['f_sub'], 20.0)
        cents_diff = abs(1200.0 * math.log2(cand_f0 / ref_f0))
        sim_pitch = math.exp(- (cents_diff / 250.0) ** 2)
        tuning_cents = round(1200.0 * math.log2(ref_f0 / cand_f0))

    # 3. 24-Band Mel Micro-Timbre
    cand_mel = cand.get('mel24')
    if cand_mel is None and 'mel24_json' in cand and cand['mel24_json']:
        try:
            cand_mel = json.loads(cand['mel24_json'])
        except Exception:
            cand_mel = [0.0] * 24

    ref_mel = ref.get('mel24', [0.0] * 24)
    if cand_mel and len(cand_mel) == 24 and len(ref_mel) == 24:
        sim_mel24 = cosine_sim_mel(ref_mel, cand_mel)
    else:
        sim_mel24 = 0.70

    # 4. Attack Frequency Distribution (Low / Mid / High)
    v_ref_att = (ref['att_low'], ref['att_mid'], ref['att_high'])
    v_cand_att = (cand['att_low'], cand['att_mid'], cand['att_high'])
    sim_att_bands = cosine_sim_3(v_ref_att, v_cand_att)

    # 5. Sustain Texture & Wires
    v_ref_sus = (ref['sus_low'], ref['sus_mid'], ref['sus_high'])
    v_cand_sus = (cand['sus_low'], cand['sus_mid'], cand['sus_high'])
    sim_sus_bands = cosine_sim_3(v_ref_sus, v_cand_sus)

    noise_diff = abs(ref['noise_ratio'] - cand['noise_ratio'])
    sim_noise = max(0.0, 1.0 - (noise_diff / 0.5))
    sim_sustain = 0.65 * sim_sus_bands + 0.35 * sim_noise

    # 6. Decay Time Envelope
    ref_dec = max(ref['decay_ms'], 5.0)
    cand_dec = max(cand['decay_ms'], 5.0)
    decay_ratio = abs(math.log(cand_dec / ref_dec))
    sim_decay = math.exp(- (decay_ratio / 0.5) ** 2)

    # Category Weights with Mel Micro-Timbre
    if cat in ('kick', '808'):
        weights = {
            'pitch': 0.30,
            'mel24': 0.25,
            'attack': 0.20,
            'decay': 0.15,
            'att_bands': 0.10
        }
    elif cat in ('snare', 'clap', 'rim'):
        weights = {
            'mel24': 0.25,
            'attack': 0.22,
            'pitch': 0.20,
            'sustain': 0.18,
            'decay': 0.15
        }
    elif cat == 'hat':
        weights = {
            'mel24': 0.30,
            'decay': 0.30,
            'sustain': 0.25,
            'attack': 0.15,
            'pitch': 0.00
        }
    else:
        weights = {
            'mel24': 0.25,
            'attack': 0.20,
            'pitch': 0.20,
            'sustain': 0.18,
            'decay': 0.17
        }

    # Relative Modifiers
    if modifier == 'punchier':
        weights['attack'] *= 2.0
    elif modifier == 'tighter':
        weights['decay'] *= 2.0
    elif modifier == 'darker':
        weights['mel24'] *= 1.5
    elif modifier == 'brighter':
        weights['mel24'] *= 1.5

    w_sum = sum(weights.values())
    weights = {k: v / w_sum for k, v in weights.items()}

    total_score = (
        weights['attack'] * sim_attack +
        weights.get('pitch', 0.0) * sim_pitch +
        weights['mel24'] * sim_mel24 +
        weights.get('sustain', 0.0) * sim_sustain +
        weights['decay'] * sim_decay
    )

    return {
        'total': total_score,
        'sim_attack': sim_attack,
        'sim_pitch': sim_pitch,
        'sim_mel24': sim_mel24,
        'sim_att_bands': sim_att_bands,
        'sim_sustain': sim_sustain,
        'sim_decay': sim_decay,
        'tuning_cents': tuning_cents
    }


def match_sample(ref_target, category=None, top_k=10, db_path=DB_NAME, modifier=None, apply_debleed=False, sample_type='oneshot'):
    """
    Matches reference audio (either file path OR in-memory slice numpy array).
    sample_type: 'oneshot' (default), 'loop', or 'all'
    """
    if isinstance(ref_target, str):
        if not os.path.exists(ref_target):
            print(f"Error: File not found: {ref_target}")
            return [], {}
        ref_feats = extract_features(ref_target, category=category, apply_debleed=apply_debleed)
        ref_title = os.path.basename(ref_target)
    else:
        ref_feats = extract_features(ref_target, sr=44100, category=category, apply_debleed=apply_debleed)
        ref_title = "Selected Audio Slice"

    if not os.path.exists(db_path):
        print(f"Error: Database '{db_path}' not found. Please run indexer.py first.")
        return [], {}

    cat = ref_feats['category']

    print("\n" + "=" * 68)
    print(f"🎯 REFERENCE HIT: {ref_title}")
    print(f"   Category     : {cat.upper()}{' [DE-BLEEDED]' if apply_debleed else ''} | Type Filter: {sample_type.upper()}")
    print(f"   Onset Offset : {ref_feats['onset_time_ms']} ms (Pre-delay trimmed)")
    print(f"   Attack Punch : {ref_feats['crest_db']} dB Crest | Rise-Time: {ref_feats['attack_time_ms']} ms")
    if cat in ('kick', '808'):
        print(f"   Sub Fund.    : {ref_feats['f_sub']} Hz | Click: {ref_feats['f_top']} Hz (Sweep: -{ref_feats['pitch_drop_st']} st)")
    else:
        print(f"   Body Fund.   : {ref_feats['f_sub']} Hz (Confidence: {int(ref_feats['f0_conf']*100)}%)")
    print(f"   Decay T30    : {ref_feats['decay_ms']} ms | Noise: {int(ref_feats['noise_ratio']*100)}%")
    if modifier:
        print(f"   Modifier     : +{modifier.upper()}")
    print("=" * 68)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if cat == 'snare':
        target_cats = ('snare', 'clap', 'rim')
    elif cat in ('clap', 'rim'):
        target_cats = ('clap', 'rim', 'snare')
    elif cat in ('kick', '808'):
        target_cats = ('kick', '808')
    elif cat == 'hat':
        target_cats = ('hat',)
    else:
        target_cats = (cat,)

    placeholders = ','.join('?' * len(target_cats))
    
    if sample_type == 'oneshot':
        query = f"SELECT * FROM samples WHERE category IN ({placeholders}) AND is_loop = 0"
        cur.execute(query, target_cats)
    elif sample_type == 'loop':
        query = f"SELECT * FROM samples WHERE category IN ({placeholders}) AND is_loop = 1"
        cur.execute(query, target_cats)
    else:
        query = f"SELECT * FROM samples WHERE category IN ({placeholders})"
        cur.execute(query, target_cats)

    candidates = cur.fetchall()

    if not candidates:
        if sample_type == 'oneshot':
            cur.execute("SELECT * FROM samples WHERE is_loop = 0")
        elif sample_type == 'loop':
            cur.execute("SELECT * FROM samples WHERE is_loop = 1")
        else:
            cur.execute("SELECT * FROM samples")
        candidates = cur.fetchall()

    conn.close()

    results = []
    for row in candidates:
        cand = dict(row)
        sims = compute_similarity(ref_feats, cand, category=cat, modifier=modifier)
        results.append((sims['total'], cand, sims))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k], ref_feats



def print_matches(matches, ref_feats):
    if not matches:
        print("No matches found.")
        return

    print(f"\nTop {len(matches)} Closest Drum Matches from Your Library:\n")

    for rank, (score, cand, sims) in enumerate(matches, 1):
        pct = round(score * 100, 1)
        tuning = sims['tuning_cents']
        if abs(tuning) < 15:
            tune_str = "in-key"
        elif tuning > 0:
            tune_str = f"tune +{round(tuning/100, 1)} st"
        else:
            tune_str = f"tune {round(tuning/100, 1)} st"

        is_kick = cand['category'] in ('kick', '808')
        tone_str = f"{cand['f_sub']} Hz ({tune_str})"
        if is_kick and cand['pitch_drop_st'] > 0:
            tone_str += f" [drop -{cand['pitch_drop_st']} st]"

        print(f"#{rank:<2} [{pct}% Match] {cand['filename']} ({cand['kit_name']})")
        print(f"    ├── Attack Snap & Rise : {int(sims['sim_attack']*100)}% ({cand['crest_db']} dB, {cand['attack_time_ms']} ms)")
        print(f"    ├── Tone & Resonance   : {int(sims['sim_pitch']*100)}% ({tone_str})")
        print(f"    ├── Mel Micro-Timbre   : {int(sims['sim_mel24']*100)}% (24-Band Contour)")
        print(f"    ├── Sustain & Wires    : {int(sims['sim_sustain']*100)}% ({int(cand['noise_ratio']*100)}% noise)")
        print(f"    └── Decay Length       : {int(sims['sim_decay']*100)}% ({cand['decay_ms']} ms)")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Match a reference drum hit to your local drum library")
    parser.add_argument("reference", help="Path to reference audio file (.wav)")
    parser.add_argument("--category", "-c", choices=['kick', 'snare', 'clap', 'rim', 'hat', '808', 'perc', 'auto'], default='auto')
    parser.add_argument("--type", "-t", choices=['oneshot', 'loop', 'all'], default='oneshot', help="Filter by sample type (default: oneshot)")
    parser.add_argument("--top", "-n", type=int, default=10, help="Number of results (default 10)")
    parser.add_argument("--modifier", "-m", choices=['punchier', 'tighter', 'darker', 'brighter'], help="Relative modifier")
    parser.add_argument("--debleed", action="store_true", help="Apply spectral de-bleeding for song slices")
    parser.add_argument("--no-audition", action="store_true", help="Skip interactive audition prompt")
    args = parser.parse_args()

    matches, ref_feats = match_sample(
        args.reference,
        category=None if args.category == 'auto' else args.category,
        top_k=args.top,
        modifier=args.modifier,
        apply_debleed=args.debleed,
        sample_type=args.type
    )

    print_matches(matches, ref_feats)

