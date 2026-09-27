"""
matcher.py - V2 producer-oriented replacement matcher.

Ranking goal:
    "Which sample would I actually use to remake this hit?"
not:
    "Which file is mathematically closest in raw acoustics?"

The ranking is intentionally inspectable. A later CLAP/embedding model can be
added as a reranker without replacing these features.
"""

import argparse
import json
import math
import os
import sqlite3

import numpy as np

from audio_features import extract_features


DB_NAME = "drums.db"
ROLE_FAMILY = {
    "kick": ("kick", "808"),
    "808": ("808", "kick"),
    "snare": ("snare", "clap", "rim"),
    "clap": ("clap", "snare", "rim"),
    "rim": ("rim", "snare", "clap"),
    "hat": ("hat",),
    "perc": ("perc", "rim", "hat"),
    "other": ("other", "kick", "snare", "hat", "perc"),
}


def _gaussian(delta, scale):
    return float(math.exp(-((float(delta) / max(scale, 1e-6)) ** 2)))


def _cosine(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _load_json(value, fallback):
    try:
        parsed = json.loads(value) if value else fallback
        return parsed if parsed is not None else fallback
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


def _pitch_similarity(ref, cand, category):
    if category not in ("kick", "808", "snare", "clap", "rim"):
        return 0.5, 0

    rf = max(float(ref.get("f_sub", 0.0)), 20.0)
    cf = max(float(cand.get("f_sub", 0.0)), 20.0)

    cents = abs(1200.0 * math.log2(cf / rf))
    sim = _gaussian(cents, 180.0 if category in ("kick", "808") else 260.0)

    if category in ("kick", "808"):
        drop = abs(float(ref.get("pitch_drop_st", 0.0)) - float(cand.get("pitch_drop_st", 0.0)))
        sim = 0.72 * sim + 0.28 * _gaussian(drop, 5.5)

    tuning = int(round(1200.0 * math.log2(rf / cf)))
    return float(np.clip(sim, 0.0, 1.0)), tuning


def _role_compatibility(ref, cand):
    ref_probs = _load_json(ref.get("category_probs_json"), {})
    cand_probs = _load_json(cand.get("category_probs_json"), {})

    if not ref_probs or not cand_probs:
        return 1.0

    # Dot product of role affinities is softer than a hard category gate.
    keys = set(ref_probs) | set(cand_probs)
    score = sum(float(ref_probs.get(k, 0.0)) * float(cand_probs.get(k, 0.0)) for k in keys)
    return float(np.clip(score * 6.0, 0.0, 1.0))


def compute_similarity(ref, cand, category=None, modifier=None):
    category = (category or ref.get("category") or "other").lower()

    ref_mel = _load_json(ref.get("mel24_json"), ref.get("mel24", []))
    cand_mel = _load_json(cand.get("mel24_json"), cand.get("mel24", []))
    ref_band = _load_json(ref.get("band_profile_json"), [])
    cand_band = _load_json(cand.get("band_profile_json"), [])

    # 1) Attack: what makes the hit "speak"?
    attack_crest = _gaussian(
        float(ref.get("crest_db", 0.0)) - float(cand.get("crest_db", 0.0)), 4.0
    )
    attack_time = _gaussian(
        float(ref.get("attack_time_ms", 1.0)) - float(cand.get("attack_time_ms", 1.0)), 2.8
    )
    attack_shape = _gaussian(
        float(ref.get("attack_ratio", 0.0)) - float(cand.get("attack_ratio", 0.0)), 0.13
    )
    sim_attack = 0.45 * attack_crest + 0.35 * attack_time + 0.20 * attack_shape

    # 2) Body: the part that usually determines whether a replacement feels right.
    body_shape = _gaussian(
        float(ref.get("body_ratio", 0.0)) - float(cand.get("body_ratio", 0.0)), 0.13
    )
    body_rms = _gaussian(
        float(ref.get("body_rms", 0.0)) - float(cand.get("body_rms", 0.0)), 0.12
    )
    body_spectrum = _cosine(
        ref_band[:5], cand_band[:5]
    ) if len(ref_band) >= 5 and len(cand_band) >= 5 else 0.5
    sim_body = 0.45 * body_shape + 0.25 * body_rms + 0.30 * body_spectrum

    # 3) Tail/decay: especially important for hats, snares and roomy percussion.
    decay = _gaussian(
        math.log(max(float(ref.get("decay_ms", 5.0)), 5.0) /
                 max(float(cand.get("decay_ms", 5.0)), 5.0)), 0.38
    )
    tail_shape = _gaussian(
        float(ref.get("tail_ratio", 0.0)) - float(cand.get("tail_ratio", 0.0)), 0.10
    )
    sim_tail = 0.70 * decay + 0.30 * tail_shape

    # 4) Timbre contour: useful, but deliberately not allowed to dominate.
    sim_mel = _cosine(ref_mel, cand_mel) if len(ref_mel) == len(cand_mel) and ref_mel else 0.5

    # 5) Noise/wire character and broad spectral distribution.
    noise = _gaussian(
        float(ref.get("noise_ratio", 0.0)) - float(cand.get("noise_ratio", 0.0)), 0.18
    )
    sim_band = _cosine(ref_band, cand_band) if len(ref_band) == len(cand_band) and ref_band else 0.5
    sim_texture = 0.55 * noise + 0.45 * sim_band

    # 6) Role-specific pitch.
    sim_pitch, tuning_cents = _pitch_similarity(ref, cand, category)

    role = _role_compatibility(ref, cand)

    # Category weights encode how beatmakers tend to hear replacement identity.
    weights = {
        "attack": 0.28,
        "body": 0.26,
        "tail": 0.18,
        "timbre": 0.16,
        "texture": 0.08,
        "pitch": 0.04,
    }

    if category in ("kick", "808"):
        weights.update(attack=0.24, body=0.29, tail=0.16, timbre=0.12, texture=0.06, pitch=0.13)
    elif category in ("snare", "clap", "rim"):
        weights.update(attack=0.28, body=0.22, tail=0.18, timbre=0.17, texture=0.11, pitch=0.04)
    elif category == "hat":
        weights.update(attack=0.20, body=0.12, tail=0.30, timbre=0.22, texture=0.16, pitch=0.0)
    elif category == "perc":
        weights.update(attack=0.24, body=0.23, tail=0.22, timbre=0.18, texture=0.13, pitch=0.0)

    # Modifiers are ranking biases, not destructive transforms.
    if modifier == "punchier":
        weights["attack"] *= 1.55
        weights["body"] *= 0.85
    elif modifier == "tighter":
        weights["tail"] *= 1.60
        weights["body"] *= 0.90
    elif modifier in ("darker", "brighter"):
        weights["timbre"] *= 1.45
        weights["texture"] *= 0.90

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    total = (
        weights["attack"] * sim_attack +
        weights["body"] * sim_body +
        weights["tail"] * sim_tail +
        weights["timbre"] * sim_mel +
        weights["texture"] * sim_texture +
        weights["pitch"] * sim_pitch
    )

    # Role mismatch is a soft penalty, not an all-or-nothing exclusion.
    total *= 0.78 + 0.22 * role

    return {
        "total": float(np.clip(total, 0.0, 1.0)),
        "sim_attack": float(sim_attack),
        "sim_body": float(sim_body),
        "sim_tail": float(sim_tail),
        "sim_mel24": float(sim_mel),
        "sim_texture": float(sim_texture),
        "sim_pitch": float(sim_pitch),
        "role_compatibility": float(role),
        "tuning_cents": tuning_cents,
        "reason": {
            "attack": round(sim_attack, 3),
            "body": round(sim_body, 3),
            "tail": round(sim_tail, 3),
            "timbre": round(sim_mel, 3),
            "texture": round(sim_texture, 3),
            "pitch": round(sim_pitch, 3),
        },
    }


def _target_categories(category):
    return ROLE_FAMILY.get(category, ROLE_FAMILY["other"])


def _candidate_rows(conn, category, sample_type):
    placeholders = ",".join("?" for _ in _target_categories(category))
    args = list(_target_categories(category))
    where = f"category IN ({placeholders})"

    if sample_type == "oneshot":
        where += " AND COALESCE(is_loop, 0) = 0"
    elif sample_type == "loop":
        where += " AND COALESCE(is_loop, 0) = 1"

    cur = conn.cursor()
    cur.execute(f"SELECT * FROM samples WHERE {where}", args)
    return cur.fetchall()


def _prefilter(rows, ref, category, limit=2500):
    """
    Cheap coarse gate before the detailed score.

    It deliberately keeps a generous pool so an unusual but musically useful
    sample is not thrown away too early.
    """
    scored = []
    rf = float(ref.get("f_sub", 0.0))
    rc = float(ref.get("centroid", 0.0))
    rd = max(float(ref.get("decay_ms", 10.0)), 5.0)

    for row in rows:
        c = dict(row)
        cf = float(c.get("f_sub", 0.0))
        cc = float(c.get("centroid", 0.0))
        cd = max(float(c.get("decay_ms", 10.0)), 5.0)

        pitch = _gaussian(abs(1200.0 * math.log2(max(cf, 20.0) / max(rf, 20.0))), 900.0)
        centroid = _gaussian(rc - cc, 3000.0)
        decay = _gaussian(math.log(rd / cd), 1.0)

        role = _role_compatibility(ref, c)
        coarse = 0.45 * pitch + 0.30 * centroid + 0.25 * decay
        coarse *= 0.75 + 0.25 * role
        scored.append((coarse, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]


def match_sample(
    ref_target,
    category=None,
    top_k=10,
    db_path=DB_NAME,
    modifier=None,
    apply_debleed=False,
    sample_type="oneshot",
):
    if isinstance(ref_target, str):
        if not os.path.exists(ref_target):
            raise FileNotFoundError(ref_target)
        ref = extract_features(ref_target, category=category, apply_debleed=apply_debleed)
        title = os.path.basename(ref_target)
    else:
        ref = extract_features(
            ref_target, sr=44100, category=category, apply_debleed=apply_debleed
        )
        title = "Selected Audio Slice"

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database '{db_path}' not found. Index a drum-kit folder first."
        )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = _candidate_rows(conn, ref["category"], sample_type)
    if not rows and sample_type != "all":
        # Never fail silently when the user's library has incomplete metadata.
        rows = _candidate_rows(conn, ref["category"], "all")

    candidates = _prefilter(rows, ref, ref["category"])
    ranked = []

    for row in candidates:
        cand = dict(row)
        sims = compute_similarity(ref, cand, ref["category"], modifier)
        ranked.append((sims["total"], cand, sims))

    ranked.sort(key=lambda item: item[0], reverse=True)
    conn.close()

    return ranked[:max(1, int(top_k))], ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find usable replacement drum samples.")
    parser.add_argument("reference")
    parser.add_argument("--db", default=DB_NAME)
    parser.add_argument("--category", default=None)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--type", choices=("oneshot", "loop", "all"), default="oneshot")
    args = parser.parse_args()

    matches, ref = match_sample(
        args.reference, category=args.category, top_k=args.top,
        db_path=args.db, sample_type=args.type
    )
    print(f"\nReference: {ref['filename']} | role={ref['category']} "
          f"| confidence={ref['category_confidence']:.2f}")
    for rank, (score, cand, sims) in enumerate(matches, 1):
        print(
            f"{rank:2}. {score*100:5.1f}%  {cand['filename']}  "
            f"[{cand['kit_name']}]  {cand['category']}  "
            f"A:{sims['sim_attack']:.2f} B:{sims['sim_body']:.2f} "
            f"T:{sims['sim_tail']:.2f} D:{sims['sim_mel24']:.2f}"
        )
