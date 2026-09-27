"""
audio_features.py - V2 producer-oriented drum analysis.

The V2 feature model is deliberately deterministic and interpretable:
- transient / body / tail analysis instead of whole-file averaging
- acoustic drum-role classification for raw song slices
- 24-band Mel timbre plus explicit spectral-band profile
- kick / 808 pitch contour
- conservative percussive-focus processing for mixed/song excerpts
- one-shot / loop classification based on audio duration + path hints
"""

import json
import math
import os
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    from scipy import ndimage, signal
except ImportError:  # pragma: no cover
    ndimage = None
    signal = None


TARGET_SR = 44100
EPS = 1e-9
FEATURE_VERSION = 2

DRUM_CATEGORIES = ("kick", "808", "snare", "clap", "rim", "hat", "perc", "other")


def load_audio(file_path, target_sr=TARGET_SR):
    """Load an audio file, mono it, resample, remove DC and peak-normalize."""
    if sf is None:
        raise ImportError("soundfile is required to load audio files.")

    data, sr = sf.read(file_path, dtype="float32")
    data = np.asarray(data, dtype=np.float32)

    if data.ndim > 1:
        data = np.mean(data, axis=1)

    data = np.nan_to_num(data, copy=False)
    data = data - float(np.mean(data))

    if sr != target_sr:
        if signal is None:
            new_len = max(1, int(round(len(data) * target_sr / sr)))
            x_old = np.linspace(0.0, 1.0, len(data), endpoint=False)
            x_new = np.linspace(0.0, 1.0, new_len, endpoint=False)
            data = np.interp(x_new, x_old, data).astype(np.float32)
        else:
            new_len = max(1, int(round(len(data) * target_sr / sr)))
            data = signal.resample_poly(data, target_sr, sr).astype(np.float32)
            if len(data) > new_len:
                data = data[:new_len]
            elif len(data) < new_len:
                data = np.pad(data, (0, new_len - len(data)))
        sr = target_sr

    peak = float(np.max(np.abs(data))) if len(data) else 0.0
    if peak > 1e-7:
        data = data / peak

    return data.astype(np.float32), int(sr)


def _normalize_vector(values):
    values = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm < EPS:
        return np.zeros_like(values)
    return values / norm


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    keys = list(scores.keys())
    vals = np.asarray([scores[k] for k in keys], dtype=np.float64)
    vals -= np.max(vals)
    expv = np.exp(vals)
    total = float(np.sum(expv)) + EPS
    return {k: float(v / total) for k, v in zip(keys, expv)}


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    a = np.asarray(list(a), dtype=np.float32)
    b = np.asarray(list(b), dtype=np.float32)
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def detect_all_onsets(audio, sr, min_distance_ms=70.0, sensitivity=0.65):
    """
    Detect transients with a hybrid energy/spectral-flux onset curve.

    Returns slices sized for drum matching rather than arbitrary audio regions.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 512:
        return [(0, len(audio))]

    frame = 1024
    hop = 256
    if len(audio) < frame:
        frame = 512
        hop = 128

    if signal is None:
        env = np.abs(audio)
        smooth = np.convolve(env, np.ones(max(1, int(sr * 0.004))) /
                             max(1, int(sr * 0.004)), mode="same")
        strength = np.maximum(np.diff(smooth, prepend=smooth[0]), 0.0)
        threshold = np.median(strength) + sensitivity * (np.max(strength) - np.median(strength))
        distance = max(1, int(sr * min_distance_ms / 1000.0))
        peaks = []
        last = -distance
        for i in np.argsort(strength)[::-1]:
            if strength[i] < threshold:
                break
            if i - last >= distance:
                peaks.append(int(i))
                last = int(i)
        peaks.sort()
    else:
        f, t, z = signal.stft(
            audio,
            fs=sr,
            nperseg=frame,
            noverlap=frame - hop,
            boundary=None,
            padded=False,
        )
        mag = np.abs(z)
        if mag.shape[1] < 2:
            return [(0, len(audio))]

        energy = np.sqrt(np.mean(mag ** 2, axis=0) + EPS)
        energy = np.log1p(energy)
        flux = np.maximum(np.diff(mag, axis=1), 0.0).mean(axis=0)
        flux = np.pad(flux, (1, 0))
        curve = 0.62 * energy + 0.38 * (flux / (np.max(flux) + EPS))
        curve -= np.median(curve)

        med = float(np.median(curve))
        spread = float(np.median(np.abs(curve - med))) + EPS
        threshold = med + sensitivity * spread * 3.0

        distance_frames = max(1, int((sr * min_distance_ms / 1000.0) / hop))
        prominence = max(0.002, float(np.std(curve)) * 0.18)
        peaks_frames, _ = signal.find_peaks(
            curve,
            distance=distance_frames,
            prominence=prominence,
            height=threshold,
        )
        peaks = [int(p * hop) for p in peaks_frames]

    if not peaks:
        return [(0, min(len(audio), int(sr * 0.24)))]

    # Include an initial event when the source begins near full scale.
    if peaks[0] > int(sr * 0.05):
        initial_energy = float(np.sqrt(np.mean(audio[:int(sr * 0.03)] ** 2) + EPS))
        overall = float(np.sqrt(np.mean(audio ** 2) + EPS))
        if initial_energy > overall * 0.35:
            peaks.insert(0, 0)

    min_len = int(sr * 0.055)
    max_len = int(sr * 0.28)
    slices = []

    for idx, start in enumerate(peaks):
        start = max(0, min(start, len(audio) - 1))
        next_peak = peaks[idx + 1] if idx + 1 < len(peaks) else len(audio)
        end = min(len(audio), start + max_len, next_peak)
        if end - start < min_len:
            end = min(len(audio), start + min_len)
        if end > start:
            slices.append((start, end))

    return slices or [(0, min(len(audio), int(sr * 0.24)))]


def build_mel_filterbank(sr=TARGET_SR, n_fft=2048, n_mels=24,
                         f_min=20.0, f_max=16000.0):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(
        hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2
    )
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        if center <= left or right <= center:
            continue
        bank[m - 1, left:center] = (
            np.arange(left, center) - left
        ) / (center - left)
        bank[m - 1, center:right] = (
            right - np.arange(center, right)
        ) / (right - center)
    return bank


_MEL_CACHE = {}


def compute_mel24_spectrum(audio, sr):
    n_fft = 2048
    key = (sr, n_fft)
    if key not in _MEL_CACHE:
        _MEL_CACHE[key] = build_mel_filterbank(sr=sr, n_fft=n_fft)

    segment = np.asarray(audio[:min(len(audio), int(sr * 0.09))], dtype=np.float32)
    if len(segment) < 64:
        return [0.0] * 24

    window = np.hanning(len(segment))
    spectrum = np.abs(np.fft.rfft(segment * window, n=n_fft))
    power = spectrum ** 2
    mel = np.dot(_MEL_CACHE[key], power)
    log_mel = np.log10(mel + 1e-7)

    # Subtract the mean so cosine distance represents contour/shape,
    # not absolute level.
    log_mel -= np.mean(log_mel)
    return [round(float(x), 5) for x in _normalize_vector(log_mel)]


def _fft_band_energy(audio, sr, lo_hz, hi_hz):
    if len(audio) < 32:
        return 0.0
    n_fft = max(2048, 2 ** int(np.ceil(np.log2(min(len(audio), 8192)))))
    segment = audio[:min(len(audio), n_fft)]
    spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= lo_hz) & (freqs < hi_hz)
    return float(np.sum(spec[mask])) if np.any(mask) else 0.0


def compute_band_profile(audio, sr):
    bands = (
        (20, 60), (60, 120), (120, 250), (250, 500),
        (500, 2000), (2000, 6000), (6000, 16000)
    )
    energy = np.asarray([_fft_band_energy(audio, sr, a, b) for a, b in bands],
                        dtype=np.float64)
    total = float(np.sum(energy)) + EPS
    profile = energy / total
    return [round(float(x), 6) for x in profile]


def _spectral_stats(audio, sr):
    if len(audio) < 64:
        return 0.0, 0.0, 0.0

    n_fft = max(2048, 2 ** int(np.ceil(np.log2(min(len(audio), 8192)))))
    segment = audio[:min(len(audio), n_fft)]
    mag = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=n_fft))
    power = mag ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    denom = float(np.sum(mag)) + EPS
    centroid = float(np.sum(freqs * mag) / denom)

    cumulative = np.cumsum(mag)
    roll_target = cumulative[-1] * 0.85 if len(cumulative) else 0.0
    idx = int(np.searchsorted(cumulative, roll_target)) if len(cumulative) else 0
    rolloff = float(freqs[min(idx, len(freqs) - 1)]) if len(freqs) else 0.0

    geo = math.exp(float(np.mean(np.log(power + 1e-12))))
    arith = float(np.mean(power)) + EPS
    flatness = float(np.clip(geo / arith, 0.0, 1.0))

    return centroid, rolloff, flatness


def detect_onset(audio, sr):
    """Align the analysis window to the leading transient."""
    audio = np.asarray(audio, dtype=np.float32)
    if len(audio) < 128:
        return audio, 0, 0.0

    abs_audio = np.abs(audio)
    if float(np.max(abs_audio)) < 1e-6:
        return audio, 0, 0.0

    smooth_n = max(1, int(sr * 0.0025))
    kernel = np.ones(smooth_n, dtype=np.float32) / smooth_n
    envelope = np.convolve(abs_audio, kernel, mode="same")

    search_end = int(min(len(envelope), sr * 0.08))
    peak_idx = int(np.argmax(envelope[:search_end]))
    peak_val = float(envelope[peak_idx])

    threshold = max(peak_val * 0.04, 1e-5)
    before = np.where(envelope[:peak_idx + 1] <= threshold)[0]
    onset_idx = int(before[-1]) if len(before) else max(0, peak_idx - int(sr * 0.003))
    onset_idx = max(0, onset_idx - int(sr * 0.001))

    return audio[onset_idx:], onset_idx, onset_idx * 1000.0 / sr


def compute_log_attack_time(aligned_audio, sr):
    if len(aligned_audio) < 64:
        return 1.0, 0.0

    limit = min(len(aligned_audio), int(sr * 0.04))
    env = np.abs(aligned_audio[:limit])
    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx])

    if peak < 1e-6 or peak_idx == 0:
        return 0.5, -0.3

    ten = peak * 0.10
    ninety = peak * 0.90

    ten_candidates = np.where(env[:peak_idx + 1] <= ten)[0]
    ninety_candidates = np.where(env[:peak_idx + 1] >= ninety)[0]

    i10 = int(ten_candidates[-1]) if len(ten_candidates) else 0
    i90 = int(ninety_candidates[0]) if len(ninety_candidates) else peak_idx
    samples = max(1, i90 - i10)
    ms = samples * 1000.0 / sr
    return float(ms), float(np.log10(max(ms, 0.05)))


def compute_transient_punch(audio, sr, window_ms=20.0):
    length = min(len(audio), max(1, int(sr * window_ms / 1000.0)))
    chunk = np.asarray(audio[:length], dtype=np.float32)
    if len(chunk) == 0:
        return 0.0
    peak = float(np.max(np.abs(chunk)))
    rms = float(np.sqrt(np.mean(chunk ** 2) + EPS))
    return float(np.clip(20.0 * np.log10((peak + 1e-6) / (rms + 1e-6)), 0.0, 40.0))


def _dominant_frequency(audio, sr, lo, hi):
    if len(audio) < 64:
        return 0.0, 0.0

    n_fft = 8192
    segment = audio[:min(len(audio), int(sr * 0.16))]
    if len(segment) < 32:
        return 0.0, 0.0

    mag = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0, 0.0

    local = mag[mask]
    idx = int(np.argmax(local))
    peak = float(local[idx])
    conf = float(np.clip((peak / (np.mean(local) + EPS) - 1.0) / 7.0, 0.0, 1.0))
    return float(freqs[np.where(mask)[0][idx]]), conf


def compute_pitch_contour(aligned_audio, sr, category="other"):
    cat = category.lower()
    if cat in ("kick", "808"):
        f_top, _ = _dominant_frequency(aligned_audio[:max(1, int(sr * 0.035))],
                                       sr, 90.0, 420.0)
        start = int(sr * 0.035)
        end = min(len(aligned_audio), int(sr * 0.16))
        f_sub, conf = _dominant_frequency(aligned_audio[start:end], sr, 25.0, 120.0)
        if f_sub <= 0.0:
            f_sub = 55.0
        if f_top <= 0.0:
            f_top = f_sub
        drop = 12.0 * math.log2(max(f_top, f_sub) / max(f_sub, 20.0))
        return round(float(f_sub), 2), round(float(f_top), 2), round(float(max(drop, 0.0)), 2), round(float(conf), 3)

    if cat in ("snare", "clap", "rim"):
        f0, conf = _dominant_frequency(aligned_audio, sr, 120.0, 420.0)
        return round(float(f0 or 220.0), 2), round(float(f0 or 220.0), 2), 0.0, round(float(conf), 3)

    f0, conf = _dominant_frequency(aligned_audio, sr, 40.0, 900.0)
    return round(float(f0 or 200.0), 2), round(float(f0 or 200.0), 2), 0.0, round(float(conf), 3)


def _segment_rms(audio):
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2) + EPS))


def compute_envelope_features(audio, sr):
    length = len(audio)
    if length < 16:
        return {
            "attack_ratio": 1.0, "body_ratio": 0.0, "tail_ratio": 0.0,
            "transient_rms": 0.0, "body_rms": 0.0, "tail_rms": 0.0,
        }

    a_end = min(length, int(sr * 0.018))
    b_start = a_end
    b_end = min(length, int(sr * 0.10))
    t_start = b_end
    t_end = min(length, int(sr * 0.28))

    a = _segment_rms(audio[:a_end])
    b = _segment_rms(audio[b_start:b_end]) if b_end > b_start else a
    t = _segment_rms(audio[t_start:t_end]) if t_end > t_start else 0.0
    total = a + b + t + EPS

    return {
        "attack_ratio": float(a / total),
        "body_ratio": float(b / total),
        "tail_ratio": float(t / total),
        "transient_rms": a,
        "body_rms": b,
        "tail_rms": t,
    }


def compute_decay_time(audio, sr, threshold_db=-30.0):
    if len(audio) < 64:
        return 0.0

    env_n = max(1, int(sr * 0.004))
    env = np.convolve(np.abs(audio), np.ones(env_n) / env_n, mode="same")
    peak_idx = int(np.argmax(env))
    peak = float(env[peak_idx])

    if peak < 1e-7:
        return 0.0

    target = peak * (10.0 ** (threshold_db / 20.0))
    hits = np.where(env[peak_idx:] <= target)[0]
    decay_ms = (
        float(hits[0]) * 1000.0 / sr
        if len(hits)
        else float(len(env) - peak_idx) * 1000.0 / sr
    )
    return float(np.clip(decay_ms, 5.0, 3000.0))


def compute_noise_ratio(audio, sr):
    if len(audio) < 128:
        return 0.0
    _, _, flatness = _spectral_stats(audio, sr)
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
    return float(np.clip(0.65 * flatness + 0.35 * min(zcr * 4.0, 1.0), 0.0, 1.0))


def percussive_focus(audio, sr, strength=0.72):
    """
    Conservative HPSS-style focus. It is intentionally blended with the
    original signal so a useful drum body is not destroyed.
    """
    if signal is None or ndimage is None or len(audio) < 1024:
        return np.asarray(audio, dtype=np.float32)

    nperseg = 1024
    noverlap = 768
    f, t, z = signal.stft(
        audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
        boundary="zeros", padded=True
    )
    mag = np.abs(z)
    phase = np.angle(z)

    # Time-median favours harmonic/sustained content; frequency-median
    # favours vertically concentrated transients.
    harmonic = ndimage.median_filter(mag, size=(1, 9))
    percussive = ndimage.median_filter(mag, size=(9, 1))

    p2 = percussive ** 2
    h2 = harmonic ** 2
    mask = p2 / (p2 + h2 + EPS)

    focused = mag * (0.25 + 0.75 * np.clip(mask, 0.0, 1.0))
    _, restored = signal.istft(
        focused * np.exp(1j * phase),
        fs=sr, nperseg=nperseg, noverlap=noverlap,
        input_onesided=True, boundary=True
    )

    if len(restored) < len(audio):
        restored = np.pad(restored, (0, len(audio) - len(restored)))
    restored = restored[:len(audio)]
    peak = float(np.max(np.abs(restored))) if len(restored) else 0.0
    if peak > 1e-7:
        restored = restored / peak

    strength = float(np.clip(strength, 0.0, 1.0))
    return ((1.0 - strength) * audio + strength * restored).astype(np.float32)


def debleed_audio(audio, sr):
    """Backward-compatible alias for the safer percussive-focus processor."""
    return percussive_focus(np.asarray(audio, dtype=np.float32), sr, strength=0.70)


def detect_category_from_path(file_path):
    normalized = str(file_path).lower().replace("\\", "/")
    text = os.path.basename(normalized) + " " + os.path.dirname(normalized)

    if any(k in text for k in ("808", "subbass", "sub-bass", "sub bass")):
        return "808"
    if any(k in text for k in ("clap", "clp", "snap")):
        return "clap"
    if any(k in text for k in ("rim", "rimshot")):
        return "rim"
    if any(k in text for k in ("snare", "snr")):
        return "snare"
    if any(k in text for k in ("kick", "kik", "bassdrum", "bass drum", "/bd/", " bd ")):
        return "kick"
    if any(k in text for k in (
        "hihat", "hi-hat", "hi hat", "hat", "cymbal", "ride", "crash", "splash", "hh_"
    )):
        return "hat"
    if any(k in text for k in (
        "perc", "shaker", "conga", "bongo", "tom", "tambourine", "cowbell",
        "triangle", "guiro", "woodblock"
    )):
        return "perc"
    return "other"


def is_loop_sample(path, filename, duration_ms, category):
    text = (str(path) + " " + str(filename)).lower().replace("\\", "/")
    explicit = (
        "loop", "drumloop", "drum_loop", "stem", "break", "bpm ",
        "bpm_", "drum fill", "drum_fill"
    )
    if any(k in text for k in explicit):
        return 1

    seconds = float(duration_ms or 0.0) / 1000.0
    if category in ("kick", "808"):
        return int(seconds >= 4.5)
    return int(seconds >= 3.0)


def classify_drum_role(
    audio,
    sr,
    spectral_centroid,
    spectral_flatness,
    decay_ms,
    noise_ratio,
    band_profile,
    attack_ratio,
    body_ratio,
    tail_ratio,
):
    """
    Acoustic heuristic classifier used for song slices and as a second opinion
    to filename metadata. Scores are not probabilities of truth; they are
    relative role affinities used to gate and rank candidates.
    """
    b = np.asarray(band_profile, dtype=np.float64)
    b = b / (np.sum(b) + EPS)

    low = float(b[0] + b[1] + b[2])
    low_mid = float(b[2] + b[3])
    mid = float(b[3] + b[4])
    high = float(b[5] + b[6])
    very_high = float(b[6])

    short = math.exp(-((decay_ms - 90.0) / 120.0) ** 2)
    medium = math.exp(-((decay_ms - 280.0) / 260.0) ** 2)
    long_tail = 1.0 / (1.0 + math.exp(-(decay_ms - 650.0) / 180.0))

    low_centroid = max(0.0, 1.0 - spectral_centroid / 1800.0)
    high_centroid = min(1.0, spectral_centroid / 7000.0)

    scores = {
        "kick": (
            2.0 * low + 0.6 * low_mid + 0.5 * attack_ratio
            + 0.35 * (1.0 - noise_ratio) + 0.3 * low_centroid
            - 0.25 * very_high
        ),
        "808": (
            2.15 * low + 0.75 * low_mid + 0.6 * body_ratio
            + 0.6 * long_tail + 0.35 * low_centroid
            - 0.25 * attack_ratio
        ),
        "snare": (
            1.2 * high + 0.9 * mid + 0.75 * noise_ratio
            + 0.35 * medium + 0.35 * body_ratio
        ),
        "clap": (
            1.25 * high + 1.0 * noise_ratio + 0.7 * short
            + 0.25 * (1.0 - body_ratio)
        ),
        "rim": (
            1.0 * mid + 0.85 * attack_ratio + 0.55 * short
            + 0.25 * (1.0 - noise_ratio)
        ),
        "hat": (
            2.0 * high + 0.9 * very_high + 0.85 * high_centroid
            + 0.7 * short + 0.45 * noise_ratio
        ),
        "perc": (
            0.8 * mid + 0.55 * high + 0.55 * body_ratio
            + 0.25 * medium
        ),
        "other": 0.15 + 0.25 * spectral_flatness,
    }

    probs = _softmax(scores)
    top = max(probs, key=probs.get)
    confidence = float(probs[top])
    return top, confidence, probs


def extract_features(file_or_data, sr=TARGET_SR, category=None, apply_debleed=False):
    """Extract V2 features from a file path or a raw audio slice."""
    path = ""
    filename = "slice_audio.wav"

    if isinstance(file_or_data, str):
        if not os.path.exists(file_or_data):
            raise FileNotFoundError(file_or_data)
        data, sr = load_audio(file_or_data, target_sr=sr)
        path = os.path.abspath(file_or_data)
        filename = os.path.basename(file_or_data)
    else:
        data = np.asarray(file_or_data, dtype=np.float32)
        data = np.nan_to_num(data)
        if len(data) == 0:
            raise ValueError("Audio slice is empty.")
        data = data - np.mean(data)
        peak = float(np.max(np.abs(data)))
        if peak > 1e-7:
            data = data / peak

    if apply_debleed:
        data = debleed_audio(data, sr)

    raw_duration_ms = len(data) * 1000.0 / sr

    aligned, onset_idx, onset_ms = detect_onset(data, sr)
    attack_time_ms, lat = compute_log_attack_time(aligned, sr)
    crest_db = compute_transient_punch(aligned, sr)

    provisional_category = (
        category.lower() if category and category.lower() != "auto"
        else "other"
    )

    # First pass uses broad acoustic features independent of drum role.
    band_profile = compute_band_profile(aligned, sr)
    centroid, rolloff, flatness = _spectral_stats(aligned, sr)
    envelope = compute_envelope_features(aligned, sr)
    decay_ms = compute_decay_time(aligned, sr)
    noise_ratio = compute_noise_ratio(aligned, sr)

    detected_category, role_confidence, role_probs = classify_drum_role(
        aligned, sr, centroid, flatness, decay_ms, noise_ratio,
        band_profile, envelope["attack_ratio"], envelope["body_ratio"],
        envelope["tail_ratio"]
    )

    if category and category.lower() != "auto":
        selected_category = category.lower()
        # Preserve the user choice, but retain the acoustic distribution for
        # later candidate gating.
        role_probs[selected_category] = max(role_probs.get(selected_category, 0.0), 0.75)
        total = sum(role_probs.values()) + EPS
        role_probs = {k: float(v / total) for k, v in role_probs.items()}
        detected_category = selected_category

    f_sub, f_top, pitch_drop_st, f0_conf = compute_pitch_contour(
        aligned, sr, category=detected_category
    )
    mel24 = compute_mel24_spectrum(aligned, sr)

    rms = _segment_rms(aligned)
    rms_db = 20.0 * math.log10(rms + 1e-7)

    path_category = detect_category_from_path(path) if path else "other"
    if path and detected_category == "other" and path_category != "other":
        detected_category = path_category

    is_loop = is_loop_sample(path, filename, raw_duration_ms, detected_category)

    return {
        "feature_version": FEATURE_VERSION,
        "path": path,
        "filename": filename,
        "category": detected_category,
        "category_confidence": float(role_confidence),
        "category_probs_json": json.dumps(role_probs, separators=(",", ":")),
        "is_loop": int(is_loop),
        "onset_time_ms": round(float(onset_ms), 3),
        "attack_time_ms": round(float(attack_time_ms), 3),
        "lat": round(float(lat), 4),
        "crest_db": round(float(crest_db), 3),
        "rms_db": round(float(rms_db), 3),
        "f0": round(float(f_sub), 2),
        "f_sub": round(float(f_sub), 2),
        "f_top": round(float(f_top), 2),
        "pitch_drop_st": round(float(pitch_drop_st), 3),
        "f0_conf": round(float(f0_conf), 4),
        "att_low": round(float(band_profile[0] + band_profile[1]), 5),
        "att_mid": round(float(band_profile[2] + band_profile[3] + band_profile[4]), 5),
        "att_high": round(float(band_profile[5] + band_profile[6]), 5),
        "sus_low": round(float(band_profile[0] + band_profile[1] + band_profile[2]), 5),
        "sus_mid": round(float(band_profile[3] + band_profile[4]), 5),
        "sus_high": round(float(band_profile[5] + band_profile[6]), 5),
        "mel24_json": json.dumps(mel24, separators=(",", ":")),
        "band_profile_json": json.dumps(band_profile, separators=(",", ":")),
        "centroid": round(float(centroid), 3),
        "rolloff": round(float(rolloff), 3),
        "spectral_flatness": round(float(flatness), 6),
        "noise_ratio": round(float(noise_ratio), 6),
        "decay_ms": round(float(decay_ms), 3),
        "duration_ms": round(float(raw_duration_ms), 3),
        "attack_ratio": round(float(envelope["attack_ratio"]), 6),
        "body_ratio": round(float(envelope["body_ratio"]), 6),
        "tail_ratio": round(float(envelope["tail_ratio"]), 6),
        "transient_rms": round(float(envelope["transient_rms"]), 6),
        "body_rms": round(float(envelope["body_rms"]), 6),
        "tail_rms": round(float(envelope["tail_rms"]), 6),
        "path_category": path_category,
    }
