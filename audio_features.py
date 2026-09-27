"""
audio_features.py - Advanced High-Precision Drum Acoustic Descriptors.
Includes:
1. Multi-Onset Transient Detection & In-Loop Slicer
2. 24-Band Mel-Scale Perceptual Micro-Timbre Filterbank
3. Sub-millisecond Onset Detection & Silence Trimming
4. Log-Attack Time (LAT) and Transient Rise-Slope
5. Kick/808 Dynamic Pitch Contour (f_click, f_sub, pitch_drop_st)
6. Tri-Band Energy Matrix (Low/Mid/High across Attack vs Sustain phases)
7. Spectral Centroid, Noise-to-Tone Harmonicity, T30 Decay
8. Spectral De-Bleeding Filter for Song Slices
"""

import math
import os
import json
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy import signal
except ImportError:
    signal = None


def load_audio(file_path, target_sr=44100):
    """
    Loads an audio file (supports WAV, FLAC, AIFF, 16/24/32-bit float).
    Converts stereo to mono and normalizes.
    """
    if sf is None:
        raise ImportError("soundfile library is required to load audio files.")

    data, sr = sf.read(file_path, dtype='float32')
    
    # Convert stereo to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample if necessary
    if sr != target_sr:
        if signal is not None:
            num_samples = int(len(data) * target_sr / sr)
            data = signal.resample(data, num_samples)
            sr = target_sr
        else:
            indices = np.linspace(0, len(data) - 1, int(len(data) * target_sr / sr))
            data = np.interp(indices, np.arange(len(data)), data)
            sr = target_sr

    # Remove DC offset
    data = data - np.mean(data)

    # Normalize peak to 1.0 (if not silence)
    peak = np.max(np.abs(data))
    if peak > 1e-6:
        data = data / peak

    return data, sr


# --- Multi-Onset Transient Slicer for Loops & Songs ---

def detect_all_onsets(audio, sr, min_distance_ms=100.0, sensitivity=0.15):
    """
    Detects drum hit transient onsets across an entire loop or song section.
    Ensures each hit has a natural 220ms one-shot window for accurate drum matching.
    Returns: list of (start_sample, end_sample) slice regions.
    """
    if len(audio) < 256:
        return [(0, len(audio))]

    abs_audio = np.abs(audio)
    win_size = max(1, int(sr * 0.005))
    kernel = np.ones(win_size) / win_size
    env = np.convolve(abs_audio, kernel, mode='same')

    # Positive onset strength derivative
    onset_env = np.diff(env)
    onset_env = np.maximum(onset_env, 0)

    max_slope = np.max(onset_env)
    if max_slope < 1e-4:
        return [(0, len(audio))]

    threshold = sensitivity * max_slope
    min_dist_samples = int(sr * (min_distance_ms / 1000.0))

    peaks = []
    last_peak = -min_dist_samples

    for i in range(1, len(onset_env) - 1):
        if onset_env[i] > threshold and onset_env[i] > onset_env[i - 1] and onset_env[i] >= onset_env[i + 1]:
            if (i - last_peak) >= min_dist_samples:
                backtrack = max(0, i - int(sr * 0.002))
                peaks.append(backtrack)
                last_peak = i

    if not peaks or peaks[0] > int(sr * 0.05):
        peaks.insert(0, 0)

    # For each onset, allow a natural one-shot drum decay duration (~220ms)
    slices = []
    default_slice_len = int(sr * 0.24)
    for idx, start in enumerate(peaks):
        if idx < len(peaks) - 1:
            raw_end = peaks[idx + 1]
            end = max(raw_end, min(len(audio), start + default_slice_len))
        else:
            end = min(len(audio), start + default_slice_len)

        if (end - start) > int(sr * 0.02):
            slices.append((start, end))

    if not slices:
        slices = [(0, len(audio))]

    return slices



# --- 24-Band Mel-Scale Perceptual Micro-Timbre ---

_MEL_FILTERBANK_CACHE = {}

def build_mel_filterbank(sr=44100, n_fft=2048, n_mels=24, f_min=20.0, f_max=16000.0):
    cache_key = (sr, n_fft, n_mels, f_min, f_max)
    if cache_key in _MEL_FILTERBANK_CACHE:
        return _MEL_FILTERBANK_CACHE[cache_key]

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fbank = np.zeros((n_mels, n_bins), dtype=np.float32)

    for m in range(1, n_mels + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]

        for k in range(f_left, f_center):
            if f_center > f_left:
                fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                fbank[m - 1, k] = (f_right - k) / (f_right - f_center)

    _MEL_FILTERBANK_CACHE[cache_key] = fbank
    return fbank


def compute_mel24_spectrum(audio, sr):
    """
    Computes a 24-band log Mel-filterbank energy vector for fine-grained timbre matching.
    Returns: list of 24 normalized floats.
    """
    n_fft = 2048
    fbank = build_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=24)

    # Analyze first 80ms of transient
    max_samples = min(len(audio), int(sr * 0.08))
    segment = audio[:max_samples]

    if len(segment) < 64:
        return [0.0] * 24

    window = np.hanning(len(segment))
    spec = np.abs(np.fft.rfft(segment * window, n=n_fft))
    power = spec ** 2

    # Mel energies
    mel_energies = np.dot(fbank, power)
    log_mel = np.log10(mel_energies + 1e-6)

    # L2 normalize
    norm = np.linalg.norm(log_mel) + 1e-9
    norm_mel = log_mel / norm

    return [round(float(x), 4) for x in norm_mel]


# --- Onset & Acoustic Feature Descriptors ---

def detect_onset(audio, sr):
    """
    Finds the exact sub-millisecond start of the transient impact.
    Trims leading pre-delay, air, or silence.
    Returns: aligned_audio, onset_idx, onset_time_ms
    """
    if len(audio) < 128:
        return audio, 0, 0.0

    abs_audio = np.abs(audio)
    peak_idx = np.argmax(abs_audio)
    peak_val = abs_audio[peak_idx]

    if peak_val < 1e-5:
        return audio, 0, 0.0

    threshold = 0.03 * peak_val
    sub_slice = abs_audio[:peak_idx + 1]
    
    below_thresh = np.where(sub_slice <= threshold)[0]
    if len(below_thresh) > 0:
        onset_idx = below_thresh[-1]
    else:
        onset_idx = 0

    margin = int(sr * 0.001)
    onset_idx = max(0, onset_idx - margin)
    onset_time_ms = (onset_idx / sr) * 1000.0

    aligned_audio = audio[onset_idx:]
    return aligned_audio, onset_idx, onset_time_ms


def compute_log_attack_time(aligned_audio, sr):
    """
    Measures the rise time from 10% to 90% of peak amplitude in the initial transient.
    Returns: attack_time_ms, log_attack_time (LAT)
    """
    if len(aligned_audio) < 64:
        return 1.0, 0.0

    max_attack_samples = min(len(aligned_audio), int(sr * 0.03))
    attack_chunk = np.abs(aligned_audio[:max_attack_samples])

    peak_idx = np.argmax(attack_chunk)
    peak_val = attack_chunk[peak_idx]

    if peak_val < 1e-5 or peak_idx == 0:
        return 0.5, math.log10(0.5)

    val_10 = 0.10 * peak_val
    val_90 = 0.90 * peak_val

    before_peak = attack_chunk[:peak_idx + 1]
    idx_10_candidates = np.where(before_peak <= val_10)[0]
    idx_10 = idx_10_candidates[-1] if len(idx_10_candidates) > 0 else 0

    idx_90_candidates = np.where(before_peak >= val_90)[0]
    idx_90 = idx_90_candidates[0] if len(idx_90_candidates) > 0 else peak_idx

    samples_diff = max(1, idx_90 - idx_10)
    attack_time_ms = (samples_diff / sr) * 1000.0
    lat = math.log10(max(attack_time_ms, 0.05))

    return round(float(attack_time_ms), 3), round(float(lat), 3)


def compute_transient_punch(audio, sr, window_ms=20.0):
    num_samples = int(sr * (window_ms / 1000.0))
    chunk = audio[:num_samples] if len(audio) >= num_samples else audio

    if len(chunk) == 0:
        return 0.0

    peak = np.max(np.abs(chunk))
    rms = np.sqrt(np.mean(chunk**2))

    if rms < 1e-7:
        return 0.0

    crest_db = 20.0 * np.log10((peak + 1e-6) / (rms + 1e-6))
    return float(np.clip(crest_db, 0.0, 40.0))


def compute_pitch_contour(aligned_audio, sr, category='other'):
    if len(aligned_audio) < 256:
        return 0.0, 0.0, 0.0, 0.0

    cat = category.lower()
    n_fft = 8192

    if cat in ('kick', '808'):
        win1_len = min(len(aligned_audio), int(sr * 0.025))
        seg1 = aligned_audio[:win1_len]
        spec1 = np.abs(np.fft.rfft(seg1 * np.hanning(len(seg1)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        idx1_min = np.searchsorted(freqs, 100.0)
        idx1_max = np.searchsorted(freqs, 350.0)
        if idx1_max > idx1_min:
            f_top = freqs[idx1_min + np.argmax(spec1[idx1_min:idx1_max])]
        else:
            f_top = 180.0

        start2 = int(sr * 0.035)
        end2 = min(len(aligned_audio), int(sr * 0.14))
        if end2 > start2 + 128:
            seg2 = aligned_audio[start2:end2]
            spec2 = np.abs(np.fft.rfft(seg2 * np.hanning(len(seg2)), n=n_fft))
            idx2_min = np.searchsorted(freqs, 28.0)
            idx2_max = np.searchsorted(freqs, 110.0)
            if idx2_max > idx2_min:
                f_sub = freqs[idx2_min + np.argmax(spec2[idx2_min:idx2_max])]
                conf = float(np.max(spec2[idx2_min:idx2_max]) / (np.mean(spec2[idx2_min:idx2_max]) + 1e-9))
                conf = float(np.clip((conf - 1.0) / 8.0, 0.0, 1.0))
            else:
                f_sub = 55.0
                conf = 0.5
        else:
            f_sub = 55.0
            conf = 0.5

        if f_top > f_sub and f_sub > 20:
            pitch_drop_st = 12.0 * math.log2(f_top / f_sub)
        else:
            f_top = f_sub
            pitch_drop_st = 0.0

        return round(float(f_sub), 2), round(float(f_top), 2), round(float(pitch_drop_st), 1), round(conf, 2)

    elif cat in ('snare', 'clap', 'rim'):
        win_len = min(len(aligned_audio), int(sr * 0.12))
        seg = aligned_audio[:win_len]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        idx_min = np.searchsorted(freqs, 130.0)
        idx_max = np.searchsorted(freqs, 360.0)
        if idx_max > idx_min:
            peak_sub_idx = np.argmax(spec[idx_min:idx_max])
            f0 = freqs[idx_min + peak_sub_idx]
            conf = float(np.max(spec[idx_min:idx_max]) / (np.mean(spec[idx_min:idx_max]) + 1e-9))
            conf = float(np.clip((conf - 1.0) / 8.0, 0.0, 1.0))
        else:
            f0 = 220.0
            conf = 0.0

        return round(float(f0), 2), round(float(f0), 2), 0.0, round(conf, 2)

    else:
        win_len = min(len(aligned_audio), int(sr * 0.15))
        seg = aligned_audio[:win_len]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        idx_min = np.searchsorted(freqs, 40.0)
        idx_max = np.searchsorted(freqs, 800.0)
        if idx_max > idx_min:
            f0 = freqs[idx_min + np.argmax(spec[idx_min:idx_max])]
            conf = 0.5
        else:
            f0 = 200.0
            conf = 0.0

        return round(float(f0), 2), round(float(f0), 2), 0.0, round(conf, 2)


def compute_triband_energy_matrix(aligned_audio, sr):
    if len(aligned_audio) < 256 or signal is None:
        return {
            'att_low': 0.33, 'att_mid': 0.33, 'att_high': 0.34,
            'sus_low': 0.33, 'sus_mid': 0.33, 'sus_high': 0.34
        }

    nyq = sr * 0.5
    f_low_cutoff = min(260.0 / nyq, 0.99)
    f_mid_cutoff = min(2800.0 / nyq, 0.99)

    b_low, a_low = signal.butter(2, f_low_cutoff, btype='lowpass')
    b_mid, a_mid = signal.butter(2, [f_low_cutoff, f_mid_cutoff], btype='bandpass')
    b_high, a_high = signal.butter(2, f_mid_cutoff, btype='highpass')

    y_low = signal.lfilter(b_low, a_low, aligned_audio)
    y_mid = signal.lfilter(b_mid, a_mid, aligned_audio)
    y_high = signal.lfilter(b_high, a_high, aligned_audio)

    att_samples = min(len(aligned_audio), int(sr * 0.018))
    sus_end_samples = min(len(aligned_audio), int(sr * 0.12))

    e_att_l = np.sum(y_low[:att_samples]**2) + 1e-9
    e_att_m = np.sum(y_mid[:att_samples]**2) + 1e-9
    e_att_h = np.sum(y_high[:att_samples]**2) + 1e-9
    total_att = e_att_l + e_att_m + e_att_h

    att_low = round(float(e_att_l / total_att), 3)
    att_mid = round(float(e_att_m / total_att), 3)
    att_high = round(float(e_att_h / total_att), 3)

    if sus_end_samples > att_samples:
        e_sus_l = np.sum(y_low[att_samples:sus_end_samples]**2) + 1e-9
        e_sus_m = np.sum(y_mid[att_samples:sus_end_samples]**2) + 1e-9
        e_sus_h = np.sum(y_high[att_samples:sus_end_samples]**2) + 1e-9
        total_sus = e_sus_l + e_sus_m + e_sus_h
        sus_low = round(float(e_sus_l / total_sus), 3)
        sus_mid = round(float(e_sus_m / total_sus), 3)
        sus_high = round(float(e_sus_h / total_sus), 3)
    else:
        sus_low, sus_mid, sus_high = att_low, att_mid, att_high

    return {
        'att_low': att_low, 'att_mid': att_mid, 'att_high': att_high,
        'sus_low': sus_low, 'sus_mid': sus_mid, 'sus_high': sus_high
    }


def compute_spectral_centroid(audio, sr):
    if len(audio) < 128:
        return 0.0

    window = np.hanning(len(audio))
    spec = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)

    sum_spec = np.sum(spec)
    if sum_spec < 1e-9:
        return 0.0

    centroid = np.sum(freqs * spec) / sum_spec
    return float(centroid)


def compute_noise_ratio(audio, sr):
    if len(audio) < 128:
        return 0.0

    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0

    window = np.hanning(len(audio))
    spec_mag = np.abs(np.fft.rfft(audio * window)) + 1e-12
    power = spec_mag ** 2
    geo_mean = np.exp(np.mean(np.log(power)))
    arith_mean = np.mean(power)
    flatness = geo_mean / (arith_mean + 1e-12)

    noise_score = 0.6 * flatness + 0.4 * min(zcr * 4.0, 1.0)
    return float(np.clip(noise_score, 0.0, 1.0))


def compute_decay_time(audio, sr, threshold_db=-30.0):
    if len(audio) < 64:
        return 0.0

    abs_audio = np.abs(audio)
    win_size = max(1, int(sr * 0.005))
    kernel = np.ones(win_size) / win_size
    env = np.convolve(abs_audio, kernel, mode='same')

    peak_idx = np.argmax(env)
    peak_val = env[peak_idx]

    if peak_val < 1e-6:
        return 0.0

    cutoff_val = peak_val * (10.0 ** (threshold_db / 20.0))

    decay_samples = np.where(env[peak_idx:] <= cutoff_val)[0]
    if len(decay_samples) > 0:
        decay_time_ms = (decay_samples[0] / sr) * 1000.0
    else:
        decay_time_ms = ((len(env) - peak_idx) / sr) * 1000.0

    return float(np.clip(decay_time_ms, 5.0, 3000.0))


def debleed_audio(audio, sr):
    if len(audio) < 512 or signal is None:
        return audio

    nperseg = 512
    noverlap = 384
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    frame_energies = np.sum(mag**2, axis=0)
    quiet_cutoff = np.percentile(frame_energies, 20)
    noise_frames = mag[:, frame_energies <= quiet_cutoff]
    
    if noise_frames.shape[1] > 0:
        noise_profile = np.mean(noise_frames, axis=1, keepdims=True)
    else:
        noise_profile = np.min(mag, axis=1, keepdims=True)

    sub_mag = np.maximum(mag - 1.5 * noise_profile, 0.05 * mag)
    _, cleaned = signal.istft(sub_mag * np.exp(1j * phase), fs=sr, nperseg=nperseg, noverlap=noverlap)

    if len(cleaned) > len(audio):
        cleaned = cleaned[:len(audio)]
    elif len(cleaned) < len(audio):
        cleaned = np.pad(cleaned, (0, len(audio) - len(cleaned)))

    peak = np.max(np.abs(cleaned))
    if peak > 1e-6:
        cleaned = cleaned / peak

    return cleaned


def detect_category_from_path(file_path):
    normalized = file_path.lower().replace('\\', '/')
    name_and_parent = os.path.basename(normalized) + " " + os.path.dirname(normalized)

    # 1. 808 / Bass
    if any(k in name_and_parent for k in ['808', 'sub bass', 'sub-bass', 'subbass', 'bass']):
        return '808'
    # 2. Claps & Snaps
    if any(k in name_and_parent for k in ['clap', 'clp', 'snap']):
        return 'clap'
    # 3. Rims
    if any(k in name_and_parent for k in ['rim', 'rimshot']):
        return 'rim'
    # 4. Snares
    if any(k in name_and_parent for k in ['snare', 'snr']):
        return 'snare'
    # 5. Kicks
    if any(k in name_and_parent for k in ['kick', 'kik', 'bd', 'bassdrum']):
        return 'kick'
    # 6. Hats & Cymbals
    if any(k in name_and_parent for k in ['hat', 'cymbal', 'hihat', 'open hat', 'closed hat', 'hh', 'ride', 'crash', 'splash']):
        return 'hat'
    # 7. Percs & Shakers
    if any(k in name_and_parent for k in ['perc', 'shaker', 'conga', 'bongo', 'tom', 'tambourine', 'cowbell', 'triangle', 'guiro', 'woodblock']):
        return 'perc'
    # 8. Vox / Chants
    if any(k in name_and_parent for k in ['vox', 'chant', 'vocal']):
        return 'vox'
    # 9. FX / Risers
    if any(k in name_and_parent for k in ['fx', 'riser', 'fall', 'impact', 'sfx', 'downlifter', 'uplifter']):
        return 'fx'
    # 10. Loops & Melodies
    if any(k in name_and_parent for k in ['loop', 'melody', 'sample', 'chords', 'flp']):
        return 'loop'

    return 'other'



def is_loop_sample(path, filename, duration_ms, category):
    p_lower = (str(path) + ' ' + str(filename)).lower().replace('\\', '/')
    has_loop_word = any(k in p_lower for k in [
        'loop', 'drumloop', 'melody', 'stem', 'bpm', 'drum fill', 'fill '
    ])
    dur_s = duration_ms / 1000.0 if duration_ms else 0.0
    if category == '808':
        if has_loop_word and ('bpm' in p_lower or 'loop' in p_lower):
            return 1
        return 1 if dur_s > 4.5 else 0
    else:
        if has_loop_word or dur_s > 2.2:
            return 1
        return 0


def extract_features(file_or_data, sr=44100, category=None, apply_debleed=False):
    """
    Extracts high-precision drum descriptors from a file path OR raw audio array.
    """
    if isinstance(file_or_data, str):
        data, sr = load_audio(file_or_data, target_sr=sr)
        filename = os.path.basename(file_or_data)
        path = os.path.abspath(file_or_data)
        if category is None or category == 'auto':
            category = detect_category_from_path(file_or_data)
    else:
        data = np.asarray(file_or_data, dtype=np.float32)
        filename = "slice_audio.wav"
        path = ""
        if category is None:
            category = 'other'

    if apply_debleed:
        data = debleed_audio(data, sr)

    # 1. Sub-millisecond onset alignment
    aligned_data, onset_idx, onset_time_ms = detect_onset(data, sr)

    # 2. Attack dynamics
    attack_time_ms, lat = compute_log_attack_time(aligned_data, sr)
    crest_db = compute_transient_punch(aligned_data, sr, window_ms=20.0)

    # 3. Dynamic pitch contour
    f_sub, f_top, pitch_drop_st, f0_conf = compute_pitch_contour(aligned_data, sr, category=category)

    # 4. Tri-Band Energy Matrix
    bands = compute_triband_energy_matrix(aligned_data, sr)

    # 5. 24-Band Mel-Scale Micro-Timbre
    mel24 = compute_mel24_spectrum(aligned_data, sr)

    # 6. Global descriptors
    centroid = compute_spectral_centroid(aligned_data, sr)
    noise_ratio = compute_noise_ratio(aligned_data, sr)
    decay_ms = compute_decay_time(aligned_data, sr, threshold_db=-30.0)
    duration_ms = (len(data) / sr) * 1000.0
    loop_flag = is_loop_sample(path, filename, duration_ms, category)

    return {
        'path': path,
        'filename': filename,
        'category': category,
        'is_loop': loop_flag,
        'onset_time_ms': round(onset_time_ms, 2),
        'attack_time_ms': attack_time_ms,
        'lat': lat,
        'crest_db': round(crest_db, 2),
        'f0': f_sub,
        'f_sub': f_sub,
        'f_top': f_top,
        'pitch_drop_st': pitch_drop_st,
        'f0_conf': f0_conf,
        'att_low': bands['att_low'],
        'att_mid': bands['att_mid'],
        'att_high': bands['att_high'],
        'sus_low': bands['sus_low'],
        'sus_mid': bands['sus_mid'],
        'sus_high': bands['sus_high'],
        'mel24': mel24,
        'mel24_json': json.dumps(mel24),
        'centroid': round(centroid, 1),
        'noise_ratio': round(noise_ratio, 3),
        'decay_ms': round(decay_ms, 1),
        'duration_ms': round(duration_ms, 1)
    }

