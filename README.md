# Beatmaker Drum Matcher 🥁🎯

> **A high-precision, acoustic sample-matching and drum replacement engine designed specifically for beatmakers, producers, and track recreation.**

Unlike generic Music Information Retrieval (MIR) browsers (like Sononym or COSMOS) that average whole waveforms and surface unwanted room bleed or loops, **Beatmaker Drum Matcher** is built around drum physics and producer workflow.

---

## ⚡ Key Features

* **In-Loop Transient Slicer**: Drag in full 4-bar drum loops, stems, or song sections. The app automatically segments all transient onsets. Just click any hit (kick, snare, clap, hat) directly on the waveform to match it instantly.
* **24-Band Mel Micro-Timbre**: 24 triangular perceptual Mel-scale filters ($20\text{ Hz}\text{–}16\text{ kHz}$) capture nuanced boxiness, scooped mids, beater click, and top-end air.
* **One-Shot vs. Loop Segregation**: Zero loops polluting your snare and kick searches. Dedicated filters for `[ One-Shots Only ]`, `[ Loops Only ]`, and `[ All ]`.
* **Kick & 808 Pitch Contours**: Tracks the exponential beater impact dive ($f_{\text{click}} \to f_{\text{sub}}$) and computes sweep depth in semitones (e.g. $-20.3\text{ st}$).
* **Sub-Millisecond Onset Alignment**: Automatically trims pre-delay and air from amateur samples or song rips so temporal metrics are never diluted.
* **In-Key Musical Tuning Hints**: Calculates exact semitone and musical cent offsets (e.g. `tune +1.2 st` or `IN-KEY`) so you know how to pitch the sample in your DAW.
* **Direct DAW Drag-and-Drop Out**: Grab the `⠿ DRAG TO DAW` handle on any candidate and drop it directly into **FL Studio’s Channel Rack/Playlist**, **Ableton Live’s Drum Rack**, or **Bitwig/Reaper**.
* **✨ De-Bleeding Spectral Gate**: One-click attenuation of room reverb and hi-hat bleed when working with samples ripped directly from finished songs.
* **Blazing Fast Indexer**: Multi-threaded library indexer capable of analyzing **200+ samples per second** (10,000 samples indexed in ~45 seconds).

---

## 🛠️ Tech Stack

* **Audio Analysis**: `scipy.signal`, `numpy`, `soundfile` (Supports 16-bit, 24-bit PCM, and 32-bit IEEE Float FL Studio WAVs).
* **Storage**: Local `SQLite` database with indexing across frequency, crest factor, and categories.
* **Desktop GUI**: `PySide6` (Qt 6) with dark-mode aesthetic inspired by FL Studio / Ableton / FabFilter.
* **Audio Playback**: Zero-latency native asynchronous playback.

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/oldZEKS/beatmaker-drum-matcher.git
cd beatmaker-drum-matcher
pip install -r requirements.txt
```

### 2. Index Your Drum Libraries
Scan one or more drum kit folders into your local database:
```bash
python indexer.py "C:\Path\To\Your\Drum\Kits"
```
*Supports recursive scanning, automatic category classification, and multi-threading.*

### 3. Launch Desktop GUI
```bash
python gui.py
```
*(Or double-click `run_gui.bat` on Windows).*

### 4. CLI Matching
You can also run matches directly from the terminal:
```bash
# Match a snare one-shot against your library
python matcher.py "C:\path\to\reference_snare.wav" --type oneshot

# Search for tighter decay or punchier attack
python matcher.py "C:\path\to\reference_kick.wav" --modifier punchier

# Apply spectral de-bleeding on a hit sliced from a song
python matcher.py "C:\path\to\song_slice.wav" --debleed
```

---

## ⌨️ Keyboard Shortcuts (GUI)

| Key | Action |
| :--- | :--- |
| **`1` – `9`** | Instantly audition candidate matches #1 through #9 |
| **`Space`** | Audition the selected drum hit / slice |
| **`R`** | Replay the full reference loop or song section |
| **`Left` / `Right`** | Step through transient slices in the loaded loop |

---

## 📄 License
MIT License. Free for all producers and developers.
