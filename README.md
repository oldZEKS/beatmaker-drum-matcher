# Beatmaker Drum Matcher

A local-first drum replacement search tool for beatmakers.

## V2 goal

> **Find the sample from my own library that I would actually use to remake this hit.**

It is not designed as a generic acoustic-similarity browser.

### V2 changes

- Acoustic drum-role classification for raw song slices (not filename-only classification).
- Soft role compatibility rather than brittle hard category matching.
- Explicit **Attack / Body / Tail / Timbre / Texture / Pitch** ranking dimensions.
- Category-specific weighting for kick/808, snare/clap/rim, hats and percussion.
- Conservative percussive focus for mixed/mastered song excerpts.
- Fixed one-shot/loop database schema and automatic migration of older `drums.db` files.
- Broader audio import support in the indexer.
- Transparent result meters showing why a candidate ranked highly.
- Existing local SQLite architecture retained so the library stays fast and portable.
- Regression tests for the V2 analysis and schema.

## Install

    pip install -r requirements.txt

## Index a library

    python indexer.py "D:\\Samples\\Drum Kits"

or use **+ Index Drum Kit** in the GUI.

## Run

    python gui.py

## How matching works

song / loop / hit → onset detection → selected hit → acoustic drum-role analysis → attack + body + tail + timbre + texture + pitch → role-aware candidate gate → coarse retrieval → producer-oriented detailed ranking → top replacement samples

The V2 engine intentionally keeps the deterministic ranking interpretable. A neural audio embedding/reranker can be added later without replacing these features.

## Whole Sample vs. Portion Comparison

You can toggle whether the engine analyzes the **entire sample** or an **isolated portion/slice**:
- **⛶ Whole Sample**: Analyzes the complete audio file with its full decay, body, and tail. Essential when matching 808s, sustained kicks, cymbals, or unchopped one-shots.
- **✂ Portion / Slice**: Analyzes an isolated transient hit or a custom dragged time region. Essential when sampling from full drum loops, song excerpts, or stems.
- **Waveform Drag**: Click and drag across any area of the waveform to define a custom time window.
- **Shortcut `W`**: Quickly toggle between Whole Sample and Portion mode.

### CLI Examples

    # Match the entire sample (full decay & tail)
    python matcher.py "C:\path\to\808.wav" --scope whole

    # Match an isolated hit from a drum loop by slice index
    python matcher.py "C:\path\to\drum_loop.wav" --slice 1

    # Match a specific custom time portion (e.g. 0.15s to 0.45s)
    python matcher.py "C:\path\to\song.wav" --portion 0.15 0.45

## GUI Shortcuts

- **`W`**: Toggle between Whole Sample and Portion mode
- **`Space`**: Audition active target (whole sample or selected portion)
- **`R`**: Audition full source audio
- **`Left` / `Right`**: Step between detected transient hits in portion mode
- **`Drag to DAW`**: Drag any matched candidate directly into your DAW (FL Studio, Ableton, Reaper)

## Testing

    python -m pytest -q

## Important

V2 changes the feature schema. Existing databases are migrated automatically when the indexer starts. Re-indexing an old library with `--force` is recommended if you want all samples to receive the new V2 descriptors.