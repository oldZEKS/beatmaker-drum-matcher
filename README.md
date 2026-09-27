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

## Testing

    python -m pytest -q

## Important

V2 changes the feature schema. Existing databases are migrated automatically when the indexer starts. Re-indexing an old library with `--force` is recommended if you want all samples to receive the new V2 descriptors.