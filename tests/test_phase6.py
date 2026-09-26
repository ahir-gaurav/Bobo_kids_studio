"""
tests/test_phase6.py — Phase 6 Music + SFX Validation
======================================================
Tests that:
  1. All DSP math functions produce valid audio arrays
  2. All 5 music moods generate without errors
  3. All SFX in the script generate without errors
  4. Music manifest is complete and correct
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.music_composer import (
    MusicComposer,
    SAMPLE_RATE,
    MUSIC_GENERATORS,
    SFX_REGISTRY,
    sine_wave,
    adsr_envelope,
    add_harmonics,
    note_to_wave,
    normalize,
    mix,
    NOTE_FREQ,
)
from config import cfg

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def check(label, ok, detail=""):
    sym = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    print(f"  {sym}  {BOLD}{label}{RESET}")
    if detail:
        print(f"         {YELLOW}{detail}{RESET}")
    return ok


def run_tests():
    results = []
    print()
    print("=" * 58)
    print("  BOBO KIDS STUDIO -- Phase 6 Music + SFX Tests")
    print("=" * 58)

    # ── 1. DSP fundamentals ─────────────────────────────────────────────────
    print(f"\n{BOLD}[1] DSP Math Functions{RESET}")

    wave = sine_wave(440, 1.0)
    results.append(check("sine_wave: correct length",    len(wave) == SAMPLE_RATE))
    results.append(check("sine_wave: amplitude in range", np.max(np.abs(wave)) <= 1.01))
    results.append(check("sine_wave: not all zeros",     np.any(wave != 0)))

    env = adsr_envelope(SAMPLE_RATE)
    results.append(check("adsr_envelope: correct length", len(env) == SAMPLE_RATE))
    results.append(check("adsr_envelope: starts at 0",    env[0] < 0.01))
    results.append(check("adsr_envelope: ends at 0",      env[-1] < 0.05))
    results.append(check("adsr_envelope: peak reaches 1", np.max(env) > 0.9))

    note = note_to_wave("C4", 0.5, "xylophone")
    results.append(check("note_to_wave: non-empty",         len(note) > 0))
    results.append(check("note_to_wave: normalized range",  np.max(np.abs(note)) <= 1.01))

    w1 = np.ones(100) * 0.5
    w2 = np.ones(100) * 0.3
    mixed = mix(w1, w2, weights=[0.6, 0.4])
    results.append(check("mix: correct length",            len(mixed) == 100))
    results.append(check("mix: weighted sum correct",      abs(mixed[0] - 0.6*0.5 - 0.4*0.3) < 0.001))

    normalized = normalize(np.array([0.1, 0.5, 0.2, -0.3]), target_peak=0.8)
    results.append(check("normalize: peak at target",      abs(np.max(np.abs(normalized)) - 0.8) < 0.01))

    # ── 2. Music tracks ─────────────────────────────────────────────────────
    print(f"\n{BOLD}[2] Music Generation (all 5 moods){RESET}")
    for mood, gen_func in MUSIC_GENERATORS.items():
        wave = gen_func()
        ok = (
            isinstance(wave, np.ndarray) and
            len(wave) > SAMPLE_RATE * 2 and     # at least 2 seconds
            np.max(np.abs(wave)) > 0.01          # not silence
        )
        results.append(check(
            f"Mood '{mood}': {len(wave)/SAMPLE_RATE:.1f}s, peak={np.max(np.abs(wave)):.2f}",
            ok
        ))

    # ── 3. SFX ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[3] SFX Generation (all {len(SFX_REGISTRY)} effects){RESET}")
    for sfx_name, gen_func in SFX_REGISTRY.items():
        wave = gen_func()
        ok = (
            isinstance(wave, np.ndarray) and
            len(wave) > 100 and
            np.max(np.abs(wave)) > 0.01
        )
        results.append(check(
            f"SFX '{sfx_name}': {len(wave)/SAMPLE_RATE:.2f}s",
            ok
        ))

    # ── 4. Full compose pipeline ─────────────────────────────────────────────
    print(f"\n{BOLD}[4] Full Compose (reads script.json -> generates WAVs){RESET}")
    script_path = cfg.PIPELINE_DIRS["scripts"] / "20260924_learn_colors_red" / "script.json"
    results.append(check("Script file exists", script_path.exists()))

    if script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        composer = MusicComposer()
        manifest = composer.compose(script, "20260924_learn_colors_red")

        results.append(check("Manifest has music_files",  "music_files"  in manifest))
        results.append(check("Manifest has sfx_files",    "sfx_files"    in manifest))
        results.append(check("Manifest has scenes",        "scenes"       in manifest))
        results.append(check(f"All 8 scenes in manifest", len(manifest["scenes"]) == 8))

        # Verify WAV files exist on disk
        for mood, path in manifest["music_files"].items():
            results.append(check(f"Music WAV exists: {mood}", Path(path).exists()))
        for sfx, path in manifest["sfx_files"].items():
            results.append(check(f"SFX WAV exists: {sfx}", Path(path).exists()))

        # Verify manifest saved to audio dir
        manifest_path = cfg.PIPELINE_DIRS["audio"] / "20260924_learn_colors_red" / "music_manifest.json"
        results.append(check("music_manifest.json saved", manifest_path.exists()))

        # Check scene 6 (celebration) has correct mood
        scene6 = next(s for s in manifest["scenes"] if s["scene_number"] == 6)
        results.append(check("Scene 6 mood is 'celebration'",
                             scene6["bg_music_mood"] == "celebration"))

    # ── Results ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print()
    print("=" * 58)
    color = GREEN if passed == total else YELLOW
    print(f"  {color}Result: {passed}/{total} checks passed{RESET}")
    if passed == total:
        print(f"  {GREEN}Phase 6 complete — all music + SFX ready!{RESET}")
    print("=" * 58)
    print()
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
