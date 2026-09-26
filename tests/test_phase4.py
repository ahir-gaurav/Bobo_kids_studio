"""
tests/test_phase4.py — Phase 4 Voice Synthesis Validation
=========================================================
Tests the voice synthesiser without needing a real TTS engine.
Uses a mock engine to verify the orchestration logic.
Then optionally tests with the real engine if available.
"""

import json
import sys
import wave
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.voice_synthesizer import (
    VoiceConfig,
    BaseTTSEngine,
    VoiceSynthesizer,
    SAPIEngine,
    KokoroEngine,
    get_best_engine,
)
from config import cfg

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


# ── Mock TTS engine for testing without models ────────────────────────────────

class MockTTSEngine(BaseTTSEngine):
    """
    A fake TTS engine that writes silent WAV files.

    WHY MOCK?
      Tests should be fast and deterministic.
      Real TTS takes 0.5–5 seconds per clip.
      The mock writes a 1-second silent WAV instantly.
      This lets us test the ORCHESTRATION logic (scene loop, file naming,
      manifest writing) without the TTS inference cost.

      This is a standard testing technique: mock the slow/external parts,
      test the logic around them.
    """
    SAMPLE_RATE = 22050

    def synthesize(self, text: str, config: VoiceConfig, output_path: Path) -> float:
        """Write a 1-second silent WAV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # WAV PCM 16-bit, 1 second of silence
        n_samples = self.SAMPLE_RATE
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)           # 2 bytes = 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
        return 1.0  # 1 second


def check(label: str, ok: bool, detail: str = "") -> bool:
    sym = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    print(f"  {sym}  {BOLD}{label}{RESET}")
    if detail:
        print(f"         {detail}")
    return ok


def run_tests():
    results = []
    print()
    print("=" * 58)
    print("  BOBO KIDS STUDIO -- Phase 4 Voice Synthesis Tests")
    print("=" * 58)

    # ── 1. VoiceConfig tests ─────────────────────────────────────────────────
    print(f"\n{BOLD}[1] VoiceConfig{RESET}")

    narr = VoiceConfig.narrator()
    results.append(check("Narrator config speed=0.9", narr.speed == 0.9))
    results.append(check("Narrator config pitch=1", narr.pitch_semitones == 1))

    from pipeline.character_loader import CharacterLoader
    loader = CharacterLoader()
    bobo = loader.get("bobo")
    lumi = loader.get("lumi")
    pip_char = loader.get("pip")

    results.append(check("Bobo character loaded", bobo is not None))
    results.append(check("Lumi character loaded", lumi is not None))
    results.append(check("Pip character loaded", pip_char is not None))

    if bobo:
        bobo_cfg = VoiceConfig.from_character(bobo)
        results.append(check("Bobo voice speed=0.9", bobo_cfg.speed == 0.9, f"got {bobo_cfg.speed}"))
        results.append(check("Bobo pitch=+2st", bobo_cfg.pitch_semitones == 2, f"got {bobo_cfg.pitch_semitones}"))

    if lumi:
        lumi_cfg = VoiceConfig.from_character(lumi)
        results.append(check("Lumi pitch=+4st", lumi_cfg.pitch_semitones == 4, f"got {lumi_cfg.pitch_semitones}"))

    # ── 2. Mock synthesis test ───────────────────────────────────────────────
    print(f"\n{BOLD}[2] Synthesis with Mock Engine{RESET}")

    script_path = cfg.PIPELINE_DIRS["scripts"] / "20260924_learn_colors_red" / "script.json"
    results.append(check("Script file exists", script_path.exists(), str(script_path)))

    if script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        mock_engine = MockTTSEngine()
        synth = VoiceSynthesizer(engine=mock_engine)
        manifest = synth.synthesize_script(script, "20260924_learn_colors_red_test")

        results.append(check("Manifest generated", isinstance(manifest, dict)))
        results.append(check("Manifest has scenes", "scenes" in manifest))
        results.append(check(
            "All 8 scenes in manifest",
            len(manifest["scenes"]) == 8,
            f"got {len(manifest.get('scenes', []))}"
        ))
        results.append(check("Total clips > 0", manifest["total_clips"] > 0,
                             f"clips={manifest['total_clips']}"))

        # Check WAV files actually exist on disk
        audio_test_dir = cfg.PIPELINE_DIRS["audio"] / "20260924_learn_colors_red_test"
        wav_files = list(audio_test_dir.glob("*.wav"))
        results.append(check(
            f"WAV files created ({len(wav_files)} files)",
            len(wav_files) > 0
        ))

        # Validate one WAV file is readable
        if wav_files:
            test_wav = wav_files[0]
            try:
                duration = mock_engine.get_wav_duration(test_wav)
                results.append(check(
                    f"WAV file is valid ({test_wav.name})",
                    duration > 0, f"duration={duration:.2f}s"
                ))
            except Exception as e:
                results.append(check("WAV file is valid", False, str(e)))

        # Check manifest was saved to disk
        manifest_path = audio_test_dir / "audio_manifest.json"
        results.append(check("audio_manifest.json exists on disk", manifest_path.exists()))

    # ── 3. Real engine availability check ───────────────────────────────────
    print(f"\n{BOLD}[3] TTS Engine Availability{RESET}")

    sapi = SAPIEngine()
    kokoro = KokoroEngine()
    sapi_ok = sapi.is_available()
    kokoro_ok = kokoro.is_available()

    results.append(check("SAPI5 fallback available", sapi_ok,
                         "Install pyttsx3 if this fails"))
    print(f"         {YELLOW}Kokoro available: {kokoro_ok}{RESET}")
    if not kokoro_ok:
        print(f"         {YELLOW}-> Run: python -m pipeline.voice_synthesizer --download-models{RESET}")

    best = get_best_engine()
    results.append(check(
        f"Best engine selected: {type(best).__name__}",
        best is not None
    ))

    # ── Results ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print()
    print("=" * 58)
    color = GREEN if passed == total else YELLOW
    print(f"  {color}Result: {passed}/{total} checks passed{RESET}")
    if passed == total:
        print(f"  {GREEN}Phase 4 ready!{RESET}")
    else:
        print(f"  {YELLOW}Some checks failed — see details above.{RESET}")
    print("=" * 58)
    print()
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
