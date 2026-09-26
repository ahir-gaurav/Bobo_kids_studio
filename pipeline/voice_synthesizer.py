"""
pipeline/voice_synthesizer.py — BOBO KIDS STUDIO
=================================================
Stage 2 of the pipeline: Script JSON → WAV audio files per scene.

WHAT THIS MODULE DOES:
  1. Loads script.json from Phase 3
  2. Extracts narration text + dialogue lines for each scene
  3. Synthesises each chunk to a WAV file using a TTS engine
  4. Applies per-character voice settings (speed, pitch)
  5. Saves all audio to audio/<video_id>/
  6. Writes audio_manifest.json (critical for Phase 7 video assembly)

DESIGN PATTERN — Strategy Pattern:
  We define a BaseTTSEngine interface. Two concrete implementations:
    - KokoroEngine   : Kokoro-82M via ONNX runtime (best quality)
    - SAPIEngine     : Windows SAPI5 via pyttsx3 (always works, fallback)

  The VoiceSynthesizer class doesn't care WHICH engine it uses.
  It just calls engine.synthesize(text, config) → WAV file.
  Swapping engines = one line change. This is called the Strategy Pattern.

WHY PITCH SHIFT VIA SAMPLE RATE TRICK?
  Kokoro and SAPI don't have native pitch controls.
  Solution: save the WAV at a slightly different sample rate than reality.
    - Playback at 24000 Hz but saved at 24000 * 2^(semitones/12) Hz
    - The player thinks it's hearing 24000 Hz, but data is shifted up
  This changes both pitch AND speed slightly.
  In Phase 7 we'll refine this using FFmpeg's asetrate filter for
  pitch-only shift without speed change.

AUDIO MANIFEST (audio_manifest.json):
  This file is the contract between Phase 4 (voice) and Phase 7 (video).
  It lists every WAV file, who speaks it, and how long it is.
  Phase 7 uses this to know exactly when to place each audio clip.
"""

import json
import math
import os
import struct
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import cfg
from pipeline import get_logger
from pipeline.character_loader import CharacterLoader, Character

logger = get_logger("voice_synthesizer")


# ── Voice configuration per audio chunk ─────────────────────────────────────

@dataclass
class VoiceConfig:
    """
    Settings for a single TTS call.
    Holds the character's voice preferences (or narrator defaults).
    """
    voice_id: str = "af_heart"      # Kokoro voice ID or SAPI voice index
    speed: float = 1.0              # Speaking rate (0.5 = slow, 2.0 = fast)
    pitch_semitones: int = 0        # Pitch shift in semitones (+2 = higher)
    engine_hint: str = "kokoro"     # Preferred engine

    # Pre-built narrator config
    @classmethod
    def narrator(cls) -> "VoiceConfig":
        """Narrator voice: clear, warm, moderate pace."""
        return cls(voice_id="af_heart", speed=0.9, pitch_semitones=1)

    @classmethod
    def from_character(cls, char: Character) -> "VoiceConfig":
        """Build config from a Character dataclass."""
        return cls(
            voice_id=char.voice_id,
            speed=char.voice_speed,
            pitch_semitones=char.voice_pitch_semitones,
            engine_hint=char.voice_engine,
        )


# ── Abstract TTS engine interface ────────────────────────────────────────────

class BaseTTSEngine(ABC):
    """
    Abstract base class for TTS engines.

    WHY ABSTRACT?
      By defining an interface (synthesize method), we guarantee that any
      engine we write works the same way. The VoiceSynthesizer doesn't need
      to know the implementation details — just the contract.

      This is the STRATEGY PATTERN: define an algorithm interface,
      have concrete implementations, swap them freely.
    """

    @abstractmethod
    def synthesize(self, text: str, config: VoiceConfig, output_path: Path) -> float:
        """
        Convert text to speech and save as WAV.

        Parameters
        ----------
        text : str
            The text to speak.
        config : VoiceConfig
            Voice settings (speed, pitch, voice_id).
        output_path : Path
            Where to save the WAV file.

        Returns
        -------
        float
            Actual duration of the saved audio in seconds.
        """
        ...

    def get_wav_duration(self, path: Path) -> float:
        """Read duration from a WAV file without loading the full audio."""
        try:
            with wave.open(str(path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception:
            return 0.0


# ── Kokoro ONNX engine ───────────────────────────────────────────────────────

class KokoroEngine(BaseTTSEngine):
    """
    TTS engine using Kokoro-82M via ONNX runtime.

    QUALITY: Best — neural, expressive, 82M parameter model
    VRAM:    ~0 (runs on CPU via ONNX)
    SPEED:   ~0.3–1.0x realtime on CPU (fast enough for our batch use)
    VOICES:  50+ voices; af_heart for Bobo/Lumi, am_fenrir for Pip

    Setup required:
      - pip install kokoro-onnx
      - Download kokoro-v1.0.onnx + voices-v1.0.bin to models/kokoro/
    """

    MODEL_DIR = cfg.ROOT / "models" / "kokoro"
    MODEL_FILE = MODEL_DIR / "kokoro-v1.0.onnx"
    VOICES_FILE = MODEL_DIR / "voices-v1.0.bin"

    # Sample rate Kokoro outputs at
    SAMPLE_RATE = 24000

    def __init__(self):
        self._kokoro = None     # Lazy-load: don't import until first use
        self._available = None  # Cache availability check

    def is_available(self) -> bool:
        """Check if Kokoro can actually be used (model files + package exist)."""
        if self._available is not None:
            return self._available

        # Check package importable
        try:
            import kokoro_onnx  # noqa: F401
        except ImportError:
            logger.warning("kokoro-onnx package not importable.")
            self._available = False
            return False

        # Check model files downloaded
        if not self.MODEL_FILE.exists():
            logger.warning(
                f"Kokoro model not found at {self.MODEL_FILE}. "
                "Run: python -m pipeline.voice_synthesizer --download-models"
            )
            self._available = False
            return False

        if not self.VOICES_FILE.exists():
            logger.warning(f"Kokoro voices file not found at {self.VOICES_FILE}.")
            self._available = False
            return False

        self._available = True
        return True

    def _get_kokoro(self):
        """Lazy-load the Kokoro model (only on first synthesis call)."""
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            logger.info(f"Loading Kokoro model from {self.MODEL_FILE}...")
            self._kokoro = Kokoro(str(self.MODEL_FILE), str(self.VOICES_FILE))
            logger.info("Kokoro model loaded.")
        return self._kokoro

    def synthesize(self, text: str, config: VoiceConfig, output_path: Path) -> float:
        """Synthesise text using Kokoro-82M ONNX."""
        kokoro = self._get_kokoro()

        # Kokoro returns (samples_array, sample_rate)
        samples, sample_rate = kokoro.create(
            text=text,
            voice=config.voice_id,
            speed=config.speed,
            lang="en-us",
        )

        # Apply pitch shift via sample rate trick
        # WHY: Saving at a higher "declared" rate means playback at normal
        # rate is faster/higher pitched. Simple but introduces slight tempo shift.
        # Phase 7 will refine with FFmpeg for exact pitch-only adjustment.
        if config.pitch_semitones != 0:
            shift_factor = 2 ** (config.pitch_semitones / 12.0)
            effective_rate = int(sample_rate / shift_factor)
        else:
            effective_rate = sample_rate

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_wav(samples, effective_rate, output_path)

        return self.get_wav_duration(output_path)

    def _write_wav(self, samples, sample_rate: int, path: Path):
        """Write float32 numpy samples to a 16-bit PCM WAV file."""
        import numpy as np
        import soundfile as sf

        # Convert float32 [-1, 1] to int16 if not already
        if samples.dtype != np.int16:
            samples_int16 = (samples * 32767).astype(np.int16)
        else:
            samples_int16 = samples

        sf.write(str(path), samples_int16, sample_rate, subtype="PCM_16")


# ── Windows SAPI5 fallback engine ────────────────────────────────────────────

class SAPIEngine(BaseTTSEngine):
    """
    TTS engine using Windows SAPI5 via pyttsx3.

    QUALITY: Lower — robotic, no fine voice control
    VRAM:    0 (entirely CPU, Windows native)
    SPEED:   Very fast (< 1s per chunk)
    VOICES:  Windows built-in voices (usually 1-3 per language)

    WHY THIS EXISTS:
      If Kokoro fails (Python 3.14 incompatibility, model not downloaded,
      etc.), pyttsx3 keeps the WHOLE PIPELINE running. You can test all
      downstream stages (video assembly, thumbnail, upload) with SAPI
      audio, then upgrade to Kokoro later.

      This is the principle of "degrade gracefully": never let one
      component failure stop unrelated components from working.
    """

    SAMPLE_RATE = 22050   # Standard SAPI output rate

    def __init__(self):
        self._engine = None
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            self._available = True
        except Exception as e:
            logger.warning(f"pyttsx3 not available: {e}")
            self._available = False
        return self._available

    def _get_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
        return self._engine

    def synthesize(self, text: str, config: VoiceConfig, output_path: Path) -> float:
        import pyttsx3

        # WHY create a fresh engine every call?
        #   pyttsx3 wraps Windows COM (Component Object Model).
        #   Reusing a COM engine across multiple save_to_file() + runAndWait()
        #   calls can cause the engine to block indefinitely on the second call.
        #   Creating a new engine instance each time is slightly slower (~0.1s)
        #   but is 100% reliable. This is the correct SAPI5 usage pattern.
        engine = pyttsx3.init()

        # SAPI5 rate: default 200 wpm.
        sapi_rate = int(200 * config.speed)
        engine.setProperty("rate", sapi_rate)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        engine.save_to_file(text, str(output_path))
        engine.runAndWait()
        engine.stop()

        # Explicitly release COM resources
        del engine

        # Short pause to let SAPI release file locks (Windows quirk)
        time.sleep(0.05)

        duration = self.get_wav_duration(output_path)
        if config.pitch_semitones != 0:
            logger.debug(
                f"SAPI engine: pitch shift ({config.pitch_semitones}st) not applied. "
                "Will be applied in Phase 7 via FFmpeg."
            )
        return duration


# ── Engine selector ──────────────────────────────────────────────────────────

def get_best_engine() -> BaseTTSEngine:
    """
    Return the best available TTS engine.
    Tries Kokoro first; falls back to SAPI.

    WHY THIS FUNCTION EXISTS:
      Centralises the "which engine?" decision. Every module that needs
      TTS calls this function — they never hardcode an engine.
    """
    kokoro = KokoroEngine()
    if kokoro.is_available():
        logger.info("TTS Engine selected: Kokoro-82M (ONNX)")
        return kokoro

    sapi = SAPIEngine()
    if sapi.is_available():
        logger.warning(
            "Kokoro unavailable — falling back to Windows SAPI5. "
            "Voice quality will be lower. Download Kokoro models to upgrade."
        )
        return sapi

    raise RuntimeError(
        "No TTS engine available. "
        "Install pyttsx3 (pip install pyttsx3) or set up Kokoro models."
    )


# ── Manifest data structures ─────────────────────────────────────────────────

@dataclass
class AudioClip:
    """Represents one synthesised audio chunk."""
    file: str                  # relative path within audio/<video_id>/
    text: str                  # the text that was spoken
    duration_sec: float        # actual audio length in seconds
    character: str = "narrator"
    scene_number: int = 0
    clip_type: str = "narration"   # "narration" | "dialogue" | "song"

@dataclass
class SceneAudio:
    """All audio clips for one scene."""
    scene_number: int
    title: str
    narration: Optional[AudioClip] = None
    dialogue: list[AudioClip] = field(default_factory=list)
    all_clips: list[AudioClip] = field(default_factory=list)

    def total_duration(self) -> float:
        return sum(c.duration_sec for c in self.all_clips)


# ── Main synthesizer ─────────────────────────────────────────────────────────

class VoiceSynthesizer:
    """
    Orchestrates TTS synthesis for an entire script.

    USAGE:
      synth = VoiceSynthesizer()
      manifest = synth.synthesize_script(script, video_id)

    HOW IT WORKS:
      For each scene in script["scenes"]:
        1. Synthesise narration → scene_NN_narration.wav
        2. For each dialogue line → scene_NN_dialogue_<char>_NN.wav
      Then write audio_manifest.json.
    """

    def __init__(self, engine: Optional[BaseTTSEngine] = None):
        """
        Parameters
        ----------
        engine : BaseTTSEngine, optional
            TTS engine to use. Auto-selects best available if not given.
        """
        self.engine = engine or get_best_engine()
        self._characters: dict[str, Character] = {}
        self._load_characters()

    def _load_characters(self):
        """Cache character voice configs by slug name."""
        loader = CharacterLoader()
        for char in loader.all():
            self._characters[char.name.lower()] = char
        logger.info(f"Voice configs loaded for: {list(self._characters.keys())}")

    def _get_voice_config(self, character_name: str) -> VoiceConfig:
        """
        Get voice config for a character name (e.g. 'bobo', 'lumi').
        Returns narrator config if character not found.
        """
        char = self._characters.get(character_name.lower())
        if char:
            return VoiceConfig.from_character(char)
        logger.debug(f"No character config for '{character_name}' — using narrator voice.")
        return VoiceConfig.narrator()

    def _safe_filename(self, text: str, max_chars: int = 30) -> str:
        """Turn text into a safe filename fragment."""
        import re
        slug = re.sub(r"[^a-z0-9 ]", "", text.lower())
        slug = re.sub(r"\s+", "_", slug.strip())
        return slug[:max_chars]

    def synthesize_script(
        self,
        script: dict,
        video_id: str,
        force_redo: bool = False
    ) -> dict:
        """
        Synthesise ALL audio for a script.

        Parameters
        ----------
        script : dict
            Loaded script.json dict from Phase 3.
        video_id : str
            Used to build the output directory path.
        force_redo : bool
            If False (default), skip files that already exist on disk.

        Returns
        -------
        dict
            The audio manifest dict (also saved to audio_manifest.json).
        """
        audio_dir = cfg.PIPELINE_DIRS["audio"] / video_id
        audio_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = audio_dir / "audio_manifest.json"

        # ── Cache check ──────────────────────────────────────────────────────
        if manifest_path.exists() and not force_redo:
            logger.info(f"Audio manifest exists at {manifest_path}. Loading from cache.")
            logger.info("Use force_redo=True to re-synthesise.")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        scenes_audio: list[SceneAudio] = []
        all_clips: list[AudioClip] = []

        # ── Count total clips for progress bar ──────────────────────────────
        total_clips = 0
        for scene in script.get("scenes", []):
            if scene.get("narration", "").strip():
                total_clips += 1
            total_clips += len(scene.get("dialogue", []))

        logger.info(
            f"Synthesising {total_clips} audio clips for "
            f"video_id='{video_id}' using {type(self.engine).__name__}"
        )

        # ── Process each scene ───────────────────────────────────────────────
        with tqdm(total=total_clips, desc="Synthesising audio", unit="clip") as pbar:
            for scene in script.get("scenes", []):
                snum = scene["scene_number"]
                scene_audio = SceneAudio(
                    scene_number=snum,
                    title=scene.get("title", f"Scene {snum}")
                )

                # ── Narration ────────────────────────────────────────────────
                narration_text = scene.get("narration", "").strip()
                if narration_text:
                    narr_file = f"scene_{snum:02d}_narration.wav"
                    narr_path = audio_dir / narr_file

                    if narr_path.exists() and not force_redo:
                        duration = self.engine.get_wav_duration(narr_path)
                        logger.debug(f"Reusing cached: {narr_file}")
                    else:
                        logger.debug(f"Synthesising: {narr_file}")
                        config = VoiceConfig.narrator()
                        duration = self.engine.synthesize(narration_text, config, narr_path)

                    clip = AudioClip(
                        file=narr_file,
                        text=narration_text,
                        duration_sec=round(duration, 3),
                        character="narrator",
                        scene_number=snum,
                        clip_type="narration",
                    )
                    scene_audio.narration = clip
                    scene_audio.all_clips.append(clip)
                    all_clips.append(clip)
                    pbar.update(1)

                # ── Dialogue lines ───────────────────────────────────────────
                for d_idx, dialogue in enumerate(scene.get("dialogue", []), start=1):
                    char_name = dialogue.get("character", "narrator").strip()
                    line_text = dialogue.get("line", "").strip()
                    if not line_text:
                        pbar.update(1)
                        continue

                    dial_file = f"scene_{snum:02d}_dialogue_{char_name}_{d_idx:02d}.wav"
                    dial_path = audio_dir / dial_file

                    if dial_path.exists() and not force_redo:
                        duration = self.engine.get_wav_duration(dial_path)
                        logger.debug(f"Reusing cached: {dial_file}")
                    else:
                        logger.debug(f"Synthesising: {dial_file}")
                        config = self._get_voice_config(char_name)
                        duration = self.engine.synthesize(line_text, config, dial_path)

                    clip = AudioClip(
                        file=dial_file,
                        text=line_text,
                        duration_sec=round(duration, 3),
                        character=char_name,
                        scene_number=snum,
                        clip_type="dialogue",
                    )
                    scene_audio.dialogue.append(clip)
                    scene_audio.all_clips.append(clip)
                    all_clips.append(clip)
                    pbar.update(1)

                scenes_audio.append(scene_audio)
                logger.info(
                    f"Scene {snum} done | "
                    f"clips={len(scene_audio.all_clips)} | "
                    f"duration={scene_audio.total_duration():.1f}s"
                )

        # ── Build and save manifest ──────────────────────────────────────────
        total_audio_sec = sum(c.duration_sec for c in all_clips)
        manifest = {
            "video_id": video_id,
            "engine": type(self.engine).__name__,
            "total_clips": len(all_clips),
            "total_audio_duration_sec": round(total_audio_sec, 2),
            "audio_dir": str(audio_dir),
            "scenes": [
                {
                    "scene_number": sa.scene_number,
                    "title": sa.title,
                    "scene_total_duration_sec": round(sa.total_duration(), 2),
                    "narration": {
                        "file": sa.narration.file,
                        "text": sa.narration.text,
                        "duration_sec": sa.narration.duration_sec,
                    } if sa.narration else None,
                    "dialogue": [
                        {
                            "character": c.character,
                            "line": c.text,
                            "file": c.file,
                            "duration_sec": c.duration_sec,
                        }
                        for c in sa.dialogue
                    ],
                }
                for sa in scenes_audio
            ],
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Audio manifest saved: {manifest_path} | "
            f"total clips={len(all_clips)} | "
            f"total duration={total_audio_sec:.1f}s"
        )
        return manifest


# ── Model downloader ─────────────────────────────────────────────────────────

def download_kokoro_models():
    """
    Download Kokoro model files to models/kokoro/.
    Run once before using KokoroEngine.

    Files are ~340 MB total.
    """
    import urllib.request

    model_dir = cfg.ROOT / "models" / "kokoro"
    model_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "kokoro-v1.0.onnx": (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
            "model-files-v1.0/kokoro-v1.0.onnx"
        ),
        "voices-v1.0.bin": (
            "https://github.com/thewh1teagle/kokoro-onnx/releases/download/"
            "model-files-v1.0/voices-v1.0.bin"
        ),
    }

    for filename, url in files.items():
        dest = model_dir / filename
        if dest.exists():
            logger.info(f"Already downloaded: {filename} ({dest.stat().st_size / 1e6:.1f} MB)")
            continue

        logger.info(f"Downloading {filename} from {url}")
        print(f"Downloading {filename}... (this may take a few minutes)")

        def progress(count, block_size, total_size):
            pct = min(100, count * block_size * 100 // total_size)
            print(f"\r  {pct}% ({count * block_size / 1e6:.1f} MB)", end="", flush=True)

        try:
            urllib.request.urlretrieve(url, dest, reporthook=progress)
            print(f"\r  Done! {dest.stat().st_size / 1e6:.1f} MB saved to {dest}")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            if dest.exists():
                dest.unlink()
            raise


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    args = sys.argv[1:]

    if "--download-models" in args:
        print("Downloading Kokoro model files...")
        download_kokoro_models()
        print("Done. Re-run without --download-models to synthesise audio.")
        sys.exit(0)

    # Default: synthesise the most recently generated script
    video_id = args[0] if args else "20260924_learn_colors_red"
    script_path = cfg.PIPELINE_DIRS["scripts"] / video_id / "script.json"

    if not script_path.exists():
        sys.stderr.write(
            f"[ERROR] Script not found: {script_path}\n"
            f"Run Phase 3 first: python -m pipeline.script_generator\n"
        )
        sys.exit(1)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    sys.stderr.write(f"\nPhase 4 — Voice Synthesis\n")
    sys.stderr.write(f"Script: {script['topic']}\n")
    sys.stderr.write(f"Scenes: {len(script['scenes'])}\n\n")

    synthesizer = VoiceSynthesizer()
    manifest = synthesizer.synthesize_script(script, video_id)

    sys.stderr.write(f"\n=== Audio Synthesis Complete ===\n")
    sys.stderr.write(f"Engine:   {manifest['engine']}\n")
    sys.stderr.write(f"Clips:    {manifest['total_clips']}\n")
    sys.stderr.write(f"Duration: {manifest['total_audio_duration_sec']}s\n")
    sys.stderr.write(
        f"Output:   {cfg.PIPELINE_DIRS['audio'] / video_id}\n\n"
    )
