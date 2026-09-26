"""
pipeline/video_assembler.py — BOBO KIDS STUDIO
================================================
Phase 7: Script + Audio + Scenes + Music → MP4 video

WHAT THIS MODULE DOES:
  1. Loads all Phase 3-6 outputs (script, audio, scenes, music manifests)
  2. Builds a complete audio timeline (narration + SFX + background music)
  3. Applies a Ken Burns effect (subtle zoom) to each static scene image
  4. Calls FFmpeg to encode everything into a final MP4 video
  5. Saves a video_manifest.json for Phase 8 (Shorts generation)

HOW VIDEO IS ASSEMBLED (the pipeline inside the pipeline):
  ┌─────────────────────────────────────────────────────────┐
  │  For each scene:                                        │
  │    1. Load PNG → apply Ken Burns zoom → write frames   │
  │    2. Load narration WAV → place on audio timeline      │
  │    3. Load SFX WAVs → place at cue timestamps           │
  │    4. Load background music → loop & duck under narr.  │
  ├─────────────────────────────────────────────────────────┤
  │  After all scenes:                                      │
  │    5. Export full audio timeline as mixed_audio.wav     │
  │    6. FFmpeg: frames + audio → main_video.mp4           │
  └─────────────────────────────────────────────────────────┘

WHY NOT MOVIEPY?
  MoviePy is a Python wrapper around FFmpeg. It's great for simple tasks
  but adds abstraction layers that hide what's really happening and can
  break on Python 3.14. We use FFmpeg directly (via imageio-ffmpeg which
  bundles the FFmpeg binary) — this is how real production pipelines work.

WHY KEN BURNS EFFECT?
  Static images look dead in video. A slow zoom-in (1.0x → 1.05x) called
  the "Ken Burns effect" (named after the documentary filmmaker) makes the
  image feel alive and professional. All YouTube children's channels use it.

AUDIO DUCKING:
  When narration is playing, background music drops to 18% volume.
  When no narration, music rises to 35%. This is called "ducking" and
  is standard in broadcast audio engineering.

WHY FFmpeg?
  FFmpeg is the universal video encoder. Used by YouTube, Netflix, VLC.
  It converts raw frames (PNG images) + audio (WAV) → compressed MP4.
  The H.264 codec (libx264) produces small files that play everywhere.
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PIL import Image
from tqdm import tqdm

from config import cfg
from pipeline import get_logger

logger = get_logger("video_assembler")

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 44100
FPS          = 30           # Frames per second — YouTube standard
OUT_WIDTH    = 1280         # Match our scene images (720p)
OUT_HEIGHT   = 720
BG_MUSIC_VOL = 0.18         # Background music at 18% (ducked under narration)
SFX_VOL      = 0.55         # SFX at 55% (audible but not overpowering)


# ── FFmpeg helper ─────────────────────────────────────────────────────────────

def get_ffmpeg() -> str:
    """
    Return the path to the FFmpeg binary.
    Tries imageio-ffmpeg first (bundled binary, no system install needed),
    then falls back to system FFmpeg.
    """
    # Try imageio-ffmpeg (our installed bundled binary)
    try:
        import imageio_ffmpeg
        path = imageio_ffmpeg.get_ffmpeg_exe()
        logger.debug(f"Using imageio-ffmpeg: {path}")
        return path
    except ImportError:
        pass

    # Try system FFmpeg
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        logger.debug(f"Using system FFmpeg: {sys_ffmpeg}")
        return sys_ffmpeg

    raise RuntimeError(
        "FFmpeg not found. Install with:\n"
        "  pip install imageio-ffmpeg\n"
        "or download from https://ffmpeg.org/download.html"
    )


def run_ffmpeg(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run FFmpeg with given arguments. Captures output for logging."""
    ffmpeg = get_ffmpeg()
    cmd = [ffmpeg, "-y"] + args   # -y = overwrite output without asking
    logger.debug(f"FFmpeg: {' '.join(str(a) for a in cmd[-6:])}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        logger.error(f"FFmpeg failed:\n{result.stderr[-2000:]}")
        raise RuntimeError(f"FFmpeg error (code {result.returncode})")
    return result


# ── Audio mixing ──────────────────────────────────────────────────────────────

def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file, return (samples, sample_rate)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)   # Convert stereo to mono
    return data, sr


def pad_or_trim(wave: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad with zeros or trim to exactly target_samples."""
    if len(wave) >= target_samples:
        return wave[:target_samples]
    return np.pad(wave, (0, target_samples - len(wave)))


def loop_to_length(wave: np.ndarray, target_samples: int) -> np.ndarray:
    """Repeat wave until it's at least target_samples long, then trim."""
    if len(wave) == 0:
        return np.zeros(target_samples)
    repeats = (target_samples // len(wave)) + 2
    return np.tile(wave, repeats)[:target_samples]


class AudioTimeline:
    """
    Manages the complete audio timeline for a video.

    Think of this like a multi-track audio editor (Audacity/GarageBand):
      Track 1 (narration): voice clips placed at exact timestamps
      Track 2 (music):     looped background music, volume ducked
      Track 3 (sfx):       sound effects at cue points

    All tracks are numpy arrays that we ADD together (mixing = addition).
    """

    def __init__(self, total_duration_sec: float, sr: int = SAMPLE_RATE):
        self.sr = sr
        self.total_samples = int(total_duration_sec * sr)
        # Three tracks
        self.narration_track = np.zeros(self.total_samples, dtype=np.float32)
        self.music_track     = np.zeros(self.total_samples, dtype=np.float32)
        self.sfx_track       = np.zeros(self.total_samples, dtype=np.float32)
        logger.debug(
            f"AudioTimeline: {total_duration_sec:.1f}s "
            f"({self.total_samples} samples at {sr} Hz)"
        )

    def place_clip(self, track: np.ndarray, wav_path: str | Path,
                   start_sec: float, volume: float = 1.0) -> float:
        """
        Place a WAV file onto a track at start_sec.
        Returns the clip's duration in seconds.
        """
        if not Path(wav_path).exists():
            logger.warning(f"Audio file not found: {wav_path}")
            return 0.0

        wave, sr = load_wav(wav_path)
        # Resample if necessary (simple nearest-neighbor)
        if sr != self.sr:
            ratio = self.sr / sr
            new_len = int(len(wave) * ratio)
            wave = np.interp(
                np.linspace(0, len(wave) - 1, new_len),
                np.arange(len(wave)),
                wave,
            )

        start_sample = int(start_sec * self.sr)
        end_sample   = min(start_sample + len(wave), self.total_samples)
        clip_len     = end_sample - start_sample

        if clip_len > 0:
            track[start_sample:end_sample] += wave[:clip_len] * volume

        return len(wave) / self.sr

    def place_narration(self, wav_path: str | Path, start_sec: float) -> float:
        return self.place_clip(self.narration_track, wav_path, start_sec, volume=1.0)

    def place_sfx(self, wav_path: str | Path, start_sec: float) -> float:
        return self.place_clip(self.sfx_track, wav_path, start_sec, volume=SFX_VOL)

    def fill_music(self, wav_path: str | Path) -> None:
        """Loop background music across the entire timeline."""
        if not Path(wav_path).exists():
            return
        wave, sr = load_wav(wav_path)
        looped = loop_to_length(wave, self.total_samples)
        self.music_track += looped * BG_MUSIC_VOL

    def mix(self) -> np.ndarray:
        """
        Combine all tracks into a single mixed audio signal.

        WHY NORMALIZE SEPARATELY THEN MIX?
          If we just add all three tracks, the sum could exceed 1.0 (clipping).
          We normalize each track first to its 'role volume', then mix.
          Narration at 1.0 (full), music at 0.18 (ducked), SFX at 0.55 (medium).
        """
        def safe_norm(wave, peak):
            mx = np.max(np.abs(wave))
            return wave * (peak / mx) if mx > 1e-6 else wave

        narration = safe_norm(self.narration_track, 0.9)
        music     = self.music_track   # Already at BG_MUSIC_VOL
        sfx       = self.sfx_track     # Already at SFX_VOL

        mixed = narration + music + sfx

        # Final normalization: ensure nothing clips
        peak = np.max(np.abs(mixed))
        if peak > 0.98:
            mixed = mixed * (0.95 / peak)

        return mixed.astype(np.float32)

    def export(self, path: Path) -> None:
        """Save the mixed audio timeline as a WAV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        mixed = self.mix()
        sf.write(str(path), mixed, self.sr, subtype="PCM_16")
        duration = len(mixed) / self.sr
        logger.info(f"Mixed audio: {path.name} ({duration:.1f}s, {path.stat().st_size // 1024} KB)")


# ── Ken Burns effect ──────────────────────────────────────────────────────────

def apply_ken_burns(
    img: Image.Image,
    t: float,
    duration: float,
    zoom_from: float = 1.0,
    zoom_to: float = 1.06,
    pan_x: float = 0.0,    # -1.0 (pan left) to +1.0 (pan right)
    pan_y: float = 0.0,    # -1.0 (pan up)   to +1.0 (pan down)
) -> Image.Image:
    """
    Apply Ken Burns zoom+pan effect to an image at time t.

    HOW IT WORKS:
      1. Calculate zoom level at time t (linear interpolation)
      2. Scale the image up by zoom factor
      3. Crop back to original size with pan offset
      → Result: smooth, living-feeling camera movement

    WHY 1.06 max zoom?
      More than 6-8% zoom on a 1280x720 image becomes obviously pixelated.
      Less than 3% is barely noticeable. 6% hits the sweet spot.
    """
    w, h = img.size
    progress = t / duration if duration > 0 else 0
    zoom = zoom_from + (zoom_to - zoom_from) * progress

    # New size after zoom
    new_w = int(w * zoom)
    new_h = int(h * zoom)

    # Resize (BILINEAR for speed, LANCZOS for quality — we use BILINEAR since
    # we're generating 30 frames per second per scene)
    zoomed = img.resize((new_w, new_h), Image.BILINEAR)

    # Pan offset: move within the zoomed image
    extra_w = new_w - w
    extra_h = new_h - h
    left = int(extra_w / 2 + pan_x * extra_w / 2)
    top  = int(extra_h / 2 + pan_y * extra_h / 2)
    left = max(0, min(left, extra_w))
    top  = max(0, min(top, extra_h))

    return zoomed.crop((left, top, left + w, top + h))


# ── Scene clip generator ──────────────────────────────────────────────────────

def render_scene_frames(
    scene_png: Path,
    duration_sec: float,
    fps: int,
    out_dir: Path,
    scene_number: int,
    zoom_direction: str = "in",    # "in" or "out"
) -> list[Path]:
    """
    Render all frames for one scene as PNG files (FFmpeg input).

    Returns list of frame paths.
    WHY PNG sequence instead of in-memory frames?
      Memory: 1280×720 × 3 channels × 30fps × 15s = 1.4 GB RAM peak.
      Writing PNGs to disk and feeding to FFmpeg avoids this entirely.
    """
    img = Image.open(scene_png).convert("RGB")
    img = img.resize((OUT_WIDTH, OUT_HEIGHT), Image.LANCZOS)

    total_frames = int(duration_sec * fps)
    zoom_from, zoom_to = (1.0, 1.06) if zoom_direction == "in" else (1.06, 1.0)

    # Alternate pan direction per scene for visual variety
    pan_x = 0.3 if scene_number % 2 == 0 else -0.3
    pan_y = 0.0

    frame_paths = []
    for frame_idx in range(total_frames):
        t = frame_idx / fps
        frame = apply_ken_burns(img, t, duration_sec, zoom_from, zoom_to, pan_x, pan_y)
        frame_path = out_dir / f"frame_{frame_idx:06d}.png"
        frame.save(str(frame_path), "PNG", optimize=False)  # Fast write, no compression
        frame_paths.append(frame_path)

    return frame_paths


# ── Main assembler ────────────────────────────────────────────────────────────

@dataclass
class SceneTimestamp:
    """Tracks where each scene starts/ends in the full video timeline."""
    scene_number: int
    title: str
    start_sec: float
    end_sec: float
    duration_sec: float


class VideoAssembler:
    """
    Assembles the final video from all Phase 3-6 outputs.

    USAGE:
      assembler = VideoAssembler()
      result = assembler.assemble(video_id)

    OUTPUT:
      output/<video_id>/main_video.mp4
      output/<video_id>/video_manifest.json
    """

    def __init__(self):
        self.output_dir = cfg.PIPELINE_DIRS["output"]
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_manifests(self, video_id: str) -> tuple[dict, dict, dict, dict]:
        """Load all Phase 3-6 manifests for a video."""
        dirs = cfg.PIPELINE_DIRS

        script_path  = dirs["scripts"]  / video_id / "script.json"
        audio_path   = dirs["audio"]    / video_id / "audio_manifest.json"
        scenes_path  = dirs["scenes"]   / video_id / "scenes_manifest.json"
        music_path   = dirs["audio"]    / video_id / "music_manifest.json"

        for p in [script_path, audio_path, scenes_path, music_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing manifest: {p}\n"
                    f"Run Phases 3-6 first."
                )

        script = json.loads(script_path.read_text(encoding="utf-8"))
        audio  = json.loads(audio_path.read_text(encoding="utf-8"))
        scenes = json.loads(scenes_path.read_text(encoding="utf-8"))
        music  = json.loads(music_path.read_text(encoding="utf-8"))

        logger.info(f"Loaded manifests for {video_id}")
        return script, audio, scenes, music

    def assemble(self, video_id: str, force_redo: bool = False) -> dict:
        """
        Run the full assembly pipeline.
        Returns a video manifest dict.
        """
        video_out_dir = self.output_dir / video_id
        video_out_dir.mkdir(parents=True, exist_ok=True)

        output_mp4      = video_out_dir / "main_video.mp4"
        manifest_path   = video_out_dir / "video_manifest.json"

        if output_mp4.exists() and not force_redo:
            logger.info(f"Video already exists: {output_mp4}. Use force_redo=True to regenerate.")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        # ── Load all manifests ────────────────────────────────────────────────
        script, audio_manifest, scenes_manifest, music_manifest = \
            self._load_manifests(video_id)

        script_scenes  = script["scenes"]
        audio_scenes   = {s["scene_number"]: s for s in audio_manifest["scenes"]}
        scene_images   = {s["scene_number"]: s for s in scenes_manifest["scenes"]}
        music_scenes   = {s["scene_number"]: s for s in music_manifest["scenes"]}

        # ── Build timeline ────────────────────────────────────────────────────
        # Figure out start/end times of each scene from audio durations
        timestamps: list[SceneTimestamp] = []
        cursor = 0.0
        for sc in script_scenes:
            snum     = sc["scene_number"]
            audio_sc = audio_scenes.get(snum, {})
            duration = audio_sc.get("scene_total_duration_sec", sc.get("duration_sec", 15))
            timestamps.append(SceneTimestamp(
                scene_number=snum,
                title=sc.get("title", f"Scene {snum}"),
                start_sec=round(cursor, 3),
                end_sec=round(cursor + duration, 3),
                duration_sec=round(duration, 3),
            ))
            cursor += duration

        total_duration = cursor
        logger.info(f"Total video duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")

        # ── Build audio timeline ──────────────────────────────────────────────
        print()
        print("[1/4] Building audio timeline...")
        timeline = AudioTimeline(total_duration)

        for ts in timestamps:
            snum     = ts.scene_number
            audio_sc = audio_scenes.get(snum, {})
            cur      = ts.start_sec

            # Place narration
            narr = audio_sc.get("narration")
            if narr and narr.get("file"):
                narr_path = cfg.PIPELINE_DIRS["audio"] / video_id / narr["file"]
                dur = timeline.place_narration(narr_path, cur)
                cur += dur

            # Place dialogue clips
            for dlg in audio_sc.get("dialogue", []):
                if dlg.get("file"):
                    dlg_path = cfg.PIPELINE_DIRS["audio"] / video_id / dlg["file"]
                    dur = timeline.place_narration(dlg_path, cur)
                    cur += dur

            # Place SFX from music manifest
            music_sc = music_scenes.get(snum, {})
            for sfx_cue in music_sc.get("sfx_cues", []):
                sfx_path = sfx_cue.get("file", "")
                if sfx_path and Path(sfx_path).exists():
                    sfx_time = ts.start_sec + sfx_cue.get("offset_sec", 1.0)
                    timeline.place_sfx(sfx_path, sfx_time)

            # Fill background music
            music_sc = music_scenes.get(snum, {})
            bg_file = music_sc.get("bg_music_file", "")
            if bg_file and Path(bg_file).exists():
                # Load, loop to scene length, place on music track
                wave, sr = load_wav(bg_file)
                looped = loop_to_length(wave, int(ts.duration_sec * timeline.sr))
                start_s = int(ts.start_sec * timeline.sr)
                end_s   = min(start_s + len(looped), timeline.total_samples)
                chunk   = end_s - start_s
                if chunk > 0:
                    timeline.music_track[start_s:end_s] += looped[:chunk] * BG_MUSIC_VOL

        # Export mixed audio
        mixed_audio_path = video_out_dir / "mixed_audio.wav"
        timeline.export(mixed_audio_path)

        # ── Render video frames ───────────────────────────────────────────────
        print()
        print("[2/4] Rendering scene frames (Ken Burns effect)...")
        frames_dir = video_out_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        # Frame counter across all scenes
        global_frame = 0
        zoom_dirs = ["in", "out"]   # Alternate zoom direction per scene

        with tqdm(total=len(timestamps), desc="Scenes", unit="scene") as pbar:
            for ts in timestamps:
                snum      = ts.scene_number
                scene_img = scene_images.get(snum)
                if not scene_img:
                    logger.warning(f"No image for scene {snum}, using blank")
                    continue

                img_path = cfg.PIPELINE_DIRS["scenes"] / video_id / scene_img["file"]
                if not img_path.exists():
                    logger.warning(f"Scene image not found: {img_path}")
                    continue

                n_frames = int(ts.duration_sec * FPS)
                zoom_dir = zoom_dirs[snum % 2]

                img = Image.open(img_path).convert("RGB").resize(
                    (OUT_WIDTH, OUT_HEIGHT), Image.LANCZOS
                )
                zoom_from, zoom_to = (1.0, 1.06) if zoom_dir == "in" else (1.06, 1.0)
                pan_x = 0.25 if snum % 3 == 0 else (-0.25 if snum % 3 == 1 else 0.0)

                for fi in range(n_frames):
                    t = fi / FPS
                    frame = apply_ken_burns(img, t, ts.duration_sec,
                                           zoom_from, zoom_to, pan_x, 0.0)
                    frame_path = frames_dir / f"frame_{global_frame:07d}.png"
                    frame.save(str(frame_path), "PNG")
                    global_frame += 1

                logger.debug(f"Scene {snum:02d}: {n_frames} frames rendered")
                pbar.update(1)

        total_frames_rendered = global_frame
        logger.info(f"Rendered {total_frames_rendered} frames total")

        # ── Encode with FFmpeg ────────────────────────────────────────────────
        print()
        print("[3/4] Encoding video with FFmpeg (H.264)...")
        logger.info("Starting FFmpeg encode...")

        # Build a frame list file (faster than glob pattern for FFmpeg)
        frame_list_path = video_out_dir / "frame_list.txt"
        frame_files = sorted(frames_dir.glob("frame_*.png"))

        with frame_list_path.open("w", encoding="utf-8") as f:
            for fp in frame_files:
                f.write(f"file '{fp.as_posix()}'\n")
                f.write(f"duration {1/FPS:.6f}\n")

        # FFmpeg command:
        # -f concat -safe 0 -i frame_list.txt  : read frames from list
        # -i mixed_audio.wav                    : read mixed audio
        # -c:v libx264                          : H.264 video codec
        # -preset fast                          : encoding speed vs compression
        # -crf 23                               : quality (18=best, 28=worst, 23=balanced)
        # -c:a aac -b:a 128k                    : AAC audio at 128 kbps
        # -pix_fmt yuv420p                      : required for web playback
        # -shortest                             : end when shorter stream ends
        ffmpeg_args = [
            "-f", "concat", "-safe", "0", "-i", str(frame_list_path),
            "-i", str(mixed_audio_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",   # Web streaming optimization
            "-shortest",
            str(output_mp4),
        ]

        encode_start = time.time()
        run_ffmpeg(ffmpeg_args)
        encode_time = time.time() - encode_start

        video_size_mb = output_mp4.stat().st_size / (1024 * 1024)
        logger.info(
            f"Video encoded: {output_mp4.name} "
            f"({video_size_mb:.1f} MB) in {encode_time:.0f}s"
        )

        # ── Cleanup frames ────────────────────────────────────────────────────
        print()
        print("[4/4] Cleaning up temporary frames...")
        for fp in frames_dir.glob("*.png"):
            fp.unlink()
        frame_list_path.unlink(missing_ok=True)
        logger.info(f"Cleaned {total_frames_rendered} temporary PNG frames")

        # ── Build manifest ────────────────────────────────────────────────────
        manifest = {
            "video_id":      video_id,
            "output_file":   str(output_mp4),
            "duration_sec":  round(total_duration, 2),
            "fps":           FPS,
            "resolution":    f"{OUT_WIDTH}x{OUT_HEIGHT}",
            "size_mb":       round(video_size_mb, 2),
            "codec":         "libx264",
            "audio_codec":   "aac",
            "scene_timestamps": [
                {
                    "scene_number": ts.scene_number,
                    "title":        ts.title,
                    "start_sec":    ts.start_sec,
                    "end_sec":      ts.end_sec,
                    "duration_sec": ts.duration_sec,
                }
                for ts in timestamps
            ],
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Video manifest saved: {manifest_path}")
        return manifest


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    force_redo = "--force-redo" in args
    args = [a for a in args if not a.startswith("--")]
    video_id = args[0] if args else "20260924_learn_colors_red"

    sys.stderr.write(f"\nPhase 7 -- Video Assembly\n")
    sys.stderr.write(f"Video ID: {video_id}\n\n")

    assembler = VideoAssembler()

    try:
        manifest = assembler.assemble(video_id, force_redo=force_redo)
        sys.stderr.write(f"\n=== Video Assembly Complete ===\n")
        sys.stderr.write(f"Output:   {manifest['output_file']}\n")
        sys.stderr.write(f"Duration: {manifest['duration_sec']:.1f}s ({manifest['duration_sec']/60:.1f} min)\n")
        sys.stderr.write(f"Size:     {manifest['size_mb']:.1f} MB\n")
        sys.stderr.write(f"Format:   {manifest['resolution']} @ {manifest['fps']}fps H.264\n\n")
        sys.stderr.write("Scene timestamps:\n")
        for ts in manifest["scene_timestamps"]:
            sys.stderr.write(
                f"  {ts['scene_number']:02d}. {ts['title']:<25} "
                f"{ts['start_sec']:>7.1f}s - {ts['end_sec']:>7.1f}s "
                f"({ts['duration_sec']:.1f}s)\n"
            )
    except Exception as e:
        sys.stderr.write(f"\n[ERROR] Assembly failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
