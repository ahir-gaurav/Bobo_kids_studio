"""
pipeline/shorts_generator.py — BOBO KIDS STUDIO
=================================================
Phase 8: Main video + Script → YouTube Shorts (vertical 9:16 MP4s)

WHAT THIS MODULE DOES:
  1. Reads the 'shorts' section of script.json (Gemini defined these in Phase 3)
  2. For each Short: picks the source scenes + their audio
  3. Converts scene images from 16:9 → 9:16 using the blurred background method
  4. Adds text overlays: hook at top, branding at bottom, caption text
  5. Assembles audio from the source scenes
  6. Encodes with FFmpeg → vertical MP4 ready for YouTube Shorts

SHORTS FORMAT (YouTube official specs):
  Resolution:  1080 x 1920 (9:16 portrait, full vertical screen)
  Duration:    15 - 60 seconds (we aim for 25-45s)
  Frame rate:  30 fps
  Codec:       H.264 (same as main video)
  Audio:       AAC 128 kbps stereo

16:9 → 9:16 CONVERSION (Blurred Background Method):
  Why not just rotate? Rotating a 16:9 video gives you 9:16 with squished content.
  The professional way:

    ┌─────────────┐
    │  [blurred]  │  ← Top: blurred + darkened version of scene image
    │  [blurred]  │
    ├─────────────┤
    │  [original] │  ← Middle: actual scene image scaled to fill width
    │   content   │
    │  [original] │
    ├─────────────┤
    │  [blurred]  │  ← Bottom: blurred + darkened + text overlays
    │  text here  │
    └─────────────┘

TEXT OVERLAYS:
  - Hook text at top (large, bold, child-friendly)
  - Caption text at bottom (smaller, coloured bar)
  - BOBO KIDS STUDIO branding (bottom strip)
  All text is rendered by Pillow ImageDraw with drop shadows for legibility.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

from config import cfg
from pipeline import get_logger
from pipeline.video_assembler import (
    SAMPLE_RATE, FPS,
    get_ffmpeg, run_ffmpeg,
    load_wav, loop_to_length, pad_or_trim,
    AudioTimeline,
)

logger = get_logger("shorts_generator")

# ── Shorts format constants ───────────────────────────────────────────────────

SHORT_WIDTH    = 1080
SHORT_HEIGHT   = 1920
SHORT_FPS      = 30
SHORT_MAX_SEC  = 59      # YouTube Shorts must be under 60s

# Brand colours
BRAND_RED      = (220, 50, 50)
BRAND_YELLOW   = (255, 220, 50)
BRAND_WHITE    = (255, 255, 255)
BRAND_BLACK    = (20, 20, 20)
TEXT_SHADOW    = (0, 0, 0, 160)   # Semi-transparent shadow


# ── 16:9 → 9:16 Image Converter ──────────────────────────────────────────────

def convert_to_vertical(
    img: Image.Image,
    width: int = SHORT_WIDTH,
    height: int = SHORT_HEIGHT,
    blur_radius: int = 25,
    darken: float = 0.55,    # How much to darken the blurred background (0-1)
) -> Image.Image:
    """
    Convert a landscape 16:9 image to a portrait 9:16 image.

    Method: Blurred background
      1. Stretch the image to fill the full 9:16 canvas → blur heavily
      2. Darken the blurred background for contrast
      3. Scale original image to fit within width (letterbox-style)
      4. Paste the sharp image centered on the blurred background

    This is the same technique used by YouTube, Instagram Reels, and
    TikTok auto-formatters. It looks professional and wastes no space.

    Parameters
    ----------
    img : PIL Image (any aspect ratio)
    width, height : target Short dimensions
    blur_radius : how much to blur the background (higher = more blurred)
    darken : background darkening factor (0 = black, 1 = full brightness)
    """
    img = img.convert("RGB")

    # ── Step 1: Blurred background ──
    # Stretch to fill full 9:16 canvas
    bg = img.resize((width, height), Image.BILINEAR)
    # Heavy Gaussian blur
    bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # Darken so it doesn't compete with foreground content
    bg_array = np.array(bg).astype(float) * darken
    bg = Image.fromarray(bg_array.astype(np.uint8))

    # ── Step 2: Foreground (sharp image) ──
    # Scale to fit full width, maintaining aspect ratio
    scale = width / img.width
    fg_h = int(img.height * scale)
    fg = img.resize((width, fg_h), Image.LANCZOS)

    # ── Step 3: Composite ──
    result = bg.copy()
    y_offset = (height - fg_h) // 2    # Vertically center the foreground
    result.paste(fg, (0, y_offset))

    return result


def add_text_overlay(
    img: Image.Image,
    hook: str,
    caption: str,
    channel_name: str = "BOBO KIDS STUDIO",
) -> Image.Image:
    """
    Add text overlays to a 9:16 image.

    Layout:
      - Hook text: large, at top (visible on the blurred area)
      - Caption bar: coloured stripe at bottom with caption text
      - Channel brand: very bottom strip

    WHY DRAW SHADOWS?
      Text on a video frame must be readable against ANY background colour.
      A dark shadow/outline behind white text works on both light and dark images.
      This is the standard technique used by all video editors.
    """
    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size

    # Try to load a bold font; fall back to default
    try:
        # Default Pillow font — no external font needed
        font_large  = ImageFont.load_default(size=72)
        font_medium = ImageFont.load_default(size=48)
        font_small  = ImageFont.load_default(size=36)
    except Exception:
        font_large = font_medium = font_small = ImageFont.load_default()

    def draw_text_with_shadow(text, x, y, font, color, shadow_offset=3):
        """Draw text with a drop shadow for legibility on any background."""
        # Shadow
        draw.text((x + shadow_offset, y + shadow_offset), text,
                  font=font, fill=(0, 0, 0, 200))
        # Main text
        draw.text((x, y), text, font=font, fill=color)

    def wrap_text(text, font, max_width):
        """Simple word wrap for long text."""
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = font.getbbox(test)
            if bbox[2] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    # ── Hook text at top ──────────────────────────────────────────────────────
    # Semi-transparent dark bar at top for hook readability
    top_bar_h = 160
    draw.rectangle([(0, 0), (w, top_bar_h)], fill=(0, 0, 0, 120))

    lines = wrap_text(hook, font_large, w - 60)
    y_pos = 20
    for line in lines[:2]:   # Max 2 lines for hook
        bbox = font_large.getbbox(line)
        x_pos = (w - (bbox[2] - bbox[0])) // 2   # Centre horizontally
        draw_text_with_shadow(line, x_pos, y_pos, font_large, BRAND_WHITE)
        y_pos += bbox[3] - bbox[1] + 8

    # ── Caption bar at bottom ─────────────────────────────────────────────────
    bar_h = 120
    bar_y = h - bar_h - 80
    draw.rectangle([(0, bar_y), (w, bar_y + bar_h)], fill=(*BRAND_RED, 220))

    lines = wrap_text(caption, font_medium, w - 60)
    y_pos = bar_y + 15
    for line in lines[:2]:
        bbox = font_medium.getbbox(line)
        x_pos = (w - (bbox[2] - bbox[0])) // 2
        draw_text_with_shadow(line, x_pos, y_pos, font_medium, BRAND_YELLOW)
        y_pos += bbox[3] - bbox[1] + 6

    # ── Channel branding strip at very bottom ─────────────────────────────────
    brand_bar_y = h - 75
    draw.rectangle([(0, brand_bar_y), (w, h)], fill=(*BRAND_BLACK, 220))
    bbox = font_small.getbbox(channel_name)
    x_pos = (w - (bbox[2] - bbox[0])) // 2
    draw_text_with_shadow(channel_name, x_pos, brand_bar_y + 18, font_small, BRAND_YELLOW)

    return img


# ── Short definition ──────────────────────────────────────────────────────────

@dataclass
class ShortSpec:
    """Everything needed to generate one YouTube Short."""
    short_number: int
    title: str
    hook: str
    source_scenes: list[int]
    duration_sec: float
    ending: str
    caption_text: str
    script_text: str


# ── Main generator ────────────────────────────────────────────────────────────

class ShortsGenerator:
    """
    Generates YouTube Shorts from the main video's assets.

    USAGE:
      gen = ShortsGenerator()
      manifest = gen.generate(video_id)
    """

    def __init__(self):
        self.output_dir = cfg.PIPELINE_DIRS["output"]

    def _load_all_manifests(self, video_id: str) -> tuple:
        dirs = cfg.PIPELINE_DIRS
        script       = json.loads((dirs["scripts"] / video_id / "script.json").read_text("utf-8"))
        audio_mf     = json.loads((dirs["audio"]   / video_id / "audio_manifest.json").read_text("utf-8"))
        scenes_mf    = json.loads((dirs["scenes"]  / video_id / "scenes_manifest.json").read_text("utf-8"))
        video_mf     = json.loads((self.output_dir / video_id / "video_manifest.json").read_text("utf-8"))
        return script, audio_mf, scenes_mf, video_mf

    def _build_short_audio(
        self,
        source_scenes: list[int],
        audio_manifest: dict,
        video_id: str,
        target_duration: float,
    ) -> np.ndarray:
        """
        Assemble audio from the given source scenes.
        Trims/pads to target_duration.
        """
        audio_scenes = {s["scene_number"]: s for s in audio_manifest["scenes"]}
        audio_dir = cfg.PIPELINE_DIRS["audio"] / video_id

        samples_list = []
        for snum in source_scenes:
            sc = audio_scenes.get(snum, {})

            # Narration
            narr = sc.get("narration")
            if narr and narr.get("file"):
                path = audio_dir / narr["file"]
                if path.exists():
                    wav, sr = load_wav(path)
                    samples_list.append(wav)

            # Dialogue clips
            for dlg in sc.get("dialogue", []):
                if dlg.get("file"):
                    path = audio_dir / dlg["file"]
                    if path.exists():
                        wav, sr = load_wav(path)
                        samples_list.append(wav)

        if not samples_list:
            return np.zeros(int(target_duration * SAMPLE_RATE))

        combined = np.concatenate(samples_list)
        target_samples = int(target_duration * SAMPLE_RATE)
        return pad_or_trim(combined, target_samples)

    def _render_short_frames(
        self,
        source_scenes: list[int],
        scenes_manifest: dict,
        video_id: str,
        frames_dir: Path,
        duration_sec: float,
        hook: str,
        caption: str,
    ) -> int:
        """
        Render all frames for a Short as vertical 9:16 PNGs.
        Returns total frames rendered.
        """
        scene_images = {s["scene_number"]: s for s in scenes_manifest["scenes"]}
        scenes_dir = cfg.PIPELINE_DIRS["scenes"] / video_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Calculate how long to show each source scene
        n_scenes = len(source_scenes)
        per_scene_sec = duration_sec / n_scenes
        total_frames = int(duration_sec * SHORT_FPS)

        frame_idx = 0
        for i, snum in enumerate(source_scenes):
            scene_data = scene_images.get(snum)
            if not scene_data:
                logger.warning(f"No image data for scene {snum}")
                continue

            img_path = scenes_dir / scene_data["file"]
            if not img_path.exists():
                logger.warning(f"Scene image not found: {img_path}")
                continue

            # Load and convert to 9:16
            img = Image.open(img_path).convert("RGB")
            vertical = convert_to_vertical(img)

            # Add text overlays
            # Show hook on first scene, caption on last scene
            if i == 0:
                vertical = add_text_overlay(vertical, hook, caption, cfg.CHANNEL_NAME)
            else:
                vertical = add_text_overlay(vertical, "", caption, cfg.CHANNEL_NAME)

            # Write frames for this scene's portion
            scene_frames = int(per_scene_sec * SHORT_FPS)
            # Ken Burns: slight zoom on each scene
            zoom_from = 1.0 if i % 2 == 0 else 1.04
            zoom_to   = 1.04 if i % 2 == 0 else 1.0

            for fi in range(scene_frames):
                progress = fi / scene_frames
                zoom = zoom_from + (zoom_to - zoom_from) * progress

                # Apply zoom to vertical frame
                w, h = vertical.size
                new_w = int(w * zoom)
                new_h = int(h * zoom)
                zoomed = vertical.resize((new_w, new_h), Image.BILINEAR)
                left = (new_w - w) // 2
                top  = (new_h - h) // 2
                frame = zoomed.crop((left, top, left + w, top + h))

                frame_path = frames_dir / f"frame_{frame_idx:07d}.png"
                frame.save(str(frame_path), "PNG")
                frame_idx += 1

        return frame_idx

    def generate_one_short(
        self,
        spec: ShortSpec,
        video_id: str,
        audio_manifest: dict,
        scenes_manifest: dict,
        force_redo: bool = False,
    ) -> dict:
        """Generate a single YouTube Short. Returns its output info dict."""
        short_dir = self.output_dir / video_id
        short_dir.mkdir(parents=True, exist_ok=True)

        out_mp4 = short_dir / f"short_{spec.short_number:02d}.mp4"
        if out_mp4.exists() and not force_redo:
            logger.info(f"Short {spec.short_number} cached: {out_mp4.name}")
            return {
                "short_number": spec.short_number,
                "title": spec.title,
                "file": out_mp4.name,
                "duration_sec": spec.duration_sec,
                "size_mb": round(out_mp4.stat().st_size / (1024*1024), 2),
            }

        duration = min(spec.duration_sec, SHORT_MAX_SEC)
        frames_dir = short_dir / f"frames_short_{spec.short_number:02d}"

        logger.info(f"Generating Short {spec.short_number}: '{spec.title}' ({duration}s)")

        # ── Step 1: Render frames ──────────────────────────────────────────────
        n_frames = self._render_short_frames(
            source_scenes=spec.source_scenes,
            scenes_manifest=scenes_manifest,
            video_id=video_id,
            frames_dir=frames_dir,
            duration_sec=duration,
            hook=spec.hook,
            caption=spec.caption_text,
        )
        logger.debug(f"Short {spec.short_number}: {n_frames} frames rendered")

        # ── Step 2: Build audio ───────────────────────────────────────────────
        audio_data = self._build_short_audio(
            source_scenes=spec.source_scenes,
            audio_manifest=audio_manifest,
            video_id=video_id,
            target_duration=duration,
        )
        # Normalise
        peak = np.max(np.abs(audio_data))
        if peak > 1e-6:
            audio_data = audio_data * (0.9 / peak)

        audio_path = short_dir / f"short_{spec.short_number:02d}_audio.wav"
        sf.write(str(audio_path), audio_data.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")

        # ── Step 3: FFmpeg encode ─────────────────────────────────────────────
        frame_list = short_dir / f"frames_short_{spec.short_number:02d}_list.txt"
        frame_files = sorted(frames_dir.glob("frame_*.png"))

        with frame_list.open("w", encoding="utf-8") as f:
            for fp in frame_files:
                f.write(f"file '{fp.as_posix()}'\n")
                f.write(f"duration {1/SHORT_FPS:.6f}\n")

        run_ffmpeg([
            "-f", "concat", "-safe", "0", "-i", str(frame_list),
            "-i", str(audio_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-shortest",
            str(out_mp4),
        ])

        # ── Step 4: Cleanup ───────────────────────────────────────────────────
        for fp in frames_dir.glob("*.png"):
            fp.unlink()
        frame_list.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)

        size_mb = round(out_mp4.stat().st_size / (1024*1024), 2)
        logger.info(f"Short {spec.short_number} done: {out_mp4.name} ({size_mb} MB, {duration}s)")

        return {
            "short_number": spec.short_number,
            "title": spec.title,
            "file": out_mp4.name,
            "duration_sec": duration,
            "source_scenes": spec.source_scenes,
            "hook": spec.hook,
            "size_mb": size_mb,
        }

    def generate(self, video_id: str, force_redo: bool = False) -> dict:
        """Generate all Shorts for a video. Returns shorts manifest."""
        manifest_path = self.output_dir / video_id / "shorts_manifest.json"
        if manifest_path.exists() and not force_redo:
            logger.info("Shorts manifest cached. Loading.")
            return json.loads(manifest_path.read_text("utf-8"))

        script, audio_mf, scenes_mf, video_mf = self._load_all_manifests(video_id)

        # Parse Short specs from script
        shorts_data = script.get("shorts", [])
        if not shorts_data:
            # Auto-generate: pick 2 best scenes
            logger.warning("No 'shorts' in script — auto-selecting scenes 1+3 and 5+6")
            shorts_data = [
                {"short_number": 1, "title": f"{script['topic']} Highlight 1",
                 "hook": f"Learn with Bobo!", "source_scenes": [1, 3],
                 "duration_sec": 30, "ending": "Subscribe!", "caption_text": "Fun learning!",
                 "script": ""},
                {"short_number": 2, "title": f"{script['topic']} Highlight 2",
                 "hook": "Bobo needs your help!", "source_scenes": [5, 6],
                 "duration_sec": 30, "ending": "Subscribe!", "caption_text": "Watch more!",
                 "script": ""},
            ]

        specs = [
            ShortSpec(
                short_number=s["short_number"],
                title=s["title"],
                hook=s["hook"],
                source_scenes=s["source_scenes"],
                duration_sec=s.get("duration_sec", 30),
                ending=s.get("ending", ""),
                caption_text=s.get("caption_text", ""),
                script_text=s.get("script", ""),
            )
            for s in shorts_data
        ]

        shorts_results = []
        print()
        for spec in specs:
            print(f"--- Short {spec.short_number}: '{spec.title}' ---")
            result = self.generate_one_short(
                spec, video_id, audio_mf, scenes_mf, force_redo
            )
            shorts_results.append(result)

        manifest = {
            "video_id":  video_id,
            "shorts_count": len(shorts_results),
            "shorts": shorts_results,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8"
        )
        logger.info(f"Shorts manifest saved: {manifest_path}")
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

    sys.stderr.write(f"\nPhase 8 -- YouTube Shorts Generation\n")
    sys.stderr.write(f"Video ID: {video_id}\n\n")

    gen = ShortsGenerator()
    manifest = gen.generate(video_id, force_redo=force_redo)

    sys.stderr.write(f"\n=== Shorts Generation Complete ===\n")
    sys.stderr.write(f"Shorts created: {manifest['shorts_count']}\n\n")
    for s in manifest["shorts"]:
        sys.stderr.write(
            f"  Short {s['short_number']:02d}: '{s['title']}'\n"
            f"           {s['duration_sec']}s | {s['size_mb']} MB | {s['file']}\n"
        )
    sys.stderr.write("\n")
