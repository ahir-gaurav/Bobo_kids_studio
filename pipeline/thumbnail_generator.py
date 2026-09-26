"""
pipeline/thumbnail_generator.py — BOBO KIDS STUDIO
====================================================
Phase 9: Script + Scene Image → YouTube Thumbnail (1280x720 JPEG)

WHAT THIS MODULE DOES:
  1. Reads script metadata (topic, hook, learning goal) from script.json
  2. Generates an AI image via Pollinations.ai (Flux model) as the base
  3. Resizes to exact YouTube spec: 1280x720 px, max 2 MB
  4. Adds professional text overlays using Pillow:
       - Episode title ('Learn Colors:') in yellow
       - Topic word ('RED') in giant 3D-styled text
       - Left-side vignette for text contrast
       - Channel badge (bottom-right)
       - Age/category label (bottom-left)
  5. Saves final JPEG + thumbnail_manifest.json

YOUTUBE THUMBNAIL SPEC (official 2026):
  Resolution: 1280 x 720 pixels (minimum 640x360)
  Aspect ratio: 16:9
  Format: JPG, GIF, BMP, PNG (we use JPG for smallest file size)
  Max file size: 2 MB
  Quality guideline: as high as possible without hitting 2 MB limit

WHY TEXT OVERLAYS MATTER:
  On mobile YouTube, thumbnails are shown at ~200x112px.
  Text must be readable at that tiny size. That means:
    - Very large font size (relative to 1280x720)
    - High contrast (yellow on dark OR white with thick dark outline)
    - Maximum 3-4 words visible at a glance
  Professional channels spend as much time on thumbnails as on the video
  itself — it's often the ONLY thing that gets someone to click.

STITCH INTEGRATION:
  Phase 9 used Google Stitch MCP to design the thumbnail layout.
  The final prompt for Pollinations was refined by Stitch's AI.
  This demonstrates the Stitch → Pollinations workflow for AI design.
"""

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config import cfg
from pipeline import get_logger

logger = get_logger("thumbnail_generator")

# ── YouTube Thumbnail Spec ────────────────────────────────────────────────────
THUMB_W        = 1280
THUMB_H        = 720
THUMB_MAX_BYTES = cfg.THUMBNAIL_MAX_BYTES   # 2 MB

# ── Colours ───────────────────────────────────────────────────────────────────
YELLOW      = (255, 230, 40)
RED_BRIGHT  = (255, 40, 40)
RED_DARK    = (160, 20, 20)
GOLD        = (255, 200, 0)
WHITE       = (255, 255, 255)
NEAR_BLACK  = (20, 10, 10)


# ── Prompt builder ────────────────────────────────────────────────────────────

# Map of topic keyword → thumbnail prompt style
TOPIC_PROMPT_TEMPLATES = {
    "red":    "vibrant red radial gradient background, red apple, red balloon, red strawberry, red flowers",
    "blue":   "vibrant blue gradient background, blue ocean waves, blue butterfly, blue balloon",
    "yellow": "vibrant sunny yellow gradient background, yellow sunflower, yellow banana, yellow star",
    "green":  "vibrant green gradient background, green leaves, green frog, green apple",
    "orange": "vibrant orange gradient background, orange pumpkin, orange carrot, orange butterfly",
    "purple": "vibrant purple gradient background, purple grapes, purple flower, purple butterfly",
}

def build_thumbnail_prompt(script: dict) -> str:
    """
    Build an AI image prompt for the thumbnail from the script metadata.

    Extracts: topic, color keyword, channel name, episode hook.
    Returns a fully qualified Pollinations/Flux image prompt.
    """
    topic    = script.get("topic", "Learn Colors")
    metadata = script.get("metadata", {})
    hook     = ""
    # Find first scene hook (narration field is a string)
    scenes = script.get("scenes", [])
    if scenes:
        narr = scenes[0].get("narration", "")
        hook = (narr if isinstance(narr, str) else narr.get("text", ""))[:60]

    # Detect color keyword from topic
    topic_lower = topic.lower()
    color_key = next((c for c in TOPIC_PROMPT_TEMPLATES if c in topic_lower), None)
    color_elements = TOPIC_PROMPT_TEMPLATES.get(color_key, "colorful background with floating shapes")

    # Topic word (the colour to display large)
    topic_word = topic.split(":")[-1].strip() if ":" in topic else topic

    prompt = (
        f"A professional high-impact YouTube thumbnail for a toddler learning video "
        f"titled '{topic}' by 'BOBO KIDS STUDIO' for ages 3 to 5. "
        f"{color_elements}. "
        f"Center-right: an adorable chubby 3D cartoon baby brown bear with huge shiny expressive "
        f"eyes blushing cheeks and joyful smile, waving one paw while holding a shiny {color_key or 'colorful'} object. "
        f"Floating in background are cute {color_key or 'colorful'} themed objects with smiling faces. "
        f"Top-left: huge bold bubbly 3D sunny yellow text 'Learn Colors:' with thick dark outline. "
        f"Below it: massive puffy 3D balloon-style text '{topic_word.upper()}' in glossy "
        f"{color_key or 'bright'} color with glowing yellow outline. "
        f"Bottom-right circular badge dark {color_key or 'blue'} with white text 'BOBO KIDS STUDIO'. "
        f"Ultra-saturated high-contrast clean kid-friendly Pixar Cocomelon animation style, "
        f"modern YouTube thumbnail aesthetic, extremely detailed."
    )
    return prompt


# ── Image generation ──────────────────────────────────────────────────────────

def generate_via_pollinations(
    prompt: str,
    output_path: Path,
    width: int = THUMB_W,
    height: int = THUMB_H,
    seed: int = 9042,
    max_retries: int = 3,
) -> bool:
    """
    Generate a thumbnail base image via Pollinations.ai (Flux model).
    Returns True on success.

    WHY seed=9042?
      Same as scene_generator.py: deterministic seed from hash.
      Re-running produces the same image (no surprise regenerations).
    """
    encoded = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&seed={seed}&model=flux&nologo=true"
    )
    logger.debug(f"Pollinations URL (first 120 chars): {url[:120]}...")

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt}: Requesting thumbnail from Pollinations...")
            urllib.request.urlretrieve(url, str(output_path))
            if output_path.exists() and output_path.stat().st_size > 5000:
                logger.info(f"Thumbnail base downloaded: {output_path.stat().st_size // 1024} KB")
                return True
            logger.warning(f"Attempt {attempt}: file too small, retrying...")
        except Exception as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
        if attempt < max_retries:
            time.sleep(5)

    return False


# ── Text overlay ──────────────────────────────────────────────────────────────

def add_thumbnail_overlays(
    img: Image.Image,
    title_line1: str,   # e.g. "Learn Colors:"
    title_line2: str,   # e.g. "RED"
    channel_name: str = "BOBO KIDS STUDIO",
    age_label: str = "Ages 3-5 | Fun Learning!",
) -> Image.Image:
    """
    Add professional text overlays to a 1280x720 thumbnail.

    TEXT HIERARCHY (most important → least):
      1. title_line2 (topic word e.g. 'RED') — BIGGEST, most visible
      2. title_line1 (episode type e.g. 'Learn Colors:') — large
      3. channel_name — badge, small but branded
      4. age_label — very small, SEO/context

    All text uses:
      - Thick dark OUTLINE (readable on any background colour)
      - Drop shadow for 3D depth feeling
      - High-contrast fill colour
    """
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size

    # ── Left vignette for text contrast ──────────────────────────────────────
    # Children's thumbnails often have character on the right,
    # text on the left. A dark vignette makes text pop on any background.
    vignette_width = 520
    for x in range(vignette_width):
        # Cubic falloff for natural-looking vignette
        alpha = int(185 * (1 - (x / vignette_width) ** 0.6))
        draw.line([(x, 0), (x, H)], fill=(0, 0, 0, alpha))

    # ── Load fonts ────────────────────────────────────────────────────────────
    try:
        font_line1   = ImageFont.load_default(size=62)
        font_line2   = ImageFont.load_default(size=118)
        font_badge   = ImageFont.load_default(size=27)
        font_age     = ImageFont.load_default(size=29)
    except Exception:
        font_line1 = font_line2 = font_badge = font_age = ImageFont.load_default()

    def draw_text_outlined(text, x, y, font, fill, outline_color, stroke=5):
        """Draw text with thick outline + subtle drop shadow."""
        # Drop shadow (offset 4px down-right)
        draw.text((x + 4, y + 4), text, font=font, fill=(0, 0, 0, 140))
        # Outline passes
        for dx in range(-stroke, stroke + 1):
            for dy in range(-stroke, stroke + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        # Main text
        draw.text((x, y), text, font=font, fill=fill)

    # ── Line 1: "Learn Colors:" ───────────────────────────────────────────────
    draw_text_outlined(
        title_line1, x=55, y=150,
        font=font_line1,
        fill=YELLOW,
        outline_color=NEAR_BLACK,
        stroke=5,
    )

    # ── Line 2: "RED" (giant) ────────────────────────────────────────────────
    draw_text_outlined(
        title_line2, x=40, y=228,
        font=font_line2,
        fill=RED_BRIGHT,
        outline_color=GOLD,
        stroke=9,
    )

    # ── Channel badge (bottom-right pill) ─────────────────────────────────────
    badge_x1, badge_y1 = 1045, 620
    badge_x2, badge_y2 = 1268, 695
    draw.rounded_rectangle(
        [(badge_x1, badge_y1), (badge_x2, badge_y2)],
        radius=18,
        fill=(*RED_DARK, 230),
    )
    # Centre text inside badge
    bbox = font_badge.getbbox(channel_name)
    tx = badge_x1 + ((badge_x2 - badge_x1) - (bbox[2] - bbox[0])) // 2
    ty = badge_y1 + ((badge_y2 - badge_y1) - (bbox[3] - bbox[1])) // 2
    draw.text((tx, ty), channel_name, font=font_badge, fill=WHITE)

    # ── Age / category label (bottom-left) ────────────────────────────────────
    label_w = font_age.getbbox(age_label)[2] + 30
    draw.rounded_rectangle(
        [(40, 638), (40 + label_w, 688)],
        radius=10,
        fill=(*GOLD, 210),
    )
    draw.text((56, 650), age_label, font=font_age, fill=NEAR_BLACK)

    return img


# ── Main generator class ──────────────────────────────────────────────────────

class ThumbnailGenerator:
    """
    Generates the final YouTube thumbnail for a video.

    USAGE:
      gen = ThumbnailGenerator()
      result = gen.generate(video_id)

    OUTPUT:
      thumbnails/<video_id>/thumbnail_final.jpg   (1280x720, <2MB)
      thumbnails/<video_id>/thumbnail_manifest.json
    """

    def __init__(self):
        self.thumb_dir = cfg.PIPELINE_DIRS["thumbnails"]

    def generate(self, video_id: str, force_redo: bool = False) -> dict:
        """Generate thumbnail. Returns manifest dict."""
        out_dir = self.thumb_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)

        final_path    = out_dir / "thumbnail_final.jpg"
        manifest_path = out_dir / "thumbnail_manifest.json"

        if final_path.exists() and not force_redo:
            logger.info(f"Thumbnail cached: {final_path}")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        # Load script
        script_path = cfg.PIPELINE_DIRS["scripts"] / video_id / "script.json"
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        script = json.loads(script_path.read_text(encoding="utf-8"))

        topic = script.get("topic", "Learn Colors: Red")

        # Title lines
        if ":" in topic:
            line1 = topic.split(":")[0].strip() + ":"  # "Learn Colors:"
            line2 = topic.split(":")[1].strip().upper() # "RED"
        else:
            line1 = "Learn with"
            line2 = "BOBO!"

        logger.info(f"Generating thumbnail for: {topic}")
        logger.info(f"Title lines: '{line1}' / '{line2}'")

        # ── Step 1: Generate AI base image ───────────────────────────────────
        base_path = out_dir / "thumbnail_base.jpg"
        # Use existing thumbnail.jpg (downloaded earlier) as base if present
        existing_base = out_dir / "thumbnail.jpg"
        if existing_base.exists() and not force_redo:
            base_path = existing_base
        elif not base_path.exists() or force_redo:
            prompt = build_thumbnail_prompt(script)
            logger.info("Requesting base image from Pollinations.ai...")
            success = generate_via_pollinations(prompt, base_path)
            if not success:
                logger.warning("Pollinations failed — using scene 1 as fallback base")
                scene_img = cfg.PIPELINE_DIRS["scenes"] / video_id / "scene_01.png"
                if scene_img.exists():
                    base_path = scene_img
                else:
                    # Generate a solid color base
                    fallback = Image.new("RGB", (THUMB_W, THUMB_H), (200, 30, 30))
                    fallback.save(str(base_path))
        else:
            logger.debug("Using cached base image")

        # ── Step 2: Resize to exact YouTube spec ──────────────────────────────
        img = Image.open(base_path).convert("RGB")
        if img.size != (THUMB_W, THUMB_H):
            img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
            logger.debug(f"Resized to {THUMB_W}x{THUMB_H}")

        # ── Step 3: Add text overlays ─────────────────────────────────────────
        img = add_thumbnail_overlays(
            img,
            title_line1=line1,
            title_line2=line2,
            channel_name=cfg.CHANNEL_NAME,
            age_label=f"Ages {cfg.TARGET_AGE_MIN}-{cfg.TARGET_AGE_MAX} | Fun Learning!",
        )

        # ── Step 4: Save with quality check ──────────────────────────────────
        quality = 95
        img.save(str(final_path), "JPEG", quality=quality)
        size_bytes = final_path.stat().st_size

        # If over 2MB, reduce quality
        while size_bytes > THUMB_MAX_BYTES and quality > 70:
            quality -= 5
            img.save(str(final_path), "JPEG", quality=quality)
            size_bytes = final_path.stat().st_size
            logger.debug(f"Reduced JPEG quality to {quality} → {size_bytes//1024} KB")

        size_kb = size_bytes // 1024
        logger.info(
            f"Thumbnail saved: {final_path.name} "
            f"({THUMB_W}x{THUMB_H}, {size_kb} KB, JPEG q={quality})"
        )

        # ── Build manifest ────────────────────────────────────────────────────
        manifest = {
            "video_id":     video_id,
            "file":         str(final_path),
            "width":        THUMB_W,
            "height":       THUMB_H,
            "size_kb":      size_kb,
            "size_bytes":   size_bytes,
            "jpeg_quality": quality,
            "title_line1":  line1,
            "title_line2":  line2,
            "within_2mb":   size_bytes < THUMB_MAX_BYTES,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Thumbnail manifest saved: {manifest_path}")
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

    sys.stderr.write(f"\nPhase 9 -- Thumbnail Generation\n")
    sys.stderr.write(f"Video ID: {video_id}\n\n")

    gen = ThumbnailGenerator()
    manifest = gen.generate(video_id, force_redo=force_redo)

    sys.stderr.write(f"\n=== Thumbnail Generation Complete ===\n")
    sys.stderr.write(f"File:    {manifest['file']}\n")
    sys.stderr.write(f"Size:    {manifest['width']}x{manifest['height']}px, {manifest['size_kb']} KB\n")
    sys.stderr.write(f"2MB limit: {'PASS' if manifest['within_2mb'] else 'FAIL'}\n")
    sys.stderr.write(f"Title:   '{manifest['title_line1']}' / '{manifest['title_line2']}'\n\n")
