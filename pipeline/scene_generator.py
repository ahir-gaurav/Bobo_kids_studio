"""
pipeline/scene_generator.py — BOBO KIDS STUDIO
================================================
Phase 5: Script JSON → PNG scene background images.

WHAT THIS MODULE DOES:
  1. Loads script.json (from Phase 3)
  2. Reads each scene's 'visual' description
  3. Builds a child-safe, style-consistent prompt
  4. Generates a 1280×720 PNG for each scene using a cloud image engine
  5. Saves images to scenes/<video_id>/
  6. Writes scenes_manifest.json (used by Phase 7 for video assembly)

ENGINE STRATEGY (same pattern as Phase 4 voice):
  BaseImageEngine (abstract)
    ├── PollinationsEngine  ← PRIMARY: free Flux-powered HTTP API (no key, no VRAM)
    └── PillowFallbackEngine ← FALLBACK: programmatic cartoon backgrounds (always works)

WHY POLLINATIONS.AI?
  - Completely free, no API key, no rate-limit registration
  - Uses FLUX.1-schnell model — state of the art quality
  - Simple HTTP GET: https://image.pollinations.ai/prompt/{text}
  - Perfect for iteration: try prompts fast, refine, repeat
  - We can swap in local Stable Diffusion later without changing any other code

STYLE GUIDE ENFORCEMENT:
  Every prompt gets a fixed style suffix:
    "...children's cartoon, flat design, vibrant saturated colors,
     simple rounded shapes, Pixar-inspired, safe for age 3-5, no text overlay"
  This ensures ALL scene images look like they belong to the same show.

SCENES MANIFEST:
  scenes_manifest.json is the contract between Phase 5 (images) and Phase 7 (video).
  It lists every PNG with its scene number, duration, and visual metadata.
"""

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from config import cfg
from pipeline import get_logger
from pipeline.character_loader import CharacterLoader

logger = get_logger("scene_generator")

# ── Constants ─────────────────────────────────────────────────────────────────

# Target resolution — 16:9 for the long video
SCENE_WIDTH  = 1280
SCENE_HEIGHT = 720

# Style suffix appended to every scene prompt.
# WHY? Consistency. Without this, Pollinations gives a different art style
# every image. This makes every scene look like the same children's show.
STYLE_SUFFIX = (
    "children's cartoon illustration, flat design, vibrant saturated colors, "
    "simple rounded shapes, Pixar and Disney Junior inspired, "
    "clean background, warm friendly atmosphere, "
    "safe for children age 3 to 5, no text, no letters, no words, "
    "no scary elements, no photorealism"
)

# Negative things we never want in a children's video
NEGATIVE_PROMPT = (
    "realistic, photographic, dark, scary, violence, text, watermark, "
    "adult content, blurry, low quality, ugly, disturbing"
)

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_scene_prompt(visual_description: str, characters: list[str]) -> str:
    """
    Turn a scene's visual description into a Pollinations-ready prompt.

    WHY THIS FUNCTION?
      Raw visual descriptions like "Bobo walks in the park" are fine for
      narrators but weak for image AI. We enrich them with:
        1. Character descriptions (so Bobo looks consistent)
        2. Style tags (so every image looks like the same show)
        3. Composition hints (so images work as video backgrounds)

    Parameters
    ----------
    visual_description : str
        The 'visual' field from script.json for this scene.
    characters : list[str]
        Character names appearing in this scene (for visual consistency).

    Returns
    -------
    str
        Full prompt string ready for the image API.
    """
    # Character visual descriptions for the prompt
    char_descriptions = {
        "bobo": "Bobo the friendly brown bear cub wearing a blue hoodie with letter B",
        "lumi": "Lumi the glowing yellow firefly with soft warm wings",
        "pip":  "Pip the small yellow penguin with an orange beak and tiny flippers",
    }

    # Add character descriptions if they appear in the scene
    char_hints = []
    for char_name in characters:
        desc = char_descriptions.get(char_name.lower())
        if desc:
            char_hints.append(desc)

    # Build the full prompt
    parts = [visual_description.strip()]
    if char_hints:
        parts.append("featuring " + ", ".join(char_hints))
    parts.append(STYLE_SUFFIX)

    return ", ".join(parts)


# ── Abstract engine ───────────────────────────────────────────────────────────

class BaseImageEngine(ABC):
    """
    Abstract base for image generation engines.
    Concrete subclasses implement generate() differently.
    VoiceSynthesizer from Phase 4 uses the same Strategy Pattern.
    """

    @abstractmethod
    def generate(self, prompt: str, width: int, height: int, output_path: Path) -> bool:
        """
        Generate an image from a prompt and save it.

        Returns
        -------
        bool
            True if generation succeeded, False otherwise.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this engine can be used right now."""
        ...


# ── Pollinations.ai engine ────────────────────────────────────────────────────

class PollinationsEngine(BaseImageEngine):
    """
    Free image generation via Pollinations.ai (Flux model).

    HOW IT WORKS:
      Pollinations.ai exposes a simple HTTP API:
        GET https://image.pollinations.ai/prompt/{encoded_prompt}
            ?width=1280&height=720&seed=42&nologo=true&enhance=true
      The server runs Flux inference and returns a JPEG/PNG.
      No API key, no account, completely free.

    RATE LIMITS:
      Pollinations is generous but not infinite. We:
        - Add a 1-second delay between requests
        - Use seeds based on scene number for reproducibility
        - Cache all images (never regenerate unless forced)

    QUALITY NOTES:
      Flux.1-schnell (4-step diffusion) gives excellent results for
      children's illustrations when combined with the right style prompt.
    """

    BASE_URL = "https://image.pollinations.ai/prompt"
    TIMEOUT_SEC = 90       # Flux inference can take up to 60 seconds
    RETRY_DELAY = 3        # Seconds between retries

    def is_available(self) -> bool:
        """Check internet connectivity by pinging Pollinations."""
        try:
            urllib.request.urlopen(
                "https://image.pollinations.ai/",
                timeout=5
            )
            return True
        except Exception:
            # No internet or service down → fall back to Pillow
            return False

    def generate(self, prompt: str, width: int, height: int, output_path: Path) -> bool:
        """Download a generated image from Pollinations.ai."""
        encoded_prompt = urllib.parse.quote(prompt, safe="")

        # Use a seed based on the output filename for reproducibility
        # WHY? Same prompt + same seed = same image. Reproducible pipeline.
        seed = int(hashlib.md5(output_path.name.encode()).hexdigest(), 16) % 999999

        url = (
            f"{self.BASE_URL}/{encoded_prompt}"
            f"?width={width}&height={height}"
            f"&seed={seed}"
            f"&nologo=true"
            f"&enhance=true"
            f"&model=flux"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(1, 4):
            try:
                logger.debug(f"Attempt {attempt}: GET {url[:80]}...")
                urllib.request.urlretrieve(url, str(output_path))

                # Verify it's a valid image file (not an error page)
                file_size = output_path.stat().st_size
                if file_size < 10_000:   # Real images are > 50 KB
                    logger.warning(
                        f"Downloaded file too small ({file_size} bytes) — "
                        "may be an error response. Retrying..."
                    )
                    output_path.unlink(missing_ok=True)
                    time.sleep(self.RETRY_DELAY * attempt)
                    continue

                logger.debug(
                    f"Downloaded: {output_path.name} "
                    f"({file_size / 1024:.0f} KB)"
                )
                return True

            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < 3:
                    time.sleep(self.RETRY_DELAY * attempt)

        logger.error(f"All attempts failed for {output_path.name}")
        return False


# ── Pillow fallback engine ────────────────────────────────────────────────────

class PillowFallbackEngine(BaseImageEngine):
    """
    Programmatic scene image generator using Pillow.

    WHY THIS EXISTS:
      If Pollinations is down / no internet / rate-limited, the pipeline
      must still produce SOMETHING for Phase 7 to use.
      This engine creates simple but visually distinct cartoon-like scenes:
        - Gradient sky backgrounds
        - Solid ground strip
        - Geometric shapes (circles, rectangles) as scene objects
        - Scene number and title as text overlay

    QUALITY:
      Lower than AI-generated images, but functional.
      Good enough to test the full Phase 7 video assembly pipeline
      before upgrading to AI images.
    """

    # Colour palette per scene — ensures visual variety
    SCENE_PALETTES = [
        {"sky": (135, 206, 235), "ground": (124, 205, 124), "accent": (255, 100, 100)},  # Blue sky
        {"sky": (255, 220, 120), "ground": (100, 200, 100), "accent": (255, 165,  0)},  # Sunset
        {"sky": (180, 235, 180), "ground": (120, 180, 120), "accent": (200,  50,  50)},  # Green
        {"sky": (200, 180, 255), "ground": (130, 100, 200), "accent": (255, 220,  80)},  # Purple
        {"sky": (255, 200, 200), "ground": (200, 150, 100), "accent": (100, 180, 255)},  # Warm
        {"sky": (150, 220, 255), "ground": ( 80, 160,  80), "accent": (255, 255,  80)},  # Daytime
        {"sky": (255, 255, 200), "ground": (200, 200, 100), "accent": (255,  80,  80)},  # Bright
        {"sky": (200, 240, 200), "ground": (100, 180, 100), "accent": (255, 200,  50)},  # Forest
    ]

    def is_available(self) -> bool:
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def generate(self, prompt: str, width: int, height: int, output_path: Path) -> bool:
        from PIL import Image, ImageDraw, ImageFont

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pick palette based on hash of prompt (consistent per scene)
        palette_idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(self.SCENE_PALETTES)
        palette = self.SCENE_PALETTES[palette_idx]

        img = Image.new("RGB", (width, height), palette["sky"])
        draw = ImageDraw.Draw(img)

        # Simple gradient sky (top to bottom)
        sky_r, sky_g, sky_b = palette["sky"]
        for y in range(height * 2 // 3):
            ratio = y / (height * 2 // 3)
            r = int(sky_r + (255 - sky_r) * ratio * 0.3)
            g = int(sky_g + (255 - sky_g) * ratio * 0.1)
            b = int(sky_b + (255 - sky_b) * ratio * 0.2)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Ground strip (bottom third)
        ground_y = height * 2 // 3
        draw.rectangle([(0, ground_y), (width, height)], fill=palette["ground"])

        # Sun (top-right circle)
        draw.ellipse([(width - 150, 30), (width - 30, 150)], fill=(255, 230, 80))

        # Accent decoration (central element)
        cx, cy = width // 2, height // 2
        acc = palette["accent"]
        draw.ellipse([(cx - 80, cy - 80), (cx + 80, cy + 80)], fill=acc)

        # Scene title overlay (top-left, semi-transparent feel via color)
        title_text = prompt[:60] + "..." if len(prompt) > 60 else prompt
        try:
            font = ImageFont.load_default(size=28)
        except Exception:
            font = ImageFont.load_default()

        # Shadow
        draw.text((22, 22), title_text, fill=(0, 0, 0, 128), font=font)
        draw.text((20, 20), title_text, fill=(255, 255, 255), font=font)

        img.save(str(output_path), "PNG", optimize=True)
        logger.debug(f"Pillow fallback saved: {output_path.name}")
        return True


# ── Engine selector ───────────────────────────────────────────────────────────

def get_best_engine(prefer_fallback: bool = False) -> BaseImageEngine:
    """
    Return the best available image engine.
    Tries Pollinations first; falls back to Pillow.

    Parameters
    ----------
    prefer_fallback : bool
        If True, skip Pollinations and use Pillow directly.
        Useful for offline testing.
    """
    if not prefer_fallback:
        pollinations = PollinationsEngine()
        if pollinations.is_available():
            logger.info("Image engine: Pollinations.ai (Flux model, free)")
            return pollinations

    pillow = PillowFallbackEngine()
    if pillow.is_available():
        logger.warning(
            "Pollinations unavailable — using Pillow fallback. "
            "Images will be programmatically generated (lower quality)."
        )
        return pillow

    raise RuntimeError(
        "No image engine available. Install Pillow: pip install Pillow"
    )


# ── Manifest structures ───────────────────────────────────────────────────────

@dataclass
class SceneImage:
    """Represents one generated scene image."""
    scene_number: int
    title: str
    file: str                  # relative filename within scenes/<video_id>/
    prompt: str                # the prompt used to generate it
    width: int
    height: int
    engine: str
    duration_sec: float        # from script — how long this scene shows
    visual_description: str    # original visual field from script


# ── Main scene generator ──────────────────────────────────────────────────────

class SceneGenerator:
    """
    Generates all scene background images for a script.

    USAGE:
      gen = SceneGenerator()
      manifest = gen.generate_scenes(script, video_id)

    CACHING:
      Like Phase 4, if a PNG already exists on disk, it is reused.
      Use force_redo=True to regenerate everything.

    RATE LIMITING:
      Between each Pollinations request, we wait REQUEST_DELAY seconds.
      This is respectful to the free service and avoids 429 errors.
    """

    REQUEST_DELAY = 2.0    # Seconds between API calls (be a good citizen)

    def __init__(self, engine: Optional[BaseImageEngine] = None):
        self.engine = engine or get_best_engine()
        self._loader = CharacterLoader()

    def generate_scenes(
        self,
        script: dict,
        video_id: str,
        force_redo: bool = False,
    ) -> dict:
        """
        Generate all scene images for a script.

        Parameters
        ----------
        script : dict
            Loaded script.json dict from Phase 3.
        video_id : str
            Used to build output directory (scenes/<video_id>/).
        force_redo : bool
            If True, regenerate even if image already exists.

        Returns
        -------
        dict
            The scenes manifest dict (also saved to scenes_manifest.json).
        """
        scenes_dir = cfg.PIPELINE_DIRS["scenes"] / video_id
        scenes_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = scenes_dir / "scenes_manifest.json"

        # ── Cache check ──────────────────────────────────────────────────────
        if manifest_path.exists() and not force_redo:
            logger.info(f"Scenes manifest exists at {manifest_path}. Loading cache.")
            logger.info("Use force_redo=True to regenerate.")
            return json.loads(manifest_path.read_text(encoding="utf-8"))

        scenes_data: list[SceneImage] = []
        script_scenes = script.get("scenes", [])
        engine_name = type(self.engine).__name__

        logger.info(
            f"Generating {len(script_scenes)} scene images | "
            f"engine={engine_name} | video_id={video_id}"
        )

        with tqdm(total=len(script_scenes), desc="Generating scenes", unit="scene") as pbar:
            for scene in script_scenes:
                snum = scene["scene_number"]
                title = scene.get("title", f"Scene {snum}")
                visual = scene.get("visual", f"A colorful children's cartoon scene {snum}")
                duration = scene.get("duration_sec", 30)
                characters = scene.get("characters", script.get("characters", []))

                # Build prompt
                prompt = build_scene_prompt(visual, characters)
                logger.debug(f"Scene {snum} prompt: {prompt[:80]}...")

                # Output file
                img_file = f"scene_{snum:02d}.png"
                img_path = scenes_dir / img_file

                # Check cache
                if img_path.exists() and not force_redo:
                    logger.debug(f"Reusing cached: {img_file}")
                    success = True
                else:
                    logger.debug(f"Generating: {img_file}")
                    success = self.engine.generate(
                        prompt=prompt,
                        width=SCENE_WIDTH,
                        height=SCENE_HEIGHT,
                        output_path=img_path,
                    )

                    # Rate limiting — only delay after actual API calls
                    if success and isinstance(self.engine, PollinationsEngine):
                        time.sleep(self.REQUEST_DELAY)

                if not success:
                    # If engine fails, use Pillow fallback for this scene
                    logger.warning(
                        f"Scene {snum} generation failed. "
                        "Using Pillow fallback for this scene."
                    )
                    fallback = PillowFallbackEngine()
                    success = fallback.generate(prompt, SCENE_WIDTH, SCENE_HEIGHT, img_path)

                file_size_kb = img_path.stat().st_size / 1024 if img_path.exists() else 0

                scene_img = SceneImage(
                    scene_number=snum,
                    title=title,
                    file=img_file,
                    prompt=prompt,
                    width=SCENE_WIDTH,
                    height=SCENE_HEIGHT,
                    engine=engine_name if success else "PillowFallbackEngine",
                    duration_sec=duration,
                    visual_description=visual,
                )
                scenes_data.append(scene_img)

                logger.info(
                    f"Scene {snum:02d}: {title:<25} | "
                    f"{file_size_kb:.0f} KB | "
                    f"{'OK' if success else 'FAILED'}"
                )
                pbar.update(1)

        # ── Build and save manifest ──────────────────────────────────────────
        manifest = {
            "video_id": video_id,
            "engine": engine_name,
            "scene_count": len(scenes_data),
            "resolution": f"{SCENE_WIDTH}x{SCENE_HEIGHT}",
            "scenes_dir": str(scenes_dir),
            "scenes": [
                {
                    "scene_number": s.scene_number,
                    "title": s.title,
                    "file": s.file,
                    "width": s.width,
                    "height": s.height,
                    "duration_sec": s.duration_sec,
                    "visual_description": s.visual_description,
                    "prompt_used": s.prompt,
                    "engine": s.engine,
                }
                for s in scenes_data
            ],
        }

        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        logger.info(
            f"Scenes manifest saved: {manifest_path} | "
            f"{len(scenes_data)} scenes | "
            f"engine={engine_name}"
        )
        return manifest


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    force_redo = "--force-redo" in args
    use_fallback = "--fallback" in args
    args = [a for a in args if not a.startswith("--")]

    video_id = args[0] if args else "20260924_learn_colors_red"
    script_path = cfg.PIPELINE_DIRS["scripts"] / video_id / "script.json"

    if not script_path.exists():
        sys.stderr.write(
            f"[ERROR] Script not found: {script_path}\n"
            "Run Phase 3 first: python -m pipeline.script_generator\n"
        )
        sys.exit(1)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    sys.stderr.write(f"\nPhase 5 -- Scene Image Generation\n")
    sys.stderr.write(f"Topic:  {script['topic']}\n")
    sys.stderr.write(f"Scenes: {len(script['scenes'])}\n")
    sys.stderr.write(f"Mode:   {'FALLBACK (Pillow)' if use_fallback else 'Pollinations.ai'}\n\n")

    engine = get_best_engine(prefer_fallback=use_fallback)
    gen = SceneGenerator(engine=engine)
    manifest = gen.generate_scenes(script, video_id, force_redo=force_redo)

    sys.stderr.write(f"\n=== Scene Generation Complete ===\n")
    sys.stderr.write(f"Engine:  {manifest['engine']}\n")
    sys.stderr.write(f"Scenes:  {manifest['scene_count']}\n")
    sys.stderr.write(f"Output:  {cfg.PIPELINE_DIRS['scenes'] / video_id}\n\n")

    # Show quick summary
    for s in manifest["scenes"]:
        sys.stderr.write(
            f"  Scene {s['scene_number']:02d}: {s['title']:<25} -> {s['file']}\n"
        )
    sys.stderr.write("\n")
