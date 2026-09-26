"""
config.py — BOBO KIDS STUDIO
==============================
Single source of truth for all project paths, settings, and secrets.

WHY THIS FILE EXISTS:
  Every other module imports from here.
  If a path changes, you change it ONCE here, not scattered across 10 files.

HOW IT WORKS:
  1. python-dotenv reads your .env file.
  2. os.environ.get() pulls values from the environment.
  3. Path objects (from pathlib) build cross-platform paths safely.
     On Windows, pathlib handles backslashes for you automatically.

USAGE IN OTHER FILES:
  from config import cfg
  print(cfg.GEMINI_API_KEY)
  print(cfg.PIPELINE_DIRS["scripts"])
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file ───────────────────────────────────────────────────────────
# load_dotenv() looks for a .env file in the current working directory.
# It loads each line as KEY=VALUE into os.environ.
# If .env does not exist, it silently does nothing (no crash).
load_dotenv()


class Config:
    """
    Central configuration class.
    Instantiated once at the bottom of this file as `cfg`.
    Import it like: from config import cfg
    """

    # ── Project root ─────────────────────────────────────────────────────────
    # Path(__file__) is the path to THIS file (config.py).
    # .parent gives us the directory containing config.py = project root.
    ROOT: Path = Path(__file__).parent

    # ── API Keys (from .env — never hardcoded) ───────────────────────────────
    GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

    # ── YouTube OAuth paths (from .env) ──────────────────────────────────────
    YOUTUBE_CLIENT_SECRET_PATH: Path = ROOT / os.environ.get(
        "YOUTUBE_CLIENT_SECRET_PATH", "client_secret.json"
    )
    YOUTUBE_TOKEN_PATH: Path = ROOT / os.environ.get(
        "YOUTUBE_TOKEN_PATH", "token.pickle"
    )

    # ── Channel settings ─────────────────────────────────────────────────────
    CHANNEL_NAME: str = os.environ.get("CHANNEL_NAME", "BOBO KIDS STUDIO")
    TARGET_AGE_MIN: int = int(os.environ.get("TARGET_AGE_MIN", "3"))
    TARGET_AGE_MAX: int = int(os.environ.get("TARGET_AGE_MAX", "5"))
    VIDEO_LENGTH_MIN: int = int(os.environ.get("DEFAULT_VIDEO_LENGTH_MIN", "5"))
    VIDEO_LENGTH_MAX: int = int(os.environ.get("DEFAULT_VIDEO_LENGTH_MAX", "7"))
    SHORTS_COUNT: int = int(os.environ.get("DEFAULT_SHORTS_COUNT", "2"))

    # ── Video technical specs ─────────────────────────────────────────────────
    # WHY 30 FPS and not 24 or 60?
    #   24 FPS = cinematic film look. Good for live action. Slightly choppy for
    #            fast motion in simple animations.
    #   30 FPS = YouTube standard, smooth on all screens, widely supported.
    #            Slightly larger file than 24fps but not significantly.
    #   60 FPS = Gaming/sports. Overkill for children's animation, doubles
    #            render time and file size on your T1000. Not worth it.
    #   DECISION: 30 FPS — matches YouTube's preferred delivery spec.
    FPS: int = 30

    # WHY 1920x1080?
    #   Standard HD. YouTube recommends it. Higher (4K) would multiply
    #   render time 4x with no meaningful benefit for children's content.
    VIDEO_WIDTH: int = 1920
    VIDEO_HEIGHT: int = 1080

    # Shorts are vertical (portrait mode)
    SHORTS_WIDTH: int = 1080
    SHORTS_HEIGHT: int = 1920

    # WHY H.264 codec?
    #   Universal support on all browsers, YouTube players, mobile.
    #   Good compression. Hardware-accelerated encoding available on your T1000.
    #   Alternative: H.265 (smaller files, but slower encode, less compatible).
    VIDEO_CODEC: str = "libx264"
    AUDIO_CODEC: str = "aac"

    # ── Thumbnail specs (YouTube official, verified 2026-09-19) ──────────────
    THUMBNAIL_WIDTH: int = 1280
    THUMBNAIL_HEIGHT: int = 720
    THUMBNAIL_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MB

    # ── Directory paths (all relative to ROOT) ───────────────────────────────
    PIPELINE_DIRS: dict = {
        "pipeline":    ROOT / "pipeline",
        "characters":  ROOT / "characters",
        "assets":      ROOT / "assets",
        "backgrounds": ROOT / "assets" / "backgrounds",
        "music":       ROOT / "assets" / "music",
        "sfx":         ROOT / "assets" / "sfx",
        "fonts":       ROOT / "assets" / "fonts",
        "scripts":     ROOT / "scripts",
        "audio":       ROOT / "audio",
        "scenes":      ROOT / "scenes",        # Phase 5: generated scene images
        "thumbnails":  ROOT / "thumbnails",    # Phase 9: generated thumbnails
        "output":      ROOT / "output",
        "db":          ROOT / "db",
        "logs":        ROOT / "logs",
        "dashboard":   ROOT / "dashboard",
        "tests":       ROOT / "tests",
    }

    # ── Database ─────────────────────────────────────────────────────────────
    DB_PATH: Path = ROOT / "db" / "channel.db"

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_DIR: Path = ROOT / "logs"
    LOG_LEVEL: str = "DEBUG"  # DEBUG during development, INFO in production

    def validate(self) -> bool:
        """
        Check that the environment is set up correctly.
        Called by validate_setup.py.
        Returns True if all critical checks pass, False otherwise.
        """
        errors = []

        # Check all required directories exist
        for name, path in self.PIPELINE_DIRS.items():
            if not path.exists():
                errors.append(f"Missing directory: {path}")

        # Check .env was loaded (GEMINI_API_KEY can be empty in Phase 1,
        # but .env file itself must exist)
        env_file = self.ROOT / ".env"
        if not env_file.exists():
            errors.append("Missing .env file — copy .env.template to .env")

        if errors:
            for e in errors:
                print(f"  [FAIL] {e}")
            return False

        return True


# ── Singleton instance ───────────────────────────────────────────────────────
# WHY a singleton?
#   We create ONE cfg object here. Every other file imports this same object.
#   This means the .env is only loaded once, and all settings are consistent.
cfg = Config()

