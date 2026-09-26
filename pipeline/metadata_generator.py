"""
pipeline/metadata_generator.py — BOBO KIDS STUDIO
===================================================
Phase 10: Script + Video Manifest → YouTube Metadata JSON

WHAT THIS MODULE DOES:
  1. Reads the script, video manifest, and shorts manifest
  2. Calls Gemini AI to generate optimised YouTube metadata:
       - Main video: title, description (with timestamps!), tags
       - Each Short: short-form title + description
  3. Applies YouTube limits and SEO best practices automatically
  4. Saves metadata.json for Phase 11 (human review) and Phase 12 (upload)

WHY GEMINI FOR METADATA?
  Writing good YouTube metadata is a skill. For children's educational content:
    - Titles must include keywords parents search for ("learn colors for kids")
    - Descriptions must have timestamps (YouTube SEO requirement for longer videos)
    - Tags must cover both specific ("red apple preschool") and broad ("kids learning")
    - Language must be age-appropriate and safe
  Gemini understands all of this and generates better metadata in 2 seconds
  than a human could write in 20 minutes.

YOUTUBE METADATA LIMITS (official, verified 2026):
  Title:       max 100 characters
  Description: max 5000 characters
  Tags:        max 500 characters total (all tags combined)
  Category:    27 = Education (correct for children's learning)
  Made for kids: True (legally required for content targeting under 13)
  Language:    en (English)

SEO STRATEGY FOR CHILDREN'S YOUTUBE:
  1. TITLE FORMULA: [Emoji] + [Action Verb] + [Topic] + [Age/Audience]
     Example: "🔴 Learn Colors RED with Bobo Bear! | Fun for Toddlers"
  2. DESCRIPTION: Always include timestamps — YouTube indexes them
     and parents search for specific segments ("what time does the song start?")
  3. TAGS: Mix of:
     - Broad: "learn colors", "kids education", "preschool learning"
     - Specific: "learn red color", "red objects for kids", "bobo kids studio"
     - Competitor-adjacent: "cocomelon colors", "blippi colors"
     - Long-tail: "learn colors with cartoon bear", "red color lesson preschool"
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from tqdm import tqdm

from config import cfg
from pipeline import get_logger

logger = get_logger("metadata_generator")

# ── YouTube Limits ────────────────────────────────────────────────────────────
TITLE_MAX_CHARS       = 100
DESCRIPTION_MAX_CHARS = 5000
TAGS_MAX_CHARS        = 500     # Sum of all tag lengths + separators
YOUTUBE_CATEGORY_EDUCATION = "27"

# ── Gemini Setup ──────────────────────────────────────────────────────────────

def _get_gemini_model():
    """Initialise Gemini. Raises if GEMINI_API_KEY is missing."""
    if not cfg.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Add it to your .env file:\n"
            "  GEMINI_API_KEY=your_key_here\n"
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=cfg.GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-3.8-flash")


def _call_gemini(model, prompt: str, max_retries: int = 4) -> str:
    """
    Call Gemini and return the text response.
    Handles 429 rate limits by reading the suggested retry_delay from the
    error message and waiting that long before retrying.

    Free tier limit: 5 requests/minute for gemini-3.8-flash.
    We add a 15s gap between successful calls (see generate() below).
    """
    import re as _re
    for attempt in range(1, max_retries + 1):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            logger.warning(f"Gemini attempt {attempt} failed: {err_str[:200]}")
            if attempt < max_retries:
                # Parse suggested retry delay from 429 message
                # e.g. "Please retry in 45.856178579s."
                match = _re.search(r'retry in (\d+\.?\d*)s', err_str)
                if match:
                    wait = float(match.group(1)) + 5   # add 5s buffer
                    logger.info(f"Rate limit: waiting {wait:.0f}s before retry...")
                    print(f"  [rate limit] waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    time.sleep(15 * attempt)   # fallback: 15s, 30s, 45s
    raise RuntimeError(f"Gemini failed after {max_retries} attempts")


# ── Metadata validation helpers ───────────────────────────────────────────────

def trim_to_limit(text: str, limit: int) -> str:
    """Trim text to character limit. Cuts at last sentence boundary if possible."""
    if len(text) <= limit:
        return text
    trimmed = text[:limit]
    # Try to cut at last sentence
    last_period = trimmed.rfind(".")
    if last_period > limit * 0.8:
        return trimmed[:last_period + 1]
    return trimmed.rstrip() + "..."


def parse_tags_from_response(raw: str) -> list[str]:
    """
    Parse a list of tags from Gemini's response.
    Handles various formats: comma-separated, numbered, bullet points.
    """
    # Remove markdown formatting
    raw = re.sub(r'\*\*|__|\*|_|`', '', raw)
    # Try to split by comma first
    if "," in raw:
        tags = [t.strip().strip('"').strip("'") for t in raw.split(",")]
    else:
        # Split by newline (numbered or bullet list)
        lines = raw.strip().split("\n")
        tags = []
        for line in lines:
            line = re.sub(r'^\s*[\d\-\*\.]+\s*', '', line).strip().strip('"').strip("'")
            if line:
                tags.append(line)

    # Filter out empty strings and overly long tags
    tags = [t for t in tags if 1 < len(t) <= 50]
    return tags


def enforce_tag_limit(tags: list[str], max_chars: int = TAGS_MAX_CHARS) -> list[str]:
    """
    Trim tags list so total character count stays within YouTube's limit.
    YouTube counts: sum of all tag lengths + (num_tags - 1) commas + spaces.
    """
    result = []
    total = 0
    for tag in tags:
        # Each tag: length + ", " separator (2 chars) except last
        cost = len(tag) + (2 if result else 0)
        if total + cost > max_chars:
            break
        result.append(tag)
        total += cost
    return result


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_title_prompt(script: dict) -> str:
    topic       = script.get("topic", "Learn Colors")
    target_age  = script.get("target_age", "3-5")
    learning    = script.get("learning_goal", "Identify the color red")
    channel     = cfg.CHANNEL_NAME

    return f"""You are an expert YouTube SEO specialist for children's educational content.

Generate ONE YouTube video title for this children's educational video.

VIDEO DETAILS:
  Channel: {channel}
  Topic: {topic}
  Target Age: {target_age} years
  Learning Goal: {learning}

TITLE REQUIREMENTS:
  - Maximum 100 characters (STRICT — count carefully)
  - Must include: topic keyword parents would search for
  - Must feel exciting and child-friendly
  - Include 1-2 emojis (relevant to topic)
  - Include audience descriptor: "for Kids", "for Toddlers", "for Preschoolers"
  - Format: [Emoji] + Action phrase + character name + audience descriptor
  - Example good title: "🔴 Learn Colors RED with Bobo Bear! Fun for Toddlers & Kids"

Output ONLY the title text. No explanation, no quotes, no other text."""


def build_description_prompt(script: dict, video_manifest: dict) -> str:
    topic        = script.get("topic", "Learn Colors")
    learning     = script.get("learning_goal", "")
    raw_chars = script.get("characters", [])
    if raw_chars and isinstance(raw_chars[0], dict):
        characters = [c.get("name", "") for c in raw_chars]
    else:
        characters = [str(c) for c in raw_chars]
    intro_song   = script.get("intro_song", {})
    channel      = cfg.CHANNEL_NAME
    target_age   = script.get("target_age", "3-5")

    # Build timestamp list from video manifest
    timestamps = []
    for ts in video_manifest.get("scene_timestamps", []):
        m = int(ts["start_sec"]) // 60
        s = int(ts["start_sec"]) % 60
        timestamps.append(f"  {m:02d}:{s:02d} - {ts['title']}")
    timestamp_block = "\n".join(timestamps)

    chars_str = ", ".join(characters) if characters else "Bobo, Lumi, Pip"

    return f"""You are an expert YouTube SEO specialist for children's educational content.

Write a complete YouTube video description for this children's educational video.

VIDEO DETAILS:
  Channel: {channel}
  Topic: {topic}
  Target Age: {target_age} years
  Learning Goal: {learning}
  Characters: {chars_str}
  Duration: {video_manifest.get('duration_sec', 102):.0f} seconds

VIDEO TIMESTAMPS (USE EXACTLY THESE, do not change the times):
{timestamp_block}

DESCRIPTION REQUIREMENTS:
  1. Opening hook: 2-3 exciting sentences about what kids will learn (use emojis)
  2. Include the exact timestamp section formatted like:
     📚 WHAT'S IN THIS VIDEO:
     00:00 - [scene title]
     (etc.)
  3. Learning outcomes paragraph: "In this video, your child will learn..."
  4. Parent/teacher note (2-3 sentences about educational value)
  5. Subscribe CTA: encourage subscribing for more episodes
  6. Hashtag block at the end: 10-15 relevant hashtags
  7. Maximum 5000 characters total
  8. Safe, child-friendly language throughout

IMPORTANT: Include these hashtags at the very end:
#LearnColors #KidsEducation #PreschoolLearning #ToddlerLearning #BoboKidsStudio

Output ONLY the description text. No extra commentary."""


def build_tags_prompt(script: dict) -> str:
    topic       = script.get("topic", "Learn Colors")
    learning    = script.get("learning_goal", "")
    target_age  = script.get("target_age", "3-5")

    return f"""You are an expert YouTube SEO specialist for children's educational content.

Generate a list of YouTube tags for this video.

VIDEO DETAILS:
  Topic: {topic}
  Learning Goal: {learning}
  Target Age: {target_age}
  Channel: BOBO KIDS STUDIO

TAG REQUIREMENTS:
  - Generate exactly 25-30 tags
  - Mix of: broad terms + specific terms + long-tail phrases
  - Each tag: 2-40 characters
  - Total character count of ALL tags combined must be under 480 characters
  - Include: character names, topic keywords, age keywords, competitor-adjacent
  - Examples of good tags: "learn colors", "colors for kids", "red color",
    "preschool learning", "toddler education", "bobo bear", "kids cartoons"

Output ONLY a comma-separated list of tags. No numbering, no bullets, no extra text.
Example format: learn colors, red color, colors for kids, toddler learning, ..."""


def build_short_metadata_prompt(
    short: dict, script: dict, short_number: int
) -> str:
    topic   = script.get("topic", "Learn Colors")
    channel = cfg.CHANNEL_NAME

    return f"""You are an expert YouTube Shorts SEO specialist for children's content.

Generate metadata for ONE YouTube Short.

SHORT DETAILS:
  Short Number: {short_number}
  Title: {short.get('title', '')}
  Hook: {short.get('hook', '')}
  Script: {short.get('script', '')}
  Duration: {short.get('duration_sec', 30)} seconds
  Parent video topic: {topic}
  Channel: {channel}

Generate:
1. SHORTS_TITLE: A catchy Shorts title (max 100 chars, include emoji)
2. SHORTS_DESCRIPTION: A Shorts description (max 500 chars, with relevant hashtags)

Format your response EXACTLY like this:
SHORTS_TITLE: [title here]
SHORTS_DESCRIPTION: [description here]

End description with: #Shorts #LearnColors #KidsEducation #BoboKidsStudio"""


# ── Metadata Generator class ──────────────────────────────────────────────────

class MetadataGenerator:
    """
    Generates complete YouTube metadata for a video using Gemini AI.

    USAGE:
      gen = MetadataGenerator()
      metadata = gen.generate(video_id)

    OUTPUT:
      output/<video_id>/metadata.json
    """

    def __init__(self):
        self.output_dir = cfg.PIPELINE_DIRS["output"]
        self._model = None

    def _model_lazy(self):
        """Lazily initialise Gemini model."""
        if self._model is None:
            self._model = _get_gemini_model()
        return self._model

    def _load_manifests(self, video_id: str) -> tuple:
        dirs = cfg.PIPELINE_DIRS
        script      = json.loads((dirs["scripts"] / video_id / "script.json").read_text("utf-8"))
        video_mf    = json.loads((self.output_dir / video_id / "video_manifest.json").read_text("utf-8"))
        shorts_mf   = json.loads((self.output_dir / video_id / "shorts_manifest.json").read_text("utf-8"))
        return script, video_mf, shorts_mf

    def generate(self, video_id: str, force_redo: bool = False) -> dict:
        """Generate complete metadata. Returns metadata dict."""
        out_dir = self.output_dir / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = out_dir / "metadata.json"

        if metadata_path.exists() and not force_redo:
            logger.info("Metadata cached. Loading.")
            return json.loads(metadata_path.read_text("utf-8"))

        script, video_mf, shorts_mf = self._load_manifests(video_id)
        model = self._model_lazy()
        topic = script.get("topic", "Learn Colors")

        def safe_print(text):
            sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()

        safe_print(f"  Generating metadata for: {topic}")
        print()

        # ── Main video title ──────────────────────────────────────────────────
        print("  [1/4] Generating title...")
        raw_title = _call_gemini(model, build_title_prompt(script))
        title = trim_to_limit(raw_title.strip().strip('"'), TITLE_MAX_CHARS)
        logger.info(f"Title ({len(title)} chars): {title}")
        safe_print(f"         -> '{title}'")
        time.sleep(15)   # 15s gap → stay under 5 RPM free tier

        # ── Main video description ────────────────────────────────────────────
        print("  [2/4] Generating description...")
        raw_desc = _call_gemini(model, build_description_prompt(script, video_mf))
        description = trim_to_limit(raw_desc, DESCRIPTION_MAX_CHARS)
        logger.info(f"Description: {len(description)} characters")
        print(f"         -> {len(description)} characters, {description.count(chr(10))} lines")
        time.sleep(15)   # 15s gap

        # ── Tags ──────────────────────────────────────────────────────────────
        print("  [3/4] Generating tags...")
        raw_tags = _call_gemini(model, build_tags_prompt(script))
        tags = parse_tags_from_response(raw_tags)
        tags = enforce_tag_limit(tags)
        tags_char_count = sum(len(t) for t in tags) + max(0, len(tags) - 1) * 2
        logger.info(f"Tags: {len(tags)} tags, {tags_char_count} chars")
        print(f"         -> {len(tags)} tags ({tags_char_count} chars)")
        time.sleep(15)   # 15s gap

        # ── Shorts metadata (generated locally — no extra Gemini call) ───────
        # WHY LOCAL GENERATION?
        #   The script.json already contains all the creative content for each
        #   Short: title, hook, caption_text, ending. Using these directly:
        #   - Saves 2 Gemini requests (avoids free-tier rate limit)
        #   - Is deterministic (no random variation between runs)
        #   - Is faster (~0s vs ~15s per Short)
        #   The Gemini calls are reserved for the more nuanced tasks:
        #   title phrasing, description writing, and tag research.
        print("  [4/4] Building Shorts metadata from script data...")
        shorts_metadata = []
        script_shorts = script.get("shorts", [])
        channel = cfg.CHANNEL_NAME

        for short_data in script_shorts:
            snum        = short_data.get("short_number", 1)
            s_title     = short_data.get("title", f"Short {snum}")
            hook        = short_data.get("hook", "")
            caption     = short_data.get("caption_text", "")
            ending      = short_data.get("ending", f"Subscribe to {channel}!")
            script_txt  = short_data.get("script", "")
            duration    = short_data.get("duration_sec", 30)

            # Build a compact, emoji-rich Shorts title
            # Formula: [Emoji] + [hook phrase] | [channel]
            short_title = trim_to_limit(
                f"🔴 {s_title} | {channel}",
                TITLE_MAX_CHARS,
            )

            # Build Shorts description: hook + script snippet + CTA + hashtags
            short_desc = (
                f"{hook}\n\n"
                f"{script_txt[:200].rstrip()}...\n\n"
                f"{ending}\n\n"
                f"#Shorts #LearnColors #KidsEducation #ToddlerLearning "
                f"#BoboKidsStudio #Preschool #{caption.replace(' ', '').replace('!', '')}"
            )
            short_desc = trim_to_limit(short_desc.strip(), 500)

            shorts_metadata.append({
                "short_number":  snum,
                "title":         short_title,
                "description":   short_desc,
                "original_hook": hook,
            })
            safe_print(f"         -> Short {snum}: '{short_title}'")

        # ── Build final metadata dict ─────────────────────────────────────────
        # YouTube category 27 = Education
        # made_for_kids = True: required by YouTube for content targeting under 13
        # This triggers COPPA compliance mode (no personalised ads shown to kids)
        metadata = {
            "video_id": video_id,

            # Main video
            "main_video": {
                "title":            title,
                "description":      description,
                "tags":             tags,
                "category_id":      YOUTUBE_CATEGORY_EDUCATION,
                "made_for_kids":    True,
                "default_language": "en",
                "privacy_status":   "private",   # Start private → Phase 11 review → public
                "file":             str(self.output_dir / video_id / "main_video.mp4"),
                "thumbnail_file":   str(cfg.PIPELINE_DIRS["thumbnails"] / video_id / "thumbnail_final.jpg"),
            },

            # Shorts
            "shorts": [
                {
                    **sm,
                    "category_id":      YOUTUBE_CATEGORY_EDUCATION,
                    "made_for_kids":    True,
                    "default_language": "en",
                    "privacy_status":   "private",
                    "file":             str(self.output_dir / video_id / f"short_{sm['short_number']:02d}.mp4"),
                }
                for sm in shorts_metadata
            ],

            # Stats
            "_meta": {
                "title_chars":       len(title),
                "description_chars": len(description),
                "tags_count":        len(tags),
                "tags_chars":        tags_char_count,
                "shorts_count":      len(shorts_metadata),
                "title_within_limit":       len(title) <= TITLE_MAX_CHARS,
                "description_within_limit": len(description) <= DESCRIPTION_MAX_CHARS,
                "tags_within_limit":        tags_char_count <= TAGS_MAX_CHARS,
            },
        }

        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Metadata saved: {metadata_path}")
        return metadata


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = sys.argv[1:]
    force_redo = "--force-redo" in args
    args = [a for a in args if not a.startswith("--")]
    video_id = args[0] if args else "20260924_learn_colors_red"

    sys.stderr.write(f"\nPhase 10 -- Metadata Generation\n")
    sys.stderr.write(f"Video ID: {video_id}\n")

    gen = MetadataGenerator()
    meta = gen.generate(video_id, force_redo=force_redo)
    mv = meta["main_video"]

    sys.stderr.write(f"\n=== Metadata Generation Complete ===\n\n")
    sys.stderr.write(f"  Title ({len(mv['title'])} chars):\n  '{mv['title']}'\n\n")
    sys.stderr.write(f"  Tags ({meta['_meta']['tags_count']} tags, {meta['_meta']['tags_chars']} chars):\n")
    sys.stderr.write(f"  {', '.join(mv['tags'][:8])}...\n\n")
    sys.stderr.write(f"  Description: {meta['_meta']['description_chars']} characters\n\n")
    sys.stderr.write(f"  Shorts:\n")
    for s in meta["shorts"]:
        sys.stderr.write(f"    Short {s['short_number']:02d}: '{s['title']}'\n")
    sys.stderr.write(f"\n  All within YouTube limits: "
                     f"Title={'OK' if meta['_meta']['title_within_limit'] else 'OVER'} | "
                     f"Desc={'OK' if meta['_meta']['description_within_limit'] else 'OVER'} | "
                     f"Tags={'OK' if meta['_meta']['tags_within_limit'] else 'OVER'}\n\n")
    sys.stderr.write(f"  Saved: {cfg.PIPELINE_DIRS['output'] / video_id / 'metadata.json'}\n\n")
