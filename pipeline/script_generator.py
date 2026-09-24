"""
pipeline/script_generator.py — BOBO KIDS STUDIO
=================================================
Stage 1 of the pipeline: Topic → Structured JSON Script.

WHAT THIS MODULE DOES:
  1. Reads the channel style guide and character definitions
  2. Builds a very detailed prompt for Gemini Flash
  3. Calls the Gemini API and asks for a JSON-formatted script
  4. Validates the response (is it real JSON? does it have scenes?)
  5. Saves the result to scripts/<video_id>/script.json
  6. Returns the script dict so the next stage can use it

WHY IS THE PROMPT SO DETAILED?
  LLMs are probabilistic — the same model gives different outputs
  every run. A detailed prompt with explicit constraints (age band,
  word count, scene count, JSON schema) dramatically reduces variance.
  Think of it as "programming with English" — more specific = more
  reliable output.

WHY JSON OUTPUT FROM AN LLM?
  The LLM returns text. We need structured data. We instruct it to
  output ONLY valid JSON, then parse it with json.loads(). If it fails
  to parse, we retry (up to 3 times) with a stricter extraction step.

RETRY STRATEGY:
  API call fails → wait 2^attempt seconds → retry (exponential backoff)
  This is the standard pattern for any network call: don't hammer a
  failing API, give it time to recover.
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from config import cfg
from pipeline import get_logger
from pipeline.character_loader import CharacterLoader


# ── Module-level logger ─────────────────────────────────────────────────────
# We set topic_slug="general" here; once we have a topic we'll re-init.
logger = get_logger("script_generator")

# ── Gemini model configuration ──────────────────────────────────────────────
MODEL_NAME = "gemini-3.6-flash"   # Updated 2026-09-24: 2.0-flash deprecated; 3.6-flash is current free-tier Flash
MAX_RETRIES = 3
RETRY_WAIT_BASE = 2   # seconds; we wait 2^attempt: 2s, 4s, 8s


def _build_system_prompt(style_guide: dict, characters: list) -> str:
    """
    Build the SYSTEM prompt — the persistent instructions Gemini follows
    for the entire conversation.

    WHY A SYSTEM PROMPT?
      The system prompt sets the AI's "role" and "rules" before the user
      message. It's like briefing a writer before they start. The user
      message (below) then gives the specific assignment.
      Gemini treats system prompts with higher authority than user turns.
    """
    char_summaries = []
    for c in characters:
        char_summaries.append(
            f"  - {c.name} ({c.species}, {c.role}): {'; '.join(c.personality[:3])}. "
            f"Catchphrase: '{c.catchphrase}'"
        )
    chars_text = "\n".join(char_summaries)

    age_rules = style_guide.get("age_band_language_rules", {})
    max_words = age_rules.get("max_words_per_sentence", 8)

    return f"""You are a professional children's educational video scriptwriter for BOBO KIDS STUDIO.

CHANNEL: {style_guide['channel']['name']}
TAGLINE: {style_guide['channel']['tagline']}
TARGET AGE: {style_guide['channel']['target_age_min']}–{style_guide['channel']['target_age_max']} years old

RECURRING CHARACTERS:
{chars_text}

STRICT LANGUAGE RULES (age {style_guide['channel']['target_age_min']}–{style_guide['channel']['target_age_max']}):
- Maximum {max_words} words per sentence in narration. No exceptions.
- Use concrete nouns, not abstract concepts.
- No metaphors. Children this age take language literally.
- Repeat the key learning word at least 5 times naturally.
- Include the child directly at least twice: 'Can YOU find...?', 'Now YOU try!'
- Pause cues: write [PAUSE 2s] after any question directed at the child.

CONTENT RULES:
- ONE learning goal per video. Not 'colors and shapes' — just ONE color.
- No violence, fear, dark themes, adult references, or unsafe behaviors.
- No imitation of CoComelon, Baby Shark, Bluey, or any existing IP.
- Original story, original characters, original dialogue only.
- Age-appropriate: gentle mistakes, kind resolution, happy ending.

VIDEO STRUCTURE (5–7 minutes, 30 FPS, 1920×1080):
1. Hook (0:00–0:15): Bobo shows/finds something surprising. Asks a question.
2. Intro Song (0:15–0:35): Short 4-bar catchy chant. Include lyrics.
3. Scene 1 (0:35–1:30): Introduce the concept. One character explores.
4. Scene 2 (1:30–2:30): A friend joins (Lumi or Pip). Together they discover.
5. Scene 3 (2:30–3:30): Small challenge or mistake. Learning moment.
6. Scene 4 (3:30–4:30): Success! Celebrate with a song or chant.
7. Question (4:30–4:45): Direct child participation. [PAUSE 2s] included.
8. Outro (4:45–5:00): Bobo waves goodbye, teases next episode.

EACH SCENE MUST SPECIFY:
- scene_number, title, duration_sec
- visual: description of what is shown on screen
- narration: the voiceover text (narrator speaks these words)
- dialogue: list of {{"character": name, "line": text}} objects
- music: one of [gentle_curious, playful_adventure, calm_learning, celebration, lullaby]
- sfx: list of sound effect names (e.g. ["bird_chirp", "tada"])
- animation: camera/movement note (e.g. "slow zoom in", "pan left to right")
- transition: one of [fade, cut, wipe]

SHORTS (2 Shorts, each 20–50 seconds):
Each Short must be a STANDALONE piece with its own beginning, middle, end.
Not a random clip — a mini-story derived from the long video.

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown, no code fences, no explanation text.
Just the raw JSON starting with {{ and ending with }}.
The JSON must be parseable by Python's json.loads() with no modifications.
"""


def _build_user_prompt(topic: str, video_id: str, style_guide: dict) -> str:
    """
    Build the USER prompt — the specific assignment for this video.

    WHY SEPARATE SYSTEM AND USER PROMPTS?
      System = standing rules (same for all videos).
      User = specific task (changes per video).
      This separation lets us reuse the system prompt across all videos
      while varying only the topic. Cleaner, more maintainable.
    """
    topics_done = style_guide.get("_topic_history", [])
    history_note = ""
    if topics_done:
        history_note = f"\nAVOID OVERLAP WITH THESE RECENT TOPICS: {', '.join(topics_done[-10:])}"

    return f"""Write a complete children's educational video script for BOBO KIDS STUDIO.

TOPIC: {topic}
VIDEO ID: {video_id}
{history_note}

Return the script as a single JSON object with this exact structure:

{{
  "video_id": "{video_id}",
  "topic": "<specific learning goal, e.g. Learn Colors: Red>",
  "target_age": "3-5",
  "learning_goal": "<one sentence: what the child learns>",
  "total_duration_sec": <number between 300 and 420>,
  "characters": ["bobo", "<lumi or pip>"],
  "intro_song": {{
    "lyrics": "<4–8 lines of the catchy intro chant>",
    "duration_sec": 20
  }},
  "scenes": [
    {{
      "scene_number": 1,
      "title": "<short title like 'The Hook'>",
      "duration_sec": <number>,
      "visual": "<what is on screen, described for an artist>",
      "narration": "<voiceover text, max {style_guide['age_band_language_rules']['max_words_per_sentence']} words/sentence>",
      "dialogue": [
        {{"character": "bobo", "line": "<what Bobo says>"}}
      ],
      "music": "<one of: gentle_curious, playful_adventure, calm_learning, celebration, lullaby>",
      "sfx": ["<sound1>", "<sound2>"],
      "animation": "<camera/movement instruction>",
      "transition": "<fade|cut|wipe>"
    }}
  ],
  "shorts": [
    {{
      "short_number": 1,
      "title": "<catchy short title>",
      "hook": "<first line — must grab attention in 2 seconds>",
      "source_scenes": [<scene numbers this is derived from>],
      "script": "<the complete short script, max 60 words>",
      "duration_sec": <number between 20 and 50>,
      "ending": "<how it ends — call to action or punchline>",
      "caption_text": "<on-screen text for this short>"
    }}
  ],
  "metadata": {{
    "title_options": [
      "<option 1: emoji + topic + benefit>",
      "<option 2: question format>",
      "<option 3: character name + topic>"
    ],
    "recommended_title": "<your top pick and why in 1 sentence>",
    "description": "<YouTube description, 3–4 sentences, age-appropriate>",
    "tags": ["<tag1>", "<tag2>", "<tag3>", "<up to 15 tags>"],
    "shorts_title": "<title for the Shorts version>",
    "shorts_description": "<shorter description for Shorts>"
  }}
}}

IMPORTANT REMINDERS:
- Narration sentences: max {style_guide['age_band_language_rules']['max_words_per_sentence']} words each.
- Include [PAUSE 2s] after every question directed at the child.
- The learning word must appear at least 5 times across all scenes.
- Write 8 scenes total (Hook + Intro Song + 4 story scenes + Question + Outro).
- Write exactly 2 Shorts.
- Return ONLY the JSON. No extra text.
"""


def _extract_json_from_response(text: str) -> Optional[dict]:
    """
    Attempt to parse JSON from the LLM response.

    WHY IS THIS NECESSARY?
      Even with explicit instructions, LLMs sometimes add:
        - Markdown code fences: ```json ... ```
        - Leading explanation text: "Here is your script: {...}"
        - Trailing commentary: "{...} Hope this helps!"
      We try three increasingly aggressive strategies to find the JSON.

    Parameters
    ----------
    text : str
        Raw text response from the Gemini API.

    Returns
    -------
    dict or None
    """
    # Strategy 1: Direct parse (works if response is clean JSON)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract content between first { and last }
    # This handles leading/trailing explanation text
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace: last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Strip markdown code fences and retry
    # Pattern: ```json ... ``` or ``` ... ```
    fenced = re.sub(r"```(?:json)?\s*", "", text)
    fenced = re.sub(r"```", "", fenced).strip()
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass

    return None


def _validate_script(script: dict) -> list[str]:
    """
    Validate the parsed script dict for required fields and basic sanity.

    Returns a list of error strings. Empty list = valid.

    WHY VALIDATE?
      The LLM might return valid JSON but missing a field we depend on
      (e.g. no "scenes" key). Better to catch this here and retry than
      to crash 3 stages later when the voice module tries to iterate scenes.
    """
    errors = []

    required_top = ["video_id", "topic", "target_age", "learning_goal",
                    "total_duration_sec", "characters", "scenes", "shorts", "metadata"]
    for field in required_top:
        if field not in script:
            errors.append(f"Missing top-level field: '{field}'")

    scenes = script.get("scenes", [])
    if len(scenes) < 5:
        errors.append(f"Too few scenes: got {len(scenes)}, expected at least 5")

    for i, scene in enumerate(scenes):
        required_scene = ["scene_number", "duration_sec", "narration", "dialogue",
                          "visual", "music", "sfx", "animation", "transition"]
        for field in required_scene:
            if field not in scene:
                errors.append(f"Scene {i+1} missing field: '{field}'")

    shorts = script.get("shorts", [])
    if len(shorts) < 1:
        errors.append("No shorts defined")

    metadata = script.get("metadata", {})
    if "title_options" not in metadata:
        errors.append("metadata missing 'title_options'")

    # Check total duration is sensible
    total_sec = script.get("total_duration_sec", 0)
    if not (240 <= total_sec <= 480):
        errors.append(f"total_duration_sec={total_sec} is outside 240–480s range")

    return errors


def generate_script(
    topic: str,
    characters: Optional[list] = None,
    force_regenerate: bool = False
) -> dict:
    """
    Main function: generate a complete video script for the given topic.

    Parameters
    ----------
    topic : str
        The learning topic, e.g. "Learn Colors: Red" or "Animals: Dog Sounds"

    characters : list of Character, optional
        Which characters to feature. Defaults to all characters.

    force_regenerate : bool
        If True, regenerate even if a cached script exists on disk.
        Default False — we reuse cached scripts to save API quota.

    Returns
    -------
    dict
        The complete script as a Python dict (same structure as script.json).

    Raises
    ------
    RuntimeError
        If the script cannot be generated after MAX_RETRIES attempts.
    """
    # ── Setup ────────────────────────────────────────────────────────────────
    # Create a URL-safe slug from the topic: "Learn Colors: Red" → "learn_colors_red"
    topic_slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    date_str = datetime.now().strftime("%Y%m%d")
    video_id = f"{date_str}_{topic_slug}"

    # Re-init logger with this specific topic so logs go to the right file
    global logger
    logger = get_logger("script_generator", topic_slug)
    logger.info(f"Starting script generation | topic='{topic}' | video_id='{video_id}'")

    # ── Check for cached script ──────────────────────────────────────────────
    output_dir = cfg.PIPELINE_DIRS["scripts"] / video_id
    output_path = output_dir / "script.json"

    if output_path.exists() and not force_regenerate:
        logger.info(f"Cached script found at {output_path}. Loading from disk.")
        logger.info("Use force_regenerate=True to regenerate.")
        return json.loads(output_path.read_text(encoding="utf-8"))

    # ── Load dependencies ────────────────────────────────────────────────────
    style_guide_path = cfg.PIPELINE_DIRS["assets"] / "style_guide.json"
    style_guide = json.loads(style_guide_path.read_text(encoding="utf-8"))

    if characters is None:
        loader = CharacterLoader()
        characters = loader.all()

    logger.info(f"Characters in script: {[c.name for c in characters]}")

    # ── Configure Gemini ─────────────────────────────────────────────────────
    # WHY configure here and not at module level?
    #   Configuring at module level would fail if GEMINI_API_KEY is not set
    #   (e.g. during testing). Configuring inside the function means it only
    #   runs when actually needed.
    genai.configure(api_key=cfg.GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=_build_system_prompt(style_guide, characters),
        generation_config=genai.GenerationConfig(
            # WHY temperature=0.7?
            #   0.0 = completely deterministic (boring, repetitive)
            #   1.0 = very creative but sometimes incoherent
            #   0.7 = good balance: creative enough for varied stories,
            #         constrained enough for consistent JSON structure
            temperature=0.7,

            # WHY max_output_tokens=8192?
            #   A full 8-scene script with 2 shorts and metadata is ~2000–4000
            #   tokens. 8192 gives plenty of headroom without wasteful cost.
            max_output_tokens=8192,
            # NOTE: frequency_penalty not supported on gemini-3.6-flash (2026-09-24)
        )
    )

    # ── API call with retry loop ─────────────────────────────────────────────
    user_prompt = _build_user_prompt(topic, video_id, style_guide)
    script = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"Gemini API call attempt {attempt}/{MAX_RETRIES}")
        try:
            response = model.generate_content(user_prompt)
            raw_text = response.text
            logger.debug(f"Raw response length: {len(raw_text)} chars")

            # Try to parse JSON from the response
            parsed = _extract_json_from_response(raw_text)
            if parsed is None:
                raise ValueError("Response did not contain parseable JSON")

            # Validate the parsed script
            errors = _validate_script(parsed)
            if errors:
                error_summary = "; ".join(errors[:3])
                raise ValueError(f"Script validation failed: {error_summary}")

            # ── All good ─────────────────────────────────────────────────────
            script = parsed
            logger.info(f"Script generated successfully on attempt {attempt}")
            logger.info(
                f"Scenes: {len(script['scenes'])} | "
                f"Shorts: {len(script['shorts'])} | "
                f"Duration: {script['total_duration_sec']}s"
            )
            break  # Exit the retry loop

        except Exception as e:
            last_error = e
            wait_time = RETRY_WAIT_BASE ** attempt
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {MAX_RETRIES} attempts failed. Last error: {e}")

    if script is None:
        raise RuntimeError(
            f"Script generation failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ── Save to disk ─────────────────────────────────────────────────────────
    # WHY save before returning?
    #   If the next stage crashes, you don't want to re-call the API.
    #   The cached file means: fix the crash → restart from here, not from scratch.
    output_dir.mkdir(parents=True, exist_ok=True)
    utf8_no_bom = "utf-8"
    with open(output_path, "w", encoding=utf8_no_bom) as f:
        json.dump(script, f, indent=2, ensure_ascii=False)

    logger.info(f"Script saved to: {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return script


def print_script_summary(script: dict) -> None:
    """
    Print a human-readable summary of the script to the terminal.
    Used for quick review before proceeding to the voice stage.
    Windows-safe: forces UTF-8 output to avoid cp1252 emoji errors.
    """
    import sys, io
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box

    # WHY this workaround?
    #   Windows cmd uses cp1252 encoding by default.
    #   Rich's emoji (🔴, ✨) can't be encoded in cp1252, causing a crash.
    #   Wrapping stdout in a UTF-8 TextIOWrapper fixes it.
    #   This is a display-only fix — the script.json is always UTF-8 clean.
    safe_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    console = Console(file=safe_stdout, highlight=False)

    console.print()
    console.print(Panel(
        f"[bold yellow]{script['topic']}[/]\n"
        f"[cyan]Learning goal:[/] {script['learning_goal']}\n"
        f"[cyan]Duration:[/] {script['total_duration_sec']}s "
        f"({script['total_duration_sec']//60}m {script['total_duration_sec']%60}s)\n"
        f"[cyan]Characters:[/] {', '.join(script['characters'])}",
        title="[bold green]BOBO KIDS STUDIO -- Script Preview[/]",
        border_style="green"
    ))

    # Scene table
    table = Table(box=box.ROUNDED, title="Scenes", show_lines=True)
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Title", style="bold")
    table.add_column("Duration", justify="right")
    table.add_column("Narration (first 60 chars)", style="italic")

    for scene in script["scenes"]:
        narration_preview = scene.get("narration", "")[:60]
        if len(scene.get("narration", "")) > 60:
            narration_preview += "..."
        table.add_row(
            str(scene["scene_number"]),
            scene.get("title", ""),
            f"{scene['duration_sec']}s",
            narration_preview
        )
    console.print(table)

    console.print()
    for short in script.get("shorts", []):
        console.print(
            f"  [bold magenta]Short {short['short_number']}:[/] "
            f"{short['title']} ({short['duration_sec']}s) -- "
            f"[italic]{short['hook']}[/]"
        )

    console.print()
    rec = script["metadata"].get("recommended_title", "")
    console.print(f"[bold]Recommended title:[/] {rec}")
    console.print()


# ── CLI entry point ──────────────────────────────────────────────────────────
# WHY THIS BLOCK?
#   if __name__ == "__main__" only runs when you call this file DIRECTLY
#   (python pipeline/script_generator.py), not when it's imported.
#   This lets you test the module standalone without running the full pipeline.
if __name__ == "__main__":
    import sys, os

    # Force UTF-8 output in Windows cmd / PowerShell
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Learn Colors: Red"
    sys.stderr.write(f"\nGenerating script for topic: '{topic}'\n")
    sys.stderr.write("This may take 5-15 seconds...\n\n")

    try:
        script = generate_script(topic)
        print_script_summary(script)
        output_path = (
            cfg.PIPELINE_DIRS["scripts"]
            / script["video_id"]
            / "script.json"
        )
        sys.stderr.write(f"\nScript saved to: {output_path}\n\n")
    except RuntimeError as e:
        sys.stderr.write(f"\n[ERROR] {e}\n")
        sys.exit(1)
