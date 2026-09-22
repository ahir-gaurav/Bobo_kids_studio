"""
pipeline/character_loader.py \u2014 BOBO KIDS STUDIO
=================================================
Loads character definitions from characters/<name>/character.json
and provides a clean Python interface for the rest of the pipeline.

WHY THIS MODULE EXISTS:
  Every pipeline stage (script gen, voice, image gen, video assembly)
  needs character data. Instead of each stage opening JSON files
  independently, they all import from here. One load, consistent access.

USAGE:
  from pipeline.character_loader import CharacterLoader

  loader = CharacterLoader()
  bobo = loader.get("bobo")
  print(bobo.name)            # "Bobo"
  print(bobo.voice_speed)     # 0.9
  print(bobo.sd_prompt)       # "cute bear cub character..."
  print(bobo.catchphrase)     # "Let'\''s learn something amazing today!"

  all_chars = loader.all()    # list of all 3 Character objects
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline import get_logger

logger = get_logger("character_loader")


@dataclass
class Character:
    """
    A single character\''s full config loaded from character.json.

    WHY A DATACLASS?
      Dataclasses are like regular Python classes but auto-generate
      __init__, __repr__ and __eq__ for free. They\''re perfect for
      data containers like this. We\''re not doing heavy processing here,
      just structured storage with type hints.
    """
    # Identity
    name: str
    slug: str                   # folder name, e.g. "bobo"
    species: str
    role: str                   # "lead", "best_friend", "music_buddy"
    age_appearance: str
    catchphrase: str

    # Visual
    appearance: dict            # raw dict from JSON
    color_palette: dict         # {"primary": hex, "secondary": hex, ...}
    sd_prompt: str              # Stable Diffusion generation prompt
    sd_negative_prompt: str

    # Voice
    voice_engine: str           # "kokoro"
    voice_id: str               # model voice ID
    voice_speed: float
    voice_pitch_semitones: int
    voice_notes: str

    # Filesystem
    character_dir: Path         # e.g. Path("characters/bobo")
    assets_dir: Path            # e.g. Path("characters/bobo/assets")
    asset_files: dict           # {"face_happy": "assets/face_happy.png", ...}

    # Personality list
    personality: list[str] = field(default_factory=list)

    def get_asset_path(self, asset_key: str) -> Optional[Path]:
        """
        Returns the full Path to a character asset file, or None if missing.

        Parameters
        ----------
        asset_key : str
            Key from asset_files dict, e.g. "face_happy", "body_front"

        Returns
        -------
        Path or None
        """
        relative = self.asset_files.get(asset_key)
        if not relative:
            logger.warning(f"Asset key \'{asset_key}\' not found for character {self.name}")
            return None

        # asset_files stores paths relative to the character dir
        full_path = self.character_dir / relative
        if not full_path.exists():
            logger.warning(f"Asset file not yet created: {full_path}")
            return None
        return full_path

    def __repr__(self) -> str:
        return f"Character(name={self.name!r}, role={self.role!r}, voice={self.voice_id!r})"


class CharacterLoader:
    """
    Scans the characters/ directory and loads all character.json files.

    DESIGN DECISION \u2014 why scan at runtime instead of hardcoding?
      If you add a 4th character in Phase 2B, you just drop a new folder
      into characters/ with a character.json. The loader picks it up
      automatically. No code changes needed.
    """

    def __init__(self, characters_dir: Optional[Path] = None):
        """
        Parameters
        ----------
        characters_dir : Path, optional
            Path to the characters/ directory.
            Defaults to <project_root>/characters/
        """
        from config import cfg
        self._dir = characters_dir or cfg.PIPELINE_DIRS["characters"]
        self._characters: dict[str, Character] = {}
        self._load_all()

    def _load_all(self):
        """Walk characters/ and load every valid character.json."""
        if not self._dir.exists():
            logger.error(f"Characters directory not found: {self._dir}")
            return

        count = 0
        for char_dir in sorted(self._dir.iterdir()):
            if not char_dir.is_dir():
                continue

            json_path = char_dir / "character.json"
            if not json_path.exists():
                logger.warning(f"No character.json in {char_dir.name}/ \u2014 skipping")
                continue

            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                char = self._parse(char_dir.name, char_dir, data)
                self._characters[char_dir.name] = char
                logger.info(f"Loaded character: {char.name} ({char.role})")
                count += 1
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to load {json_path}: {e}")

        logger.info(f"Total characters loaded: {count}")

    def _parse(self, slug: str, char_dir: Path, data: dict) -> Character:
        """
        Parse a character.json dict into a Character dataclass.

        WHY SEPARATE PARSE METHOD?
          If the JSON schema changes (e.g. you add a new field), you
          only update this one method. The rest of the code stays clean.
        """
        voice = data.get("voice", {})
        return Character(
            name=data["name"],
            slug=slug,
            species=data.get("species", "unknown"),
            role=data.get("role", "supporting"),
            age_appearance=data.get("age_appearance", ""),
            catchphrase=data.get("catchphrase", ""),
            appearance=data.get("appearance", {}),
            color_palette=data.get("color_palette", {}),
            sd_prompt=data.get("sd_prompt", ""),
            sd_negative_prompt=data.get("sd_negative_prompt", ""),
            voice_engine=voice.get("engine", "kokoro"),
            voice_id=voice.get("voice_id", "af_heart"),
            voice_speed=float(voice.get("speed", 1.0)),
            voice_pitch_semitones=int(voice.get("pitch_shift_semitones", 0)),
            voice_notes=voice.get("notes", ""),
            character_dir=char_dir,
            assets_dir=char_dir / "assets",
            asset_files=data.get("asset_files", {}),
            personality=data.get("personality", []),
        )

    def get(self, slug: str) -> Optional[Character]:
        """
        Get a specific character by folder slug.

        Example: loader.get("bobo") returns the Bobo Character object.
        Returns None if not found (with a warning log).
        """
        char = self._characters.get(slug)
        if char is None:
            logger.warning(f"Character \'{slug}\' not found. Available: {list(self._characters.keys())}")
        return char

    def all(self) -> list[Character]:
        """Return all loaded characters as a list."""
        return list(self._characters.values())

    def lead(self) -> Optional[Character]:
        """Convenience method \u2014 returns the lead character (Bobo)."""
        for char in self._characters.values():
            if char.role == "lead":
                return char
        return None

    def by_role(self, role: str) -> list[Character]:
        """Return all characters with a given role."""
        return [c for c in self._characters.values() if c.role == role]