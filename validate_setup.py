"""
validate_setup.py — BOBO KIDS STUDIO
======================================
Run this after Phase 1 setup to confirm everything is working.

USAGE:
  python validate_setup.py

EXPECTED OUTPUT:
  A table of checks, all showing [PASS] in green.
  If any show [FAIL], read the error message and fix it before Phase 2.
"""

import sys
import os
import json
from pathlib import Path


def check(label: str, condition: bool, detail: str = "") -> bool:
    """Print a coloured pass/fail line and return the result."""
    # ANSI colour codes — work in Windows Terminal and PowerShell
    GREEN = "\033[92m"
    RED   = "\033[91m"
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    status = f"{GREEN}[PASS]{RESET}" if condition else f"{RED}[FAIL]{RESET}"
    line = f"  {status}  {BOLD}{label}{RESET}"
    if detail:
        line += f"\n         {detail}"
    print(line)
    return condition


def main():
    print()
    print("=" * 60)
    print("  BOBO KIDS STUDIO — Phase 1 Setup Validator")
    print("=" * 60)

    results = []

    # ── 1. Python version ────────────────────────────────────────────────────
    # We need Python 3.10+ for modern type hints and match statements.
    # 3.14 is what you have, which is even better.
    major, minor = sys.version_info.major, sys.version_info.minor
    results.append(check(
        "Python version >= 3.10",
        major == 3 and minor >= 10,
        detail=f"Found: Python {major}.{minor}.{sys.version_info.micro}"
    ))

    # ── 2. Required directories ──────────────────────────────────────────────
    root = Path(__file__).parent
    required_dirs = [
        "pipeline", "characters/bobo", "characters/bobo/assets",
        "assets/music", "assets/sfx", "assets/fonts",
        "assets/backgrounds", "scripts", "audio", "output",
        "db", "logs", "dashboard", "tests"
    ]
    all_dirs_ok = True
    for d in required_dirs:
        path = root / d
        if not path.exists():
            results.append(check(f"Directory: {d}", False, detail=f"Missing: {path}"))
            all_dirs_ok = False
    if all_dirs_ok:
        results.append(check("All required directories", True,
                             detail=f"Checked {len(required_dirs)} directories"))

    # ── 3. Required files ────────────────────────────────────────────────────
    required_files = [
        ".gitignore", ".env", ".env.template", "requirements.txt",
        "config.py", "pipeline/__init__.py",
        "characters/bobo/character.json",
        "assets/music/license_ledger.csv"
    ]
    for f in required_files:
        path = root / f
        results.append(check(f"File: {f}", path.exists()))

    # ── 4. .env has no committed secrets ────────────────────────────────────
    # We check that .env.template exists and .env contains the placeholder
    # (i.e., the user hasn't filled in a real key yet — that's fine for Phase 1)
    env_path = root / ".env"
    if env_path.exists():
        env_content = env_path.read_text()
        results.append(check(
            ".env file exists",
            True,
            detail="Remember to add your GEMINI_API_KEY before Phase 3"
        ))
        # Make sure the real API key isn't accidentally the word "your_gemini..."
        has_placeholder = "your_gemini_api_key_here" in env_content
        real_key_set = "GEMINI_API_KEY=" in env_content and not has_placeholder
        if real_key_set:
            results.append(check(".env: GEMINI_API_KEY is set", True))
        else:
            results.append(check(
                ".env: GEMINI_API_KEY is set",
                False,
                detail="Still showing placeholder — fill this in before Phase 3"
            ))

    # ── 5. character.json is valid JSON ─────────────────────────────────────
    char_path = root / "characters" / "bobo" / "character.json"
    if char_path.exists():
        try:
            data = json.loads(char_path.read_text(encoding="utf-8"))
            results.append(check(
                "character.json is valid JSON",
                True,
                detail=f"Character name: {data.get('name', 'unknown')}"
            ))
        except json.JSONDecodeError as e:
            results.append(check("character.json is valid JSON", False, detail=str(e)))

    # ── 6. Python packages (only Phase 1 packages) ──────────────────────────
    phase1_packages = {
        "dotenv":    "python-dotenv",
        "loguru":    "loguru",
        "requests":  "requests",
        "tqdm":      "tqdm",
        "PIL":       "Pillow",
        "rich":      "rich",
    }
    for import_name, pip_name in phase1_packages.items():
        try:
            __import__(import_name)
            results.append(check(f"Package: {pip_name}", True))
        except ImportError:
            results.append(check(
                f"Package: {pip_name}", False,
                detail=f"Run: python -m pip install {pip_name}"
            ))

    # ── 7. config.py loads correctly ─────────────────────────────────────────
    try:
        from config import cfg
        results.append(check(
            "config.py loads",
            True,
            detail=f"Channel: {cfg.CHANNEL_NAME} | Target age: {cfg.TARGET_AGE_MIN}-{cfg.TARGET_AGE_MAX} | FPS: {cfg.FPS}"
        ))
        dir_check = cfg.validate()
        results.append(check("config.py directory validation", dir_check))
    except Exception as e:
        results.append(check("config.py loads", False, detail=str(e)))

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for r in results if r)
    total = len(results)
    print()
    print("=" * 60)
    color = "\033[92m" if passed == total else "\033[93m"
    print(f"  {color}Result: {passed}/{total} checks passed\033[0m")
    if passed < total:
        print("  Fix the [FAIL] items above, then re-run this script.")
    else:
        print("  \033[92mPhase 1 complete! You are ready for Phase 2.\033[0m")
    print("=" * 60)
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
