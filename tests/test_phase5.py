"""
tests/test_phase5.py — Phase 5 Scene Generation Validation
===========================================================
Tests the scene generator using a mock engine for speed,
then verifies real engine availability.
"""

import json
import sys
import struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.scene_generator import (
    SceneGenerator,
    PollinationsEngine,
    PillowFallbackEngine,
    build_scene_prompt,
    get_best_engine,
    STYLE_SUFFIX,
)
from config import cfg

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


class MockImageEngine:
    """
    Fake engine that writes a 1x1 white PNG instantly.
    Used to test orchestration logic without network calls.
    """
    def is_available(self) -> bool:
        return True

    def generate(self, prompt, width, height, output_path) -> bool:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal valid PNG: 1x1 white pixel
        # Header + IHDR + IDAT + IEND
        import zlib, struct
        def chunk(name, data):
            c = name + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c))
        png = (
            b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\x00\xff\xff\xff'))
            + chunk(b'IEND', b'')
        )
        output_path.write_bytes(png)
        return True

    def get_best_engine(self):
        return self


def check(label, ok, detail=""):
    sym = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    print(f"  {sym}  {BOLD}{label}{RESET}")
    if detail:
        print(f"         {detail}")
    return ok


def run_tests():
    results = []
    print()
    print("=" * 58)
    print("  BOBO KIDS STUDIO -- Phase 5 Scene Generator Tests")
    print("=" * 58)

    # ── 1. Config ────────────────────────────────────────────────────────────
    print(f"\n{BOLD}[1] Config & Directory Setup{RESET}")
    results.append(check("'scenes' key in PIPELINE_DIRS", "scenes" in cfg.PIPELINE_DIRS))
    results.append(check("'thumbnails' key in PIPELINE_DIRS", "thumbnails" in cfg.PIPELINE_DIRS))

    # ── 2. Prompt builder ────────────────────────────────────────────────────
    print(f"\n{BOLD}[2] Prompt Builder{RESET}")
    prompt = build_scene_prompt(
        "Bobo walks in the park and finds a red surprise box",
        ["bobo", "lumi"]
    )
    results.append(check("Prompt is non-empty", len(prompt) > 10))
    results.append(check("Prompt contains style suffix", STYLE_SUFFIX[:30] in prompt,
                          detail=f"prompt={prompt[:80]}..."))
    results.append(check("Bobo description in prompt", "brown bear" in prompt.lower()))
    results.append(check("Lumi description in prompt", "firefly" in prompt.lower()))

    # ── 3. Mock synthesis ────────────────────────────────────────────────────
    print(f"\n{BOLD}[3] Scene Generation (Mock Engine){RESET}")
    script_path = cfg.PIPELINE_DIRS["scripts"] / "20260924_learn_colors_red" / "script.json"
    results.append(check("Script file exists", script_path.exists(), str(script_path)))

    if script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        mock_engine = MockImageEngine()
        gen = SceneGenerator(engine=mock_engine)
        manifest = gen.generate_scenes(script, "20260924_learn_colors_red_test")

        results.append(check("Manifest generated", isinstance(manifest, dict)))
        results.append(check("Manifest has scenes key", "scenes" in manifest))
        results.append(check(
            f"All 8 scenes in manifest (got {len(manifest.get('scenes', []))})",
            len(manifest.get("scenes", [])) == 8
        ))

        # Check PNG files exist on disk
        test_dir = cfg.PIPELINE_DIRS["scenes"] / "20260924_learn_colors_red_test"
        pngs = list(test_dir.glob("*.png"))
        results.append(check(f"PNG files created ({len(pngs)} files)", len(pngs) == 8))

        # Verify manifest saved to disk
        manifest_path = test_dir / "scenes_manifest.json"
        results.append(check("scenes_manifest.json saved to disk", manifest_path.exists()))

        # Verify each scene has required fields
        s0 = manifest["scenes"][0]
        for field in ["scene_number", "title", "file", "duration_sec", "prompt_used"]:
            results.append(check(f"Scene has '{field}' field", field in s0))

    # ── 4. Engine availability ───────────────────────────────────────────────
    print(f"\n{BOLD}[4] Engine Availability{RESET}")
    p_engine = PollinationsEngine()
    pillow = PillowFallbackEngine()

    p_ok = p_engine.is_available()
    pillow_ok = pillow.is_available()

    results.append(check("Pillow fallback available", pillow_ok))
    print(f"         {YELLOW}Pollinations available: {p_ok}{RESET}")

    best = get_best_engine()
    results.append(check(f"Best engine: {type(best).__name__}", best is not None))

    # ── Results ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print()
    print("=" * 58)
    color = GREEN if passed == total else YELLOW
    print(f"  {color}Result: {passed}/{total} checks passed{RESET}")
    if passed == total:
        print(f"  {GREEN}Phase 5 infrastructure ready!{RESET}")
    print("=" * 58)
    print()
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
