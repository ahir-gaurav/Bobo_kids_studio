import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.character_loader import CharacterLoader

GREEN = "\033[92m"; RED = "\033[91m"; RESET = "\033[0m"; BOLD = "\033[1m"

def check(label, ok, detail=""):
    s = f"{GREEN}[PASS]{RESET}" if ok else f"{RED}[FAIL]{RESET}"
    print(f"  {s}  {BOLD}{label}{RESET}")
    if detail: print(f"         {detail}")
    return ok

results = []
print()
print("=" * 55)
print("  BOBO KIDS STUDIO -- Phase 2 Character Test")
print("=" * 55)

loader = CharacterLoader()
chars = loader.all()

results.append(check("3 characters loaded", len(chars) == 3, f"Found: {[c.name for c in chars]}"))
results.append(check("Lead character is Bobo", loader.lead() is not None and loader.lead().name == "Bobo"))

for c in chars:
    results.append(check(f"{c.name}: has catchphrase", bool(c.catchphrase), detail=c.catchphrase))
    results.append(check(f"{c.name}: has SD prompt", len(c.sd_prompt) > 20))
    results.append(check(f"{c.name}: voice_speed valid", 0.5 <= c.voice_speed <= 2.0, f"speed={c.voice_speed}"))
    results.append(check(f"{c.name}: color_palette has 3 keys", len(c.color_palette) == 3))

sg_path = Path("assets/style_guide.json")
results.append(check("style_guide.json exists", sg_path.exists()))
if sg_path.exists():
    sg = json.loads(sg_path.read_text(encoding="utf-8"))
    results.append(check("Style guide: target age 3-5",
        sg["channel"]["target_age_min"] == 3 and sg["channel"]["target_age_max"] == 5))
    results.append(check("Style guide: max 8 words/sentence",
        sg["age_band_language_rules"]["max_words_per_sentence"] == 8))

passed = sum(results); total = len(results)
print()
print("=" * 55)
color = GREEN if passed == total else "\033[93m"
print(f"  {color}Result: {passed}/{total} checks passed{RESET}")
if passed == total:
    print(f"  {GREEN}Phase 2 complete! Ready for Phase 3.{RESET}")
print("=" * 55)
print()
sys.exit(0 if passed == total else 1)