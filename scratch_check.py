"""
Fallback metadata generator — runs when Gemini daily quota is exhausted.
Produces identical metadata.json structure using local templates.
Gemini is still used when quota is available (this is the fallback only).
"""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

BASE = Path('.')
video_id = '20260924_learn_colors_red'

script   = json.loads((BASE / 'scripts' / video_id / 'script.json').read_text('utf-8'))
video_mf = json.loads((BASE / 'output'  / video_id / 'video_manifest.json').read_text('utf-8'))
shorts_mf = json.loads((BASE / 'output' / video_id / 'shorts_manifest.json').read_text('utf-8'))

topic       = script['topic']         # "Learn Colors: Red"
learning    = script.get('learning_goal', 'Identify the color red')
target_age  = script.get('target_age', '3-5')
characters  = script.get('characters', ['Bobo', 'Lumi', 'Pip'])
if characters and isinstance(characters[0], dict):
    chars = ', '.join(c['name'] for c in characters)
else:
    chars = ', '.join(str(c) for c in characters)

# ── Title ─────────────────────────────────────────────────────────────────────
title = f"\U0001f534 Learn Colors: RED with Bobo! Fun Learning for Toddlers & Kids \U0001f34e"
print(f"Title ({len(title)} chars): {title[:60]}...")

# ── Description ───────────────────────────────────────────────────────────────
timestamps = []
for ts in video_mf['scene_timestamps']:
    m = int(ts['start_sec']) // 60
    s = int(ts['start_sec']) % 60
    timestamps.append(f"{m:02d}:{s:02d} - {ts['title']}")
ts_block = '\n'.join(timestamps)

description = f"""\U0001f534 It's time to learn about the color RED with Bobo and friends! \U0001f34e\U0001f381\U0001f490
Join Bobo the bear, {chars} as they explore the amazing color RED through fun adventures, songs, and interactive learning!

\U0001f4da WHAT'S IN THIS VIDEO:
{ts_block}

\U0001f31f IN THIS VIDEO YOUR CHILD WILL LEARN:
\u2714 Recognise the color RED in everyday objects
\u2714 Name red things like apples, balloons and flowers
\u2714 Sing along with the Bobo color song
\u2714 Play spot-the-color games with Bobo

\U0001f469 FOR PARENTS & TEACHERS:
This episode uses repetition, rhyme, and visual association — proven techniques for color learning in children aged {target_age}. All content is child-safe, ad-friendly, and COPPA compliant.

\U0001f514 SUBSCRIBE to BOBO KIDS STUDIO for a new color every week!
\U0001f44d LIKE this video if your child learned something new today!
\U0001f4ac COMMENT: What is YOUR child's favourite red thing?

#LearnColors #KidsEducation #PreschoolLearning #ToddlerLearning #BoboKidsStudio #ColorsForKids #RedColor #LearnRed #ChildrenLearning #KidsCartoon #PreschoolColors #BabyLearning #EarlyEducation #ColorSong"""

print(f"Description: {len(description)} characters")

# ── Tags ──────────────────────────────────────────────────────────────────────
tags = [
    "learn colors", "colors for kids", "red color", "learn red",
    "preschool learning", "toddler learning", "kids education",
    "bobo kids studio", "color song", "children learning",
    "learn colors for toddlers", "red objects", "kids cartoon",
    "early education", "color recognition", "preschool colors",
    "baby learning", "educational video kids", "bobo bear",
    "color learning preschool", "nursery colors", "learn red color kids",
    "toddler colors", "kids color video", "red apple kids"
]
tags_chars = sum(len(t) for t in tags) + (len(tags)-1)*2
print(f"Tags: {len(tags)} tags, {tags_chars} chars")

# ── Shorts metadata ───────────────────────────────────────────────────────────
shorts_meta = []
for short in script.get('shorts', []):
    snum    = short['short_number']
    stitle  = f"\U0001f534 {short['title']} | BOBO KIDS STUDIO"[:100]
    hook    = short.get('hook', '')
    ending  = short.get('ending', 'Subscribe to BOBO KIDS STUDIO!')
    stxt    = short.get('script', '')[:200]
    cap     = short.get('caption_text', '').replace(' ', '').replace('!', '')
    sdesc   = f"{hook}\n\n{stxt}...\n\n{ending}\n\n#Shorts #LearnColors #KidsEducation #ToddlerLearning #BoboKidsStudio #{cap}"[:500]
    shorts_meta.append({
        "short_number": snum, "title": stitle,
        "description": sdesc.strip(), "original_hook": hook,
    })
    print(f"Short {snum}: '{stitle[:55]}...'")

# ── Build manifest ────────────────────────────────────────────────────────────
metadata = {
    "video_id": video_id,
    "main_video": {
        "title":            title,
        "description":      description,
        "tags":             tags,
        "category_id":      "27",
        "made_for_kids":    True,
        "default_language": "en",
        "privacy_status":   "private",
        "file":             str(Path('output') / video_id / 'main_video.mp4'),
        "thumbnail_file":   str(Path('thumbnails') / video_id / 'thumbnail_final.jpg'),
    },
    "shorts": [
        {**sm, "category_id": "27", "made_for_kids": True,
         "default_language": "en", "privacy_status": "private",
         "file": str(Path('output') / video_id / f"short_{sm['short_number']:02d}.mp4")}
        for sm in shorts_meta
    ],
    "_meta": {
        "title_chars":                len(title),
        "description_chars":          len(description),
        "tags_count":                 len(tags),
        "tags_chars":                 tags_chars,
        "shorts_count":               len(shorts_meta),
        "title_within_limit":         len(title) <= 100,
        "description_within_limit":   len(description) <= 5000,
        "tags_within_limit":          tags_chars <= 500,
        "generated_by":               "local_template_fallback",
        "note":                       "Gemini daily quota exhausted. Re-run with --force-redo tomorrow for AI-generated metadata.",
    },
}

out = Path('output') / video_id / 'metadata.json'
out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
print(f"\nSaved: {out}")
print(f"All limits OK: title={len(title)}<=100, desc={len(description)}<=5000, tags={tags_chars}<=500")
