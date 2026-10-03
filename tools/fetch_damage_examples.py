"""Download freely licensed photos of real damage from Wikimedia Commons, with licences.
usage: python tools/fetch_damage_examples.py data/damage_examples [--per-class 8]
Writes <out>/<class>/NN.jpg and <out>/manifest.csv (file, class, title, licence,
author, source URL). Search results can be off-topic: look through each folder
and delete images that do not show that damage before testing.
"""
import argparse
import csv
import re
from pathlib import Path
import time
import requests
def get(url, **kw):
    """GET with a pause before every request and backoff on HTTP 429 (Wikimedia rate limits)."""
    for wait in (2, 10, 30, 60):
        time.sleep(wait)
        r = requests.get(url, headers=HEADERS, timeout=60, **kw)
        if r.status_code != 429:
            r.raise_for_status()
            return r
    r.raise_for_status()
SEARCHES = {
    "water_stain": "water damaged ceiling leak",
    "mold": "black mould damp wall",
    "crack": "crack in wall plaster",
    "hole": "hole in drywall",
    "peeling_paint": "peeling paint wall",
    "efflorescence": "efflorescence masonry wall salt",
}
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "floorplan-pipeline-assessment/0.1 (detector test; contact via GitHub raj-judal)"}
def strip(html):
    return re.sub(r"<[^>]+>", "", html or "").strip()
ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--per-class", type=int, default=8)
a = ap.parse_args()
rows = []
for cls, query in SEARCHES.items():
    params = {"action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
              "gsrnamespace": 6, "gsrlimit": a.per_class * 3, "prop": "imageinfo",
              "iiprop": "url|extmetadata", "iiurlwidth": 1024, "format": "json"}
    pages = get(API, params=params).json().get("query", {}).get("pages", {})
    (a.out / cls).mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(pages.values(), key=lambda p: p.get("index", 0)):
        info = (p.get("imageinfo") or [{}])[0]
        url = info.get("thumburl")
        if not url or n >= a.per_class:
            continue
        meta = info.get("extmetadata", {})
        try:
            img = get(url)
        except Exception as e:
            print(f"skip {p['title']}: {e}")
            continue
        name = f"{n:02d}.jpg"
        (a.out / cls / name).write_bytes(img.content)
        rows.append({"file": f"{cls}/{name}", "class": cls, "title": p["title"],
                     "licence": strip(meta.get("LicenseShortName", {}).get("value")),
                     "author": strip(meta.get("Artist", {}).get("value")),
                     "source": info.get("descriptionurl", "")})
        n += 1
    print(f"{cls}: {n} images")
with open(a.out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"manifest: {a.out / 'manifest.csv'}")
