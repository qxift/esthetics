import os
import pandas as pd
import requests
from io import BytesIO
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
CSV_IN = "combined_artworks.csv"       # Upload this to server first
OUT_DIR = "aesthetic_art"              # Folder where images will be saved
MAP_OUT = "aesthetic_art_map.csv"      # Mapping file to generate
TIMEOUT = 10                           # Max seconds per request

# -----------------------------
# PREP
# -----------------------------
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_IN, low_memory=False)
df = df[df["image_url"].notna()].copy()
df.reset_index(drop=True, inplace=True)

mapping_rows = []

print(f"Total rows with images: {len(df)}")
print("Starting download...\n")

# -----------------------------
# MAIN DOWNLOAD LOOP
# -----------------------------
for i, row in df.iterrows():
    url = row["image_url"]

    # Skip invalid URLs
    if not isinstance(url, str) or url.strip() == "":
        continue

    # ---- SKIP known garbage URLs ----
    if url == "https://uploads.wikiart.org/Content/wiki/img/favicon-256x256.png":
        continue

    if "FRAME-600x480.jpg" in url:
        continue

    # Output filename: nationality_index.jpg
    nat = str(row.get("nationality", "unknown")).replace(" ", "_")
    filename = f"{nat}_{i:06d}.jpg"
    filepath = os.path.join(OUT_DIR, filename)

    # Resume support — skip if file already exists
    if os.path.exists(filepath):
        mapping_rows.append({
            "csv_index": i,
            "local_filename": filename,
            "server_path": filepath,
            "image_url": url
        })
        continue

    # ----------------------------------------------------------
    # DOWNLOAD IMAGE
    # ----------------------------------------------------------
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()

        img = Image.open(BytesIO(r.content)).convert("RGB")
        img.save(filepath)

        mapping_rows.append({
            "csv_index": i,
            "local_filename": filename,
            "server_path": filepath,
            "image_url": url
        })

    except Exception as e:
        print(f"[ERROR] Failed: {url} ({e})")
        continue  # DO NOT STOP — keep downloading next images

    # Progress indicator
    if i % 500 == 0 and i > 0:
        print(f"Downloaded {i} images so far...")

# -----------------------------
# SAVE MAPPING
# -----------------------------
mapping_df = pd.DataFrame(mapping_rows)
mapping_df.to_csv(MAP_OUT, index=False)

print("\nDONE!")
print(f"Saved mapping file → {MAP_OUT}")
print(f"Images saved in → {OUT_DIR}/")
