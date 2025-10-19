"""
AI Rehab Estimator (Test Version)
--------------------------------
Run this to test the CLIP-based condition and rehab cost estimator
on a single property using manually uploaded photos.

Setup:
1. Place your test images inside the 'images/' folder.
2. Update TEST_PROPERTY below with address, sqft, and listing_price.
3. Run:  python ai_rehab_estimator_test.py
Outputs: results/rehab_estimate_test.csv and .json
"""

import os
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# ---------- CONFIG ----------
ROOT = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(ROOT, "images")
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ✅ Define one test property manually
TEST_PROPERTY = {
    "id": "TEST001",
    "address": "155 Palmer St NE, Grand Rapids, MI 49505",
    "listing_price": 160000,
    "sqft": 1116,
}

# CLIP prompts
DAMAGE_PROMPTS = [
    "boarded windows", "broken roof", "peeling paint", "water damage",
    "mold stains", "collapsed ceiling", "overgrown lawn", "broken windows",
    "exposed wiring", "damaged exterior",
]

GOOD_PROMPTS = [
    "updated kitchen", "new flooring", "modern bathroom",
    "well-maintained interior", "recently renovated", "fresh paint",
]

# rehab $/sqft mapping
PER_SQFT_BY_RATING = {1: 12.5, 2: 20, 3: 30, 4: 45, 5: 60}
RATING_THRESHOLDS = [0.15, 0.30, 0.55, 0.80]

# ---------- HELPERS ----------
def map_damage_to_rating(score: float) -> int:
    if score <= RATING_THRESHOLDS[0]: return 1
    elif score <= RATING_THRESHOLDS[1]: return 2
    elif score <= RATING_THRESHOLDS[2]: return 3
    elif score <= RATING_THRESHOLDS[3]: return 4
    else: return 5

def estimate_rehab_cost(rating: int, sqft: float) -> float:
    return PER_SQFT_BY_RATING.get(rating, 60) * sqft

# ---------- MODEL ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name).to(device)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Pre-encode prompts
text_prompts = DAMAGE_PROMPTS + GOOD_PROMPTS
with torch.no_grad():
    text_inputs = clip_processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
    text_embeds = clip_model.get_text_features(**text_inputs).cpu().numpy()

# ---------- ANALYSIS ----------
def analyze_images(image_paths: List[str], top_k_issues: int = 3):
    embeddings = []
    valid = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            inputs = clip_processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs).cpu().numpy()[0]
            embeddings.append(feat)
            valid.append(path)
        except Exception as e:
            print(f"⚠️ Failed to process {path}: {e}")

    if not embeddings:
        return 0.0, []

    sims = cosine_similarity(np.stack(embeddings), text_embeds)
    n_damage = len(DAMAGE_PROMPTS)
    damage_sims = sims[:, :n_damage]
    good_sims = sims[:, n_damage:]

    per_img_scores = []
    top_issues = {}
    for i in range(len(valid)):
        max_dmg = float(damage_sims[i].max())
        avg_good = float(good_sims[i].mean())
        raw = max_dmg - avg_good
        normalized = (raw + 1) / 2
        normalized = max(0.0, min(1.0, normalized))
        per_img_scores.append(normalized)
        # collect top issues
        for idx in damage_sims[i].argsort()[::-1][:top_k_issues]:
            prompt = DAMAGE_PROMPTS[idx]
            top_issues[prompt] = max(top_issues.get(prompt, 0.0), float(damage_sims[i][idx]))

    damage_score = float(np.median(per_img_scores))
    top_issues_list = sorted(top_issues.items(), key=lambda x: x[1], reverse=True)[:top_k_issues]
    return damage_score, top_issues_list

# ---------- MAIN TEST ----------
def main():
    print("\n🧠 Running CLIP analysis on test property...")

    # load all images in images/
    image_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not image_files:
        print(f"⚠️ No images found in {IMAGES_DIR}. Please add some and re-run.")
        return

    damage_score, issues = analyze_images(image_files)
    rating = map_damage_to_rating(damage_score)
    est_cost = estimate_rehab_cost(rating, TEST_PROPERTY["sqft"])

    notes = "; ".join([f"{k} ({v:.2f})" for k, v in issues])

    result = {
        **TEST_PROPERTY,
        "num_photos": len(image_files),
        "damage_score": round(damage_score, 3),
        "condition_rating": rating,
        "est_rehab_cost": round(est_cost, 2),
        "notes": notes,
    }

    df = pd.DataFrame([result])
    csv_path = os.path.join(RESULTS_DIR, "rehab_estimate_test.csv")
    json_path = os.path.join(RESULTS_DIR, "rehab_estimate_test.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("\n✅ Test completed!")
    print(f"📄 CSV:  {csv_path}")
    print(f"📄 JSON: {json_path}")
    print("\nResult:\n", df.to_string(index=False))

if __name__ == "__main__":
    main()
