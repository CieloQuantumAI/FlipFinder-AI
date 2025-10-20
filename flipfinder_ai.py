"""
FlipFinder-AI: Complete Real Estate Deal Analysis (Streamlined)
----------------------------------------------------------------
Manual property input version - no scraping required!

SETUP:
1. Update SUBJECT_PROPERTY below with your target property data
2. Place property photos in 'images/' folder (or specify custom folder)
3. Run: python flipfinder_ai.py

The app will:
✅ Analyze property condition from photos using AI (CLIP)
✅ Fetch comparable properties from BatchData API
✅ Calculate ARV, rehab costs, ROI scenarios
✅ Generate comprehensive deal summary with recommendations

Requirements:
    - Set BATCHDATA_API_KEY environment variable
    - Optional: APIFY_API_TOKEN for better comp enrichment
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# Import modules
from ai_batchdata_comps import (
    fetch_property_comps,
    parse_comps,
    enrich_comps_with_zillow
)
from ai_deal_analyzer import DealAnalyzer, format_deal_summary

# Import rehab estimator components
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity


# ========== ⚙️ CONFIGURATION - EDIT THIS ==========
SUBJECT_PROPERTY = {
    "id": "PROP001",
    "address": "155 Palmer St NE",
    "city": "Grand Rapids",
    "state": "MI",
    "zip_code": "49505",
    "listing_price": 160000,  # Current asking price
    "sqft": 1116,             # Square footage
    "beds": 3,                # Number of bedrooms
    "baths": 2,               # Number of bathrooms
    "year_built": None,       # Optional
    "lot_size": None          # Optional
}

# Image folder - where property photos are stored
IMAGES_DIR = "images"

# Results output folder
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Optional: Override rehab cost if you have your own estimate
MANUAL_REHAB_COST = None  # Set to dollar amount (e.g., 35000) or leave as None for AI estimate
# ====================================================


class RehabEstimator:
    """Estimates rehab costs from property photos using CLIP AI."""
    
    DAMAGE_PROMPTS = [
        "boarded windows", "broken roof", "peeling paint", "water damage",
        "mold stains", "collapsed ceiling", "overgrown lawn", "broken windows",
        "exposed wiring", "damaged exterior", "cracked walls", "foundation damage"
    ]
    
    GOOD_PROMPTS = [
        "updated kitchen", "new flooring", "modern bathroom",
        "well-maintained interior", "recently renovated", "fresh paint",
        "granite countertops", "stainless appliances"
    ]
    
    # Rehab cost per sqft by condition rating (adjust for your market)
    PER_SQFT_BY_RATING = {
        1: 12.5,   # Excellent - cosmetic only
        2: 20,     # Good - light updates
        3: 30,     # Fair - moderate rehab
        4: 45,     # Poor - major rehab
        5: 60      # Very poor - gut rehab
    }
    
    RATING_THRESHOLDS = [0.15, 0.30, 0.55, 0.80]
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading CLIP model on {self.device}...")
        
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Pre-encode text prompts
        text_prompts = self.DAMAGE_PROMPTS + self.GOOD_PROMPTS
        with torch.no_grad():
            text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
            self.text_embeds = self.model.get_text_features(**text_inputs).cpu().numpy()
        
        print("✅ CLIP model loaded\n")
    
    def analyze_images(self, image_paths, top_k_issues=3):
        """Analyze images and return damage score and top issues."""
        embeddings = []
        valid = []
        
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feat = self.model.get_image_features(**inputs).cpu().numpy()[0]
                embeddings.append(feat)
                valid.append(path)
            except Exception as e:
                print(f"⚠️ Failed to process {path}: {e}")
        
        if not embeddings:
            return 0.5, []
        
        sims = cosine_similarity(np.stack(embeddings), self.text_embeds)
        n_damage = len(self.DAMAGE_PROMPTS)
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
            
            # Collect top issues
            for idx in damage_sims[i].argsort()[::-1][:top_k_issues]:
                prompt = self.DAMAGE_PROMPTS[idx]
                score = float(damage_sims[i][idx])
                top_issues[prompt] = max(top_issues.get(prompt, 0.0), score)
        
        damage_score = float(np.median(per_img_scores))
        top_issues_list = sorted(top_issues.items(), key=lambda x: x[1], reverse=True)[:top_k_issues]
        return damage_score, top_issues_list
    
    def map_damage_to_rating(self, score):
        """Convert damage score to 1-5 rating."""
        if score <= self.RATING_THRESHOLDS[0]: return 1
        elif score <= self.RATING_THRESHOLDS[1]: return 2
        elif score <= self.RATING_THRESHOLDS[2]: return 3
        elif score <= self.RATING_THRESHOLDS[3]: return 4
        else: return 5
    
    def estimate_rehab_cost(self, rating, sqft):
        """Calculate rehab cost based on rating and sqft."""
        return self.PER_SQFT_BY_RATING.get(rating, 60) * sqft
    
    def get_condition_description(self, rating):
        """Get human-readable condition description."""
        descriptions = {
            1: "Excellent - Move-in ready, cosmetic work only",
            2: "Good - Light updates recommended",
            3: "Fair - Moderate repairs needed",
            4: "Poor - Significant renovation required",
            5: "Very Poor - Major structural/system repairs needed"
        }
        return descriptions.get(rating, "Unknown")
    
    def analyze_property(self, image_dir, sqft):
        """Complete analysis of property from images."""
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        
        if os.path.exists(image_dir):
            for ext in image_extensions:
                image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
        
        image_files = [str(f) for f in image_files]
        
        if not image_files:
            print(f"⚠️ No images found in {image_dir}")
            print(f"   Using default moderate rehab estimate\n")
            return {
                'num_photos': 0,
                'damage_score': 0.5,
                'condition_rating': 3,
                'condition_description': 'Fair - Moderate repairs needed (estimated)',
                'est_rehab_cost': 30 * sqft,
                'cost_per_sqft': 30,
                'notes': 'No images available - using default estimate'
            }
        
        print(f"📸 Analyzing {len(image_files)} photos...")
        damage_score, issues = self.analyze_images(image_files)
        rating = self.map_damage_to_rating(damage_score)
        cost = self.estimate_rehab_cost(rating, sqft)
        notes = "; ".join([f"{k} ({v:.2f})" for k, v in issues]) if issues else 'Property in good condition'
        
        return {
            'num_photos': len(image_files),
            'damage_score': round(damage_score, 3),
            'condition_rating': rating,
            'condition_description': self.get_condition_description(rating),
            'est_rehab_cost': round(cost, 2),
            'cost_per_sqft': self.PER_SQFT_BY_RATING.get(rating, 60),
            'notes': notes
        }


def validate_property_data(prop):
    """Validate required property fields."""
    required = ['address', 'city', 'state', 'zip_code', 'listing_price', 'sqft']
    missing = [field for field in required if not prop.get(field)]
    
    if missing:
        print(f"❌ Missing required fields in SUBJECT_PROPERTY: {', '.join(missing)}")
        return False
    
    # Type validation
    try:
        float(prop['listing_price'])
        float(prop['sqft'])
    except (ValueError, TypeError):
        print("❌ listing_price and sqft must be numeric")
        return False
    
    return True


def run_complete_analysis(subject_property, images_dir, manual_rehab=None):
    """Run complete FlipFinder-AI analysis."""
    
    print("\n" + "=" * 80)
    print("🏠 FLIPFINDER-AI: COMPLETE DEAL ANALYSIS")
    print("=" * 80)
    print(f"\n📍 Target Property: {subject_property['address']}, {subject_property['city']}, {subject_property['state']}")
    print(f"💰 Listing Price: ${subject_property['listing_price']:,}")
    print(f"📐 Square Feet: {subject_property['sqft']:,}")
    
    # Validate property data
    if not validate_property_data(subject_property):
        return None
    
    # STEP 1: Analyze property condition from photos
    print("\n" + "=" * 80)
    print("🔬 STEP 1: Analyzing Property Condition")
    print("=" * 80)
    
    if manual_rehab:
        print(f"✅ Using manual rehab cost: ${manual_rehab:,.2f}")
        rehab_estimate = {
            'num_photos': 0,
            'damage_score': 0.5,
            'condition_rating': 3,
            'condition_description': 'Manual estimate provided',
            'est_rehab_cost': manual_rehab,
            'cost_per_sqft': round(manual_rehab / subject_property['sqft'], 2),
            'notes': 'Manual rehab cost provided by user'
        }
    else:
        estimator = RehabEstimator()
        rehab_estimate = estimator.analyze_property(images_dir, subject_property['sqft'])
        print(f"✅ Condition Rating: {rehab_estimate['condition_rating']}/5 - {rehab_estimate['condition_description']}")
        print(f"✅ Estimated Rehab: ${rehab_estimate['est_rehab_cost']:,.2f} (${rehab_estimate['cost_per_sqft']:.2f}/sqft)")
    
    # STEP 2: Fetch comps from BatchData
    print("\n" + "=" * 80)
    print("🔍 STEP 2: Fetching Comparable Properties")
    print("=" * 80)
    
    raw_comps = fetch_property_comps(
        address=subject_property['address'],
        city=subject_property['city'],
        state=subject_property['state'],
        zip_code=subject_property['zip_code']
    )
    
    if not raw_comps:
        print("❌ Failed to fetch comps from BatchData")
        print("💡 Check that BATCHDATA_API_KEY is set correctly")
        return None
    
    # STEP 3: Parse and enrich comps
    print("\n📊 Parsing comps...")
    parsed_comps = parse_comps(raw_comps)
    print(f"✅ Parsed {len(parsed_comps)} comps")
    
    print("\n🔍 Enriching comps with Zillow data (if needed)...")
    enriched_comps = enrich_comps_with_zillow(parsed_comps)
    
    # STEP 4: Calculate deal metrics
    print("\n" + "=" * 80)
    print("💰 STEP 3: Calculating Deal Metrics")
    print("=" * 80)
    
    analyzer = DealAnalyzer()
    
    # Calculate ARV from comps
    comps_data = analyzer.calculate_arv(enriched_comps)
    
    if comps_data['comp_count'] == 0:
        print("❌ No valid comps found for analysis")
        return None
    
    print(f"✅ Found {comps_data['comp_count']} valid comps")
    print(f"✅ Median $/sqft: ${comps_data['median_price_per_sqft']:.2f}")
    print(f"✅ Price range: ${comps_data['price_range']['min']:.2f} - ${comps_data['price_range']['max']:.2f} per sqft")
    
    # Generate complete deal summary
    deal_summary = analyzer.generate_deal_summary(
        property_info=subject_property,
        comps_data=comps_data,
        rehab_estimate=rehab_estimate,
        listing_price=subject_property['listing_price']
    )
    
    # STEP 5: Save results
    print("\n" + "=" * 80)
    print("💾 STEP 4: Saving Results")
    print("=" * 80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save complete deal summary (JSON)
    summary_path = os.path.join(RESULTS_DIR, f"deal_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(deal_summary, f, indent=2)
    print(f"✅ Deal summary: {summary_path}")
    
    # Save comps (CSV)
    comps_df = pd.DataFrame(enriched_comps)
    comps_path = os.path.join(RESULTS_DIR, f"comps_{timestamp}.csv")
    comps_df.to_csv(comps_path, index=False)
    print(f"✅ Comps data: {comps_path}")
    
    # Save rehab estimate (JSON)
    rehab_path = os.path.join(RESULTS_DIR, f"rehab_estimate_{timestamp}.json")
    with open(rehab_path, 'w') as f:
        json.dump(rehab_estimate, f, indent=2)
    print(f"✅ Rehab estimate: {rehab_path}")
    
    # STEP 6: Display summary
    print(format_deal_summary(deal_summary))
    
    return deal_summary


def main():
    """Main entry point."""
    
    print("\n" + "🏠" * 40)
    print("FLIPFINDER-AI: Real Estate Investment Analyzer")
    print("🏠" * 40)
    
    # Check API keys
    if not os.getenv("BATCHDATA_API_KEY"):
        print("\n❌ ERROR: BATCHDATA_API_KEY not set!")
        print("   Set it with: export BATCHDATA_API_KEY='your_key_here'")
        print("   Get your key at: https://www.batchdata.com/")
        return
    
    if not os.getenv("APIFY_API_TOKEN"):
        print("\n⚠️ Warning: APIFY_API_TOKEN not set (Zillow enrichment will be limited)")
    
    # Check images directory
    if not os.path.exists(IMAGES_DIR):
        print(f"\n⚠️ Warning: Images directory '{IMAGES_DIR}' not found")
        print(f"   Create it and add property photos, or analysis will use default estimates")
        os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Run analysis
    try:
        result = run_complete_analysis(
            SUBJECT_PROPERTY, 
            IMAGES_DIR,
            MANUAL_REHAB_COST
        )
        
        if result:
            print("\n" + "=" * 80)
            print("✅ ANALYSIS COMPLETE!")
            print("=" * 80)
            print(f"\n🎯 QUICK SUMMARY:")
            print(f"   Deal Rating: {result['deal_rating']}")
            print(f"   ARV: ${result['arv']:,.2f}")
            print(f"   Recommended Offer: ${result['recommended_offer']:,.2f}")
            print(f"   Expected ROI at MAO: {result['roi_at_mao']['roi_percent']:.1f}%")
            print(f"   Action: {result['recommendation']}")
            
            # Show target ROI scenarios
            print(f"\n💡 TARGET ROI SCENARIOS:")
            for target in [10, 15, 20, 25]:
                scenario = result['target_roi_scenarios'].get(f'roi_{target}_percent', {})
                if scenario.get('calculated_purchase_price'):
                    print(f"   {target}% ROI → Offer ${scenario['calculated_purchase_price']:,.2f}")
                else:
                    print(f"   {target}% ROI → {scenario.get('error', 'Not achievable')}")
            
            print("\n" + "=" * 80)
        else:
            print("\n❌ Analysis failed")
            
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()