"""
AI Rehab Estimator (Address-Based)
-----------------------------------
Automatically scrapes property data and images, then estimates rehab costs.

Usage:
    python ai_rehab_estimator.py --address "123 Main St, City, State"
    python ai_rehab_estimator.py --address "123 Main St, City, State" --scrape-only
    python ai_rehab_estimator.py --folder "scraped_properties/123_Main_St"

Options:
    --address: Property address to scrape and analyze
    --folder: Use existing scraped property folder
    --scrape-only: Only scrape images, don't run analysis
    --source: Preferred source (zillow, realtor, auto)

Requirements:
    pip install torch transformers scikit-learn pillow pandas requests beautifulsoup4 playwright
    playwright install chromium
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple

import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

# Import scraper
from ai_listing_scraper import ListingScraper


# ---------- CONFIG ----------
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# Rehab cost mapping ($/sqft by condition rating)
PER_SQFT_BY_RATING = {
    1: 12.5,   # Excellent condition
    2: 20,     # Good condition
    3: 30,     # Fair condition
    4: 45,     # Poor condition
    5: 60      # Very poor condition
}

# Damage score thresholds for ratings
RATING_THRESHOLDS = [0.15, 0.30, 0.55, 0.80]


class RehabEstimator:
    """Estimates rehab costs from property photos using CLIP."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading CLIP model on {self.device}...")
        
        model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Pre-encode text prompts
        text_prompts = DAMAGE_PROMPTS + GOOD_PROMPTS
        with torch.no_grad():
            text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)
            self.text_embeds = self.model.get_text_features(**text_inputs).cpu().numpy()
        
        print("✅ CLIP model loaded")
    
    def analyze_images(self, image_paths: List[str], top_k_issues: int = 3) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Analyze images and return damage score and top issues.
        
        Args:
            image_paths: List of image file paths
            top_k_issues: Number of top issues to return
        
        Returns:
            Tuple of (damage_score, list of (issue, confidence) tuples)
        """
        embeddings = []
        valid = []
        
        print(f"📸 Analyzing {len(image_paths)} images...")
        
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feat = self.model.get_image_features(**inputs).cpu().numpy()[0]
                embeddings.append(feat)
                valid.append(path)
            except Exception as e:
                print(f"   ⚠️ Failed to process {path}: {e}")
        
        if not embeddings:
            print("   ⚠️ No valid images to analyze")
            return 0.5, []
        
        # Calculate similarities
        sims = cosine_similarity(np.stack(embeddings), self.text_embeds)
        n_damage = len(DAMAGE_PROMPTS)
        damage_sims = sims[:, :n_damage]
        good_sims = sims[:, n_damage:]
        
        # Calculate per-image scores
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
                prompt = DAMAGE_PROMPTS[idx]
                top_issues[prompt] = max(top_issues.get(prompt, 0.0), float(damage_sims[i][idx]))
        
        damage_score = float(np.median(per_img_scores))
        top_issues_list = sorted(top_issues.items(), key=lambda x: x[1], reverse=True)[:top_k_issues]
        
        print(f"   Damage Score: {damage_score:.3f}")
        
        return damage_score, top_issues_list
    
    def map_damage_to_rating(self, score: float) -> int:
        """Convert damage score to 1-5 condition rating."""
        if score <= RATING_THRESHOLDS[0]:
            return 1
        elif score <= RATING_THRESHOLDS[1]:
            return 2
        elif score <= RATING_THRESHOLDS[2]:
            return 3
        elif score <= RATING_THRESHOLDS[3]:
            return 4
        else:
            return 5
    
    def estimate_rehab_cost(self, rating: int, sqft: float) -> float:
        """Calculate rehab cost based on condition rating and square footage."""
        return PER_SQFT_BY_RATING.get(rating, 60) * sqft
    
    def analyze_property(self, image_dir: str, sqft: float, property_data: Dict = None) -> Dict:
        """
        Complete analysis of property from images.
        
        Args:
            image_dir: Directory containing property images
            sqft: Property square footage
            property_data: Optional property data dict
        
        Returns:
            Dict with analysis results
        """
        # Find all images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(list(Path(image_dir).glob(f'*{ext}')))
        
        image_files = [str(f) for f in image_files]
        
        if not image_files:
            print(f"⚠️ No images found in {image_dir}")
            return {
                'num_photos': 0,
                'damage_score': 0.5,
                'condition_rating': 3,
                'est_rehab_cost': 30 * sqft,
                'notes': 'No images available - using default estimate'
            }
        
        # Analyze images
        damage_score, issues = self.analyze_images(image_files)
        rating = self.map_damage_to_rating(damage_score)
        cost = self.estimate_rehab_cost(rating, sqft)
        notes = "; ".join([f"{k} ({v:.2f})" for k, v in issues]) if issues else 'Property in good condition'
        
        result = {
            'num_photos': len(image_files),
            'damage_score': round(damage_score, 3),
            'condition_rating': rating,
            'condition_description': self.get_condition_description(rating),
            'est_rehab_cost': round(cost, 2),
            'cost_per_sqft': PER_SQFT_BY_RATING.get(rating, 60),
            'notes': notes,
            'top_issues': [{'issue': k, 'confidence': round(v, 2)} for k, v in issues]
        }
        
        return result
    
    def get_condition_description(self, rating: int) -> str:
        """Get human-readable condition description."""
        descriptions = {
            1: "Excellent - Move-in ready, minimal work needed",
            2: "Good - Light cosmetic updates recommended",
            3: "Fair - Moderate repairs and updates needed",
            4: "Poor - Significant renovation required",
            5: "Very Poor - Major structural/system repairs needed"
        }
        return descriptions.get(rating, "Unknown")


def load_property_data(folder_path: str) -> Optional[Dict]:
    """Load property data from scraped folder."""
    json_path = os.path.join(folder_path, 'property_data.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return None


def format_results(property_data: Dict, rehab_estimate: Dict) -> str:
    """Format results for display."""
    lines = [
        "\n" + "="*80,
        "🏠 REHAB COST ESTIMATE",
        "="*80,
        "",
        "📍 PROPERTY INFORMATION",
        f"   Address: {property_data.get('address', 'N/A')}",
        f"   Price: ${property_data.get('price', 0):,}" if property_data.get('price') else "   Price: N/A",
        f"   Square Feet: {property_data.get('sqft', 'N/A'):,}" if property_data.get('sqft') else "   Square Feet: N/A",
        f"   Beds: {property_data.get('beds', 'N/A')} | Baths: {property_data.get('baths', 'N/A')}",
        f"   Source: {property_data.get('source', 'N/A')}",
        "",
        "🔧 CONDITION ANALYSIS",
        f"   Photos Analyzed: {rehab_estimate['num_photos']}",
        f"   Damage Score: {rehab_estimate['damage_score']:.3f} / 1.000",
        f"   Condition Rating: {rehab_estimate['condition_rating']}/5",
        f"   Description: {rehab_estimate['condition_description']}",
        "",
        "💰 ESTIMATED REHAB COSTS",
        f"   Total Estimate: ${rehab_estimate['est_rehab_cost']:,.2f}",
        f"   Cost per Sqft: ${rehab_estimate['cost_per_sqft']:.2f}",
        "",
        "⚠️ TOP ISSUES DETECTED",
    ]
    
    if rehab_estimate.get('top_issues'):
        for issue_data in rehab_estimate['top_issues']:
            lines.append(f"   • {issue_data['issue']}: {issue_data['confidence']:.2f} confidence")
    else:
        lines.append("   • No significant issues detected")
    
    lines.extend([
        "",
        "📝 NOTES",
        f"   {rehab_estimate['notes']}",
        "",
        "="*80
    ])
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='AI Rehab Cost Estimator')
    parser.add_argument('--address', type=str, help='Property address to scrape and analyze')
    parser.add_argument('--folder', type=str, help='Use existing scraped property folder')
    parser.add_argument('--scrape-only', action='store_true', help='Only scrape images, dont run analysis')
    parser.add_argument('--source', type=str, default='auto', choices=['auto', 'zillow', 'realtor'], 
                       help='Preferred scraping source')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🏠 AI REHAB COST ESTIMATOR")
    print("="*80)
    
    # Determine mode
    if args.folder:
        # Use existing folder
        print(f"\n📁 Using existing folder: {args.folder}")
        property_folder = args.folder
        
        if not os.path.exists(property_folder):
            print(f"❌ Folder not found: {property_folder}")
            return
        
        property_data = load_property_data(property_folder)
        if not property_data:
            print("⚠️ No property_data.json found. Using folder only.")
            property_data = {'address': os.path.basename(property_folder)}
    
    elif args.address:
        # Scrape property
        print(f"\n🔍 Scraping property: {args.address}")
        
        scraper = ListingScraper()
        property_data = scraper.scrape_property(args.address, preferred_source=args.source)
        
        if not property_data:
            print("\n❌ Failed to scrape property")
            print("💡 Try:")
            print("   1. Checking the address format")
            print("   2. Setting APIFY_API_TOKEN environment variable")
            print("   3. Using --folder with manually downloaded images")
            return
        
        # Download images
        property_folder = scraper.download_property_images(property_data)
        
        if not property_folder:
            print("❌ Failed to download images")
            return
        
        if args.scrape_only:
            print("\n✅ Scraping complete! Images saved to:", property_folder)
            print(f"\n💡 To analyze, run:")
            print(f"   python ai_rehab_estimator.py --folder \"{property_folder}\"")
            return
    
    else:
        # Interactive mode
        address = input("\nEnter property address: ").strip()
        if not address:
            print("❌ No address provided")
            return
        
        args.address = address
        return main()
    
    # Validate sqft
    sqft = property_data.get('sqft')
    if not sqft:
        print("\n⚠️ Square footage not found in property data")
        sqft_input = input("Enter square footage manually: ").strip()
        try:
            sqft = float(sqft_input)
            property_data['sqft'] = sqft
        except ValueError:
            print("❌ Invalid square footage. Using default 1500 sqft.")
            sqft = 1500
            property_data['sqft'] = sqft
    
    # Run analysis
    print("\n" + "="*80)
    print("🔬 RUNNING REHAB ANALYSIS")
    print("="*80)
    
    estimator = RehabEstimator()
    rehab_estimate = estimator.analyze_property(property_folder, sqft, property_data)
    
    # Display results
    print(format_results(property_data, rehab_estimate))
    
    # Save results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine property data and rehab estimate
    full_results = {
        'property': property_data,
        'rehab_estimate': rehab_estimate,
        'analysis_date': timestamp
    }
    
    # Save JSON
    json_path = os.path.join(RESULTS_DIR, f"rehab_estimate_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    print(f"\n💾 Results saved to: {json_path}")
    
    # Save CSV summary
    csv_data = {
        'address': property_data.get('address'),
        'price': property_data.get('price'),
        'sqft': property_data.get('sqft'),
        'beds': property_data.get('beds'),
        'baths': property_data.get('baths'),
        'num_photos': rehab_estimate['num_photos'],
        'damage_score': rehab_estimate['damage_score'],
        'condition_rating': rehab_estimate['condition_rating'],
        'est_rehab_cost': rehab_estimate['est_rehab_cost'],
        'cost_per_sqft': rehab_estimate['cost_per_sqft'],
        'notes': rehab_estimate['notes']
    }
    
    csv_path = os.path.join(RESULTS_DIR, f"rehab_estimate_{timestamp}.csv")
    pd.DataFrame([csv_data]).to_csv(csv_path, index=False)
    print(f"💾 CSV summary saved to: {csv_path}")
    
    # Suggest next steps
    print("\n" + "="*80)
    print("🎯 NEXT STEPS")
    print("="*80)
    print("To run full deal analysis with comps and ROI:")
    print(f"   1. Update SUBJECT_PROPERTY in flipfinder_ai.py with this address")
    print(f"   2. Copy images from {property_folder} to images/ folder")
    print(f"   3. Run: python flipfinder_ai.py")
    print("="*80)


if __name__ == "__main__":
    main()