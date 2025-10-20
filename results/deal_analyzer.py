import os
import json
import requests
import pandas as pd
from statistics import median

# --- CONFIG ---
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- BatchData API setup ---
BATCHDATA_API_KEY = os.getenv("BATCHDATA_API_KEY")
if not BATCHDATA_API_KEY:
    raise EnvironmentError("❌ Missing BatchData API key. Set env var BATCHDATA_API_KEY first.")

BATCHDATA_SEARCH_URL = "https://api.batchdata.com/api/v1/property/search"


# ---------- 1️⃣ Fetch Comps ----------
def fetch_property_comps(address, city, state, zip_code):
    """Fetch comparable properties (comps) for a given property from BatchData."""
    payload = {
        "searchCriteria": {
            "compAddress": {
                "street": address,
                "city": city,
                "state": state,
                "zip": zip_code
            }
        },
        "options": {
            "useDistance": True,
            "distanceMiles": 0.5,
            "useBedrooms": True,
            "minBedrooms": -1,
            "maxBedrooms": 1,
            "useBathrooms": True,
            "minBathrooms": -1,
            "maxBathrooms": 1,
            "useArea": True,
            "minArea": -200,
            "maxArea": 200,
            "skip": 0,
            "take": 10
        }
    }

    headers = {
        "Authorization": f"Bearer {BATCHDATA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    try:
        response = requests.post(BATCHDATA_SEARCH_URL, json=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()

        comps = (
            data.get("results", {})
            .get("properties", [])
        )

        if not comps:
            print("⚠️ No comps found in response.")
            print(json.dumps(data, indent=2)[:1000])
            return []

        print(f"📊 Found {len(comps)} comps within 0.5 miles.\n")
        return comps

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error connecting to BatchData: {e}")
        return []


# ---------- 2️⃣ Estimate ARV ----------
def estimate_arv_from_comps(comps, target_sqft):
    """Estimate After Repair Value (ARV) using median $/sqft from comps."""
    if not comps:
        return None

    price_per_sqft = []
    for c in comps:
        # Try several possible price fields
        price = (
            c.get("salePrice")
            or c.get("lastSaleAmount")
            or c.get("listing", {}).get("price")
        )

        sqft = (
            c.get("livingArea")
            or c.get("building", {}).get("area", {}).get("buildingArea")
            or c.get("building", {}).get("size", {}).get("totalSquareFeet")
        )

        if price and sqft and isinstance(price, (int, float)) and sqft > 0:
            price_per_sqft.append(price / sqft)

    if not price_per_sqft:
        return None

    median_ppsqft = median(price_per_sqft)
    arv = median_ppsqft * target_sqft
    return round(arv, 2)


# ---------- 3️⃣ ROI & Offer Calculations ----------
def analyze_roi(listing_price, arv, rehab_cost):
    """Compute profit, ROI %, and margin."""
    total_investment = listing_price + rehab_cost
    profit = arv - total_investment
    roi = (profit / total_investment * 100) if total_investment > 0 else 0
    margin = (profit / arv * 100) if arv > 0 else 0

    return {
        "profit": round(profit, 2),
        "roi_percent": round(roi, 2),
        "margin_percent": round(margin, 2),
    }


def calculate_max_offer(arv, rehab_cost, percentage=0.7):
    """70% rule: MAO = (ARV * percentage) - rehab_cost"""
    return round((arv * percentage) - rehab_cost, 2)


# ---------- 4️⃣ End-to-End Analyzer ----------
def analyze_deal(address, city, state, zip_code, listing_price, rehab_cost, sqft):
    print(f"\n🏠 Analyzing deal for {address}, {city}, {state} {zip_code}")

    comps = fetch_property_comps(address, city, state, zip_code)
    if not comps:
        print("❌ No comps retrieved.")
        return None

    arv = estimate_arv_from_comps(comps, sqft)
    if not arv:
        print("⚠️ Could not estimate ARV — no valid comps.")
        return None

    roi_data = analyze_roi(listing_price, arv, rehab_cost)
    mao = calculate_max_offer(arv, rehab_cost)

    summary = {
        "address": f"{address}, {city}, {state} {zip_code}",
        "listing_price": listing_price,
        "rehab_cost": rehab_cost,
        "sqft": sqft,
        "arv": arv,
        "max_offer_price": mao,
        **roi_data,
        "deal_summary": (
            f"💰 ARV: ${arv:,.0f} | "
            f"Max Offer (70% rule): ${mao:,.0f} | "
            f"ROI: {roi_data['roi_percent']}% | "
            f"Profit: ${roi_data['profit']:,.0f}"
        ),
    }

    # Save results
    df = pd.DataFrame([summary])
    csv_path = os.path.join(RESULTS_DIR, "deal_analysis.csv")
    json_path = os.path.join(RESULTS_DIR, "deal_analysis.json")

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("\n✅ Deal Analysis Complete!")
    print(f"📄 CSV:  {csv_path}")
    print(f"📄 JSON: {json_path}")
    print("\n📊 Summary:\n", df.to_string(index=False))

    return summary


# ---------- Example Run ----------
if __name__ == "__main__":
    # 👇 Change this to analyze any property
    deal = analyze_deal(
        address="155 Palmer St NE",
        city="Grand Rapids",
        state="MI",
        zip_code="49505",
        listing_price=160000,
        rehab_cost=30000,
        sqft=1200
    )
