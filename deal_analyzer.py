import os
import json
import pandas as pd
from statistics import median

# --- CONFIG ---
RESULTS_DIR = os.path.join(os.getcwd(), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------- CALCULATIONS ----------
def estimate_arv_from_comps(comps, target_sqft):
    """
    Estimate After Repair Value (ARV) using median $/sqft from comps.
    """
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
    return median_ppsqft * target_sqft


def calculate_max_offer(arv, rehab_cost, percentage=0.7):
    """
    Apply the 70% rule:
    MAO = (ARV * percentage) - rehab_cost
    """
    if not arv:
        return None
    return round((arv * percentage) - rehab_cost, 2)


def analyze_roi(listing_price, arv, rehab_cost):
    """
    Compute profit, ROI %, and margin.
    """
    if not arv or not listing_price:
        return None

    total_investment = listing_price + rehab_cost
    profit = arv - total_investment

    roi = (profit / total_investment * 100) if total_investment > 0 else 0
    margin = (profit / arv * 100) if arv > 0 else 0

    return {
        "arv": round(arv, 2),
        "profit": round(profit, 2),
        "roi_percent": round(roi, 2),
        "margin_percent": round(margin, 2),
    }


# ---------- MAIN PIPELINE ----------
def analyze_deal(property_data, comps):
    """
    property_data: dict with listing_price, rehab_cost, sqft, address, etc.
    comps: list of comp dicts from BatchData
    """
    listing_price = property_data.get("listing_price")
    rehab_cost = property_data.get("rehab_cost")
    sqft = property_data.get("sqft")

    print(f"\n📊 Analyzing deal for: {property_data.get('address', 'Unknown')}")

    # --- Step 1: ARV ---
    arv = estimate_arv_from_comps(comps, sqft)
    if not arv:
        print("⚠️ Could not estimate ARV (no valid comps).")
        return None

    # --- Step 2: ROI + Profit ---
    roi_data = analyze_roi(listing_price, arv, rehab_cost)

    # --- Step 3: 70% Rule Offer ---
    mao = calculate_max_offer(arv, rehab_cost)
    roi_data["max_offer_price"] = mao

    # --- Step 4: Summary ---
    summary = {
        **property_data,
        **roi_data,
        "deal_summary": (
            f"ARV: ${roi_data['arv']:,.0f} | "
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

    print("\n✅ Deal analysis complete!")
    print(f"📄 CSV saved to: {csv_path}")
    print(f"📄 JSON saved to: {json_path}")
    print("\nResult:\n", df.to_string(index=False))
    return summary


# ---------- EXAMPLE USAGE ----------
if __name__ == "__main__":
    # Example property (replace with your actual listing data)
    property_data = {
        "address": "155 Palmer St NE, Grand Rapids, MI",
        "sqft": 1200,
        "listing_price": 160000,
        "rehab_cost": 30000,
    }

    # Example comps (replace with your real API output)
    sample_comps = [
        {"salePrice": 245000, "livingArea": 1300},
        {"salePrice": 235000, "livingArea": 1100},
        {"salePrice": 250000, "livingArea": 1250},
    ]

    analyze_deal(property_data, sample_comps)
