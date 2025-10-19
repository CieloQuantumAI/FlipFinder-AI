import os
import requests
import json

# ✅ Load your API key from environment
BATCHDATA_API_KEY = os.getenv("BATCHDATA_API_KEY")
if not BATCHDATA_API_KEY:
    raise EnvironmentError("❌ Missing BatchData API key. Set env var BATCHDATA_API_KEY first.")

# ✅ Property Search endpoint (for comps)
BATCHDATA_SEARCH_URL = "https://api.batchdata.com/api/v1/property/search"


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
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Property: {address}, {city}, {state}")

            # Check if any results exist
            if "results" in data and "properties" in data["results"] and data["results"]["properties"]:
                return data["results"]["properties"]
            else:
                print("⚠️ No comps found in response.")
                print(f"Response preview:\n{json.dumps(data, indent=2)[:1000]}")
                return None
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Error connecting to BatchData: {e}")
        return None


# 🧩 NEW: Parser to extract usable fields from BatchData comps
def parse_comps(comps):
    parsed = []
    for comp in comps:
        address = comp.get("address", {})
        listing = comp.get("listing", {})
        mls = comp.get("mls", {})

        # --- Address ---
        full_address = f"{address.get('houseNumber', '')} {address.get('street', '')}, {address.get('city', '')}, {address.get('state', '')}".strip().replace(" ,", ",")

        # --- Price ---
        price = (
            listing.get("soldPrice")
            or listing.get("price")
            or mls.get("soldPrice")
            or mls.get("price")
        )

        # --- Sqft ---
        sqft = (
            listing.get("livingArea")
            or listing.get("totalBuildingAreaSquareFeet")
            or mls.get("livingArea")
            or mls.get("totalBuildingAreaSquareFeet")
        )

        # --- Beds / Baths ---
        beds = listing.get("bedroomCount") or mls.get("bedroomCount")
        baths = (
            listing.get("bathroomCount")
            or listing.get("fullBathroomCount")
            or mls.get("bathroomCount")
            or mls.get("fullBathroomCount")
        )

        # --- Year Built ---
        year_built = listing.get("yearBuilt") or mls.get("yearBuilt")

        # --- Type / URL ---
        prop_type = listing.get("propertyType") or mls.get("propertyType")
        listing_url = listing.get("listingUrl") or mls.get("listingUrl")

        # --- Agent / Brokerage ---
        brokerage = (
            listing.get("brokerage", {}).get("name")
            or mls.get("brokerage", {}).get("name")
        )
        agent = (
            listing.get("agents", [{}])[0].get("name")
            if listing.get("agents") else None
        )

        parsed.append({
            "address": full_address or "N/A",
            "price": price or "N/A",
            "sqft": sqft or "N/A",
            "beds": beds or "N/A",
            "baths": baths or "N/A",
            "year_built": year_built or "N/A",
            "type": prop_type or "N/A",
            "brokerage": brokerage or "N/A",
            "agent": agent or "N/A",
            "listing_url": listing_url or "N/A",
        })
    return parsed


# --- Test example ---
if __name__ == "__main__":
    raw_comps = fetch_property_comps(
        address="155 Palmer St NE",
        city="Grand Rapids",
        state="MI",
        zip_code="49505"
    )

    if raw_comps:
        parsed_comps = parse_comps(raw_comps)
        print(f"\n📊 Found {len(parsed_comps)} comps:\n")
        for i, comp in enumerate(parsed_comps[:10], start=1):
            print(f"{i}. {comp['address']} — ${comp['price']} — {comp['sqft']} sqft — {comp['beds']} bd / {comp['baths']} ba")
            print(f"   URL: {comp['listing_url']}")
        print("\n✅ Comps successfully retrieved and parsed!")
    else:
        print("\n⚠️ No comps retrieved.")
