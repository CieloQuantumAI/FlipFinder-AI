import os
import requests
import json
import time
from apify_client import ApifyClient
from deal_analyzer import analyze_deal

# ✅ Load your API keys from environment
BATCHDATA_API_KEY = os.getenv("BATCHDATA_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")  # Add your Apify token here

if not BATCHDATA_API_KEY:
    raise EnvironmentError("❌ Missing BatchData API key. Set env var BATCHDATA_API_KEY first.")

if not APIFY_API_TOKEN:
    print("⚠️ Warning: APIFY_API_TOKEN not set. Zillow scraping will be skipped.")

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


# 🧩 Parser to extract usable fields from BatchData comps
def parse_comps(comps):
    parsed = []
    for comp in comps:
        address = comp.get("address", {})
        listing = comp.get("listing", {})
        mls = comp.get("mls", {})
        property_data = comp.get("propertyData", {})

        # --- Address ---
        full_address = f"{address.get('street', '')}, {address.get('city', '')}, {address.get('state', '')}".strip().replace(" ,", ",")

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
            or listing.get("buildingAreaSquareFeet")
            or mls.get("livingArea") 
            or mls.get("totalBuildingAreaSquareFeet")
            or mls.get("buildingAreaSquareFeet")
            or property_data.get("buildingAreaSquareFeet")
            or property_data.get("livingAreaSquareFeet")
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


def search_zillow_by_address(addresses):
    """
    Use Apify Zillow scraper to search for properties by address and get sqft.
    Returns a dict mapping address -> sqft
    """
    if not APIFY_API_TOKEN:
        print("⚠️ Skipping Zillow scraping (no API token)")
        return {}
    
    if not addresses:
        print("ℹ️ No addresses to search on Zillow")
        return {}
    
    print(f"\n🔍 Searching Zillow for {len(addresses)} properties by address...")
    
    try:
        # Initialize Apify client
        client = ApifyClient(APIFY_API_TOKEN)
        
        # Prepare search queries - using Zillow's search format
        search_queries = []
        for addr in addresses:
            # Create search query (Zillow accepts address format)
            search_queries.append(addr)
        
        # Prepare actor input for SEARCH mode
        run_input = {
            "search": ", ".join(search_queries),  # Comma-separated addresses
            "maxLevel": 1,  # Only get the search results, don't go deep
            "resultsLimit": len(addresses) * 2,  # Get up to 2 results per address
            "simple": True,  # Return simplified data
            "extendOutputFunction": "",
            "proxyConfiguration": {
                "useApifyProxy": True
            }
        }
        
        # Run the Zillow scraper actor
        print("⏳ Starting Apify Zillow search (this may take 30-90 seconds)...")
        run = client.actor("maxcopell/zillow-api-scraper").call(run_input=run_input)
        
        # Fetch and process results
        results = {}
        all_items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
        
        print(f"📦 Retrieved {len(all_items)} results from Zillow")
        
        # Match scraped results back to original addresses
        for item in all_items:
            # Extract address from result
            result_address = item.get("address", "")
            
            # Try various sqft field names
            sqft = (
                item.get("livingArea") or
                item.get("area") or
                item.get("livingAreaValue") or
                item.get("resoFacts", {}).get("livingArea") if isinstance(item.get("resoFacts"), dict) else None
            )
            
            if sqft and result_address:
                # Try to match this result to one of our search addresses
                for search_addr in addresses:
                    # Simple matching - check if key parts of address are present
                    # Extract street number and name from both
                    if address_match(search_addr, result_address):
                        results[search_addr] = sqft
                        print(f"   ✅ {search_addr} → {sqft} sqft")
                        break
        
        print(f"✅ Found sqft for {len(results)}/{len(addresses)} properties")
        return results
        
    except Exception as e:
        print(f"❌ Error searching Zillow: {e}")
        import traceback
        traceback.print_exc()
        return {}


def address_match(addr1, addr2):
    """
    Simple fuzzy matching for addresses.
    Returns True if addresses likely refer to the same property.
    """
    # Normalize addresses
    a1 = addr1.lower().replace(",", "").replace(".", "").strip()
    a2 = addr2.lower().replace(",", "").replace(".", "").strip()
    
    # Split into parts
    parts1 = a1.split()
    parts2 = a2.split()
    
    # Check if first 2-3 parts match (street number and name)
    if len(parts1) >= 2 and len(parts2) >= 2:
        # Check street number and first word of street name
        return parts1[0] == parts2[0] and parts1[1] in a2
    
    return False


def enrich_comps_with_zillow(parsed_comps):
    """
    Enrich comps that are missing sqft by searching Zillow by address.
    """
    # Find comps with missing sqft
    missing_sqft_comps = []
    addresses_to_search = []
    
    for i, comp in enumerate(parsed_comps):
        if comp['sqft'] == "N/A" and comp['address'] != "N/A":
            missing_sqft_comps.append(i)
            addresses_to_search.append(comp['address'])
    
    if not addresses_to_search:
        print("\n✅ All comps already have sqft data")
        return parsed_comps
    
    print(f"\n📝 Found {len(addresses_to_search)} comps missing sqft")
    print("Addresses to search:")
    for addr in addresses_to_search:
        print(f"   • {addr}")
    
    # Search Zillow for missing data
    zillow_data = search_zillow_by_address(addresses_to_search)
    
    # Update comps with scraped data
    updated_count = 0
    for idx in missing_sqft_comps:
        comp = parsed_comps[idx]
        addr = comp['address']
        
        if addr in zillow_data:
            comp['sqft'] = zillow_data[addr]
            comp['sqft_source'] = 'Zillow (scraped)'
            updated_count += 1
            print(f"   ✅ Updated: {comp['address']} → {comp['sqft']} sqft")
        else:
            print(f"   ⚠️ No match found for: {comp['address']}")
    
    print(f"\n✅ Updated {updated_count}/{len(addresses_to_search)} comps with Zillow data")
    return parsed_comps


# --- Test example ---
if __name__ == "__main__":
    # Step 1: Get comps from BatchData
    raw_comps = fetch_property_comps(
        address="155 Palmer St NE",
        city="Grand Rapids",
        state="MI",
        zip_code="49505"
    )

    if raw_comps:
        # Step 2: Parse the comps
        parsed_comps = parse_comps(raw_comps)
        
        # Step 3: Enrich with Zillow scraping for missing sqft
        enriched_comps = enrich_comps_with_zillow(parsed_comps)
        
        # Step 4: Display results
        print(f"\n📊 Final Results ({len(enriched_comps)} comps):\n")
        print("=" * 80)
        
        for i, comp in enumerate(enriched_comps[:10], start=1):
            sqft_display = f"{comp['sqft']} sqft" if comp['sqft'] != "N/A" else "N/A sqft"
            source = f" [{comp.get('sqft_source', 'BatchData')}]" if 'sqft_source' in comp else ""
            
            print(f"{i}. {comp['address']}")
            print(f"   💰 ${comp['price']} — 📐 {sqft_display}{source}")
            print(f"   🛏️ {comp['beds']} bd / 🚿 {comp['baths']} ba")
            print(f"   🔗 {comp['listing_url']}")
            print()
        
        # Show stats
        complete_sqft = sum(1 for c in enriched_comps if c['sqft'] != "N/A")
        print("=" * 80)
        print(f"📈 Data completeness: {complete_sqft}/{len(enriched_comps)} comps have sqft")
        print("✅ Process complete!")
        
    else:
        print("\n⚠️ No comps retrieved.")