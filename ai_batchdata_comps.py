import os
import requests
import json
import time
from apify_client import ApifyClient

# ✅ Load your API keys from environment
BATCHDATA_API_KEY = os.getenv("BATCHDATA_API_KEY") 
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN") or "apify_api_YOUR_TOKEN_HERE"

if not BATCHDATA_API_KEY:
    raise EnvironmentError("❌ Missing BatchData API key. Set env var BATCHDATA_API_KEY first.")

if not APIFY_API_TOKEN or APIFY_API_TOKEN == "apify_api_YOUR_TOKEN_HERE":
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


def scrape_zillow_by_addresses(addresses):
    """
    Use Apify Zillow scraper to get property details by address.
    Uses one-api/zillow-scrape-address-url-zpid actor with property_addresses mode.
    Returns a dict mapping address -> sqft
    
    Timeout: 2 minutes - aborts if scraper takes longer
    """
    if not APIFY_API_TOKEN or APIFY_API_TOKEN == "apify_api_YOUR_TOKEN_HERE":
        print("⚠️ Skipping Zillow scraping (no API token)")
        return {}
    
    if not addresses:
        print("ℹ️ No addresses to search on Zillow")
        return {}
    
    print(f"\n🔍 Scraping Zillow for {len(addresses)} properties by address...")
    
    try:
        # Initialize Apify client
        client = ApifyClient(APIFY_API_TOKEN)
        
        # Format addresses - ensure they have zip code for better matching
        formatted_addresses = []
        for addr in addresses:
            # Add zip code if not already present
            if "49505" not in addr and "Grand Rapids, MI" in addr:
                addr = addr
            formatted_addresses.append(addr)
        
        print("Addresses to scrape:")
        for addr in formatted_addresses:
            print(f"   • {addr}")
        
        # Prepare input EXACTLY as specified by one-api/zillow-scrape-address-url-zpid
        # Format: one address per line in the multiple_input_box
        addresses_text = "\n".join(formatted_addresses)
        
        run_input = {
            "scrape_type": "property_addresses",  # Set to property_addresses mode
            "multiple_input_box": addresses_text   # Addresses separated by newlines
        }
        
        print(f"\n⏳ Starting Apify actor: one-api/zillow-scrape-address-url-zpid")
        print(f"   Scraping {len(formatted_addresses)} addresses (timeout: 2 minutes)...")
        
        # Start the actor (don't wait for completion yet)
        run = client.actor("one-api/zillow-scrape-address-url-zpid").start(run_input=run_input)
        run_id = run["id"]
        
        # Wait for completion with 2-minute timeout
        start_time = time.time()
        timeout_seconds = 120  # 2 minutes
        check_interval = 5  # Check every 5 seconds
        
        while True:
            elapsed = time.time() - start_time
            
            # Check if timeout exceeded
            if elapsed > timeout_seconds:
                print(f"\n⏰ Timeout exceeded ({timeout_seconds}s). Aborting Apify actor...")
                try:
                    # Abort the actor run
                    client.run(run_id).abort()
                    print("✅ Actor aborted successfully")
                except Exception as abort_error:
                    print(f"⚠️ Could not abort actor: {abort_error}")
                
                print("⚠️ Falling back to comps with existing sqft data only")
                return {}
            
            # Check run status
            run_info = client.run(run_id).get()
            status = run_info.get("status")
            
            if status in ["SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"]:
                if status == "SUCCEEDED":
                    print(f"✅ Actor completed in {elapsed:.1f} seconds")
                    break
                else:
                    print(f"⚠️ Actor ended with status: {status}")
                    return {}
            
            # Show progress
            if int(elapsed) % 15 == 0:  # Every 15 seconds
                print(f"   ⏳ Still running... ({int(elapsed)}s elapsed)")
            
            time.sleep(check_interval)
        
        # Get results
        all_items = list(client.dataset(run_info["defaultDatasetId"]).iterate_items())
        
        print(f"✅ Successfully retrieved {len(all_items)} results from Zillow")
        
        if not all_items:
            print("⚠️ No results returned from Zillow scraper")
            return {}
        
        print(f"\n📦 Processing {len(all_items)} results...")
        
        # Debug: Show structure of first result
        if all_items:
            print(f"\n🔍 Sample result structure (first 500 chars):")
            print(json.dumps(all_items[0], indent=2)[:500])
            print("\nAvailable fields:", list(all_items[0].keys()))
        
        # Extract sqft from results
        results = {}
        for item in all_items:
            # Extract address - try multiple possible fields
            result_address = (
                item.get("address") or 
                item.get("streetAddress") or
                item.get("full_address") or
                item.get("formattedAddress") or
                ""
            )
            
            # Extract sqft - try ALL possible field names from Zillow
            sqft = (
                item.get("livingArea") or
                item.get("livingAreaSqFt") or
                item.get("area") or
                item.get("livingAreaValue") or
                item.get("finishedSqFt") or
                item.get("homeSize") or
                item.get("squareFeet") or
                item.get("size") or
                item.get("livingAreaSF") or
                (item.get("resoFacts", {}).get("livingArea") if isinstance(item.get("resoFacts"), dict) else None) or
                (item.get("propertyDetails", {}).get("livingArea") if isinstance(item.get("propertyDetails"), dict) else None)
            )
            
            if result_address:
                # Match back to original addresses
                matched = False
                for search_addr in addresses:
                    if address_match(search_addr, result_address):
                        if sqft:
                            results[search_addr] = sqft
                            print(f"   ✅ Matched: {search_addr} → {sqft} sqft")
                        else:
                            print(f"   ⚠️ Matched {search_addr} but no sqft found in result")
                        matched = True
                        break
                
                if not matched:
                    print(f"   ⚠️ Could not match result address: {result_address}")
        
        print(f"\n✅ Found sqft for {len(results)}/{len(addresses)} properties")
        return results
        
    except Exception as e:
        print(f"❌ Error scraping Zillow: {e}")
        print("⚠️ Continuing with comps that have sqft from BatchData")
        return {}


def address_match(addr1, addr2):
    """
    Improved fuzzy matching for addresses.
    Returns True if addresses likely refer to the same property.
    """
    # Normalize addresses
    a1 = addr1.lower().replace(",", "").replace(".", "").replace("  ", " ").strip()
    a2 = addr2.lower().replace(",", "").replace(".", "").replace("  ", " ").strip()
    
    # Remove common abbreviations
    replacements = {
        " street": " st", " avenue": " ave", " road": " rd", 
        " drive": " dr", " lane": " ln", " court": " ct",
        " northeast": " ne", " northwest": " nw", 
        " southeast": " se", " southwest": " sw"
    }
    
    for old, new in replacements.items():
        a1 = a1.replace(old, new)
        a2 = a2.replace(old, new)
    
    # Split into parts
    parts1 = a1.split()
    parts2 = a2.split()
    
    # Check if first 2-3 parts match (street number and name)
    if len(parts1) >= 2 and len(parts2) >= 2:
        # Match street number
        if parts1[0] != parts2[0]:
            return False
        
        # Match street name (first word)
        if parts1[1] in a2 or parts2[1] in a1:
            return True
    
    # Fallback: check if 70% of words match
    matching_words = sum(1 for word in parts1 if word in parts2)
    return matching_words >= len(parts1) * 0.7


def enrich_comps_with_zillow(parsed_comps):
    """
    Enrich comps that are missing sqft by scraping Zillow by address.
    If Zillow scraping times out or fails, returns only comps with valid sqft.
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
    
    # Scrape Zillow for missing data (with 2-minute timeout)
    zillow_data = scrape_zillow_by_addresses(addresses_to_search)
    
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
    
    # If Zillow scraping failed/timed out, filter out comps without sqft
    if updated_count == 0 and len(addresses_to_search) > 0:
        print("\n⚠️ Zillow scraping failed or timed out")
        print("📊 Filtering to use only comps with valid sqft from BatchData...")
        
        # Keep only comps that have valid sqft and price
        valid_comps = []
        for comp in parsed_comps:
            if comp['sqft'] != "N/A" and comp['price'] != "N/A":
                valid_comps.append(comp)
        
        removed_count = len(parsed_comps) - len(valid_comps)
        print(f"✅ Kept {len(valid_comps)} comps with valid data (removed {removed_count})")
        
        return valid_comps
    
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