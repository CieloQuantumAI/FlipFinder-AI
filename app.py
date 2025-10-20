"""
AI FlipFinder - Streamlit Dashboard
------------------------------------
Real estate investment analysis tool with AI-powered comps and rehab estimation.

Run with: streamlit run app.py
"""

import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path

# Import backend modules
from ai_batchdata_comps import (
    fetch_property_comps,
    parse_comps,
    enrich_comps_with_zillow
)
from ai_deal_analyzer import DealAnalyzer, format_deal_summary

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="AI FlipFinder",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CUSTOM STYLING ==========
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title styling */
    h1 {
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2563eb;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: white;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        border: none;
        width: 100%;
        transition: background-color 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    
    /* Input fields */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #374151;
    }
    </style>
""", unsafe_allow_html=True)

# ========== HELPER FUNCTIONS ==========

def check_api_keys():
    """Check if required API keys are set."""
    batchdata_key = os.getenv("BATCHDATA_API_KEY")
    apify_key = os.getenv("APIFY_API_TOKEN")
    
    issues = []
    if not batchdata_key:
        issues.append("❌ BATCHDATA_API_KEY not set (required for comps)")
    if not apify_key:
        issues.append("⚠️ APIFY_API_TOKEN not set (Zillow enrichment will be limited)")
    
    return issues


def run_analysis(address: str, city: str, state: str, zip_code: str, 
                sqft: int, listing_price: float, rehab_cost: float = None,
                beds: int = 3, baths: int = 2) -> dict:
    """
    Run complete deal analysis.
    
    Args:
        address: Street address
        city: City name
        state: State code (e.g., 'MI')
        zip_code: ZIP code
        sqft: Square footage
        listing_price: Current listing price
        rehab_cost: Manual rehab cost (if provided)
        beds: Number of bedrooms
        baths: Number of bathrooms
    
    Returns:
        dict: Complete analysis results including ARV, ROI, MAO
    """
    
    # Construct property info
    property_info = {
        "id": "STREAMLIT_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "address": address,
        "city": city,
        "state": state,
        "zip_code": zip_code,
        "listing_price": listing_price,
        "sqft": sqft,
        "beds": beds,
        "baths": baths
    }
    
    # Step 1: Fetch comps
    st.info("🔍 Fetching comparable properties from BatchData...")
    raw_comps = fetch_property_comps(address, city, state, zip_code)
    
    if not raw_comps:
        return {
            "error": "Failed to fetch comps. Check API key and property address.",
            "success": False
        }
    
    # Step 2: Parse comps
    st.info(f"📊 Parsing {len(raw_comps)} comparable properties...")
    parsed_comps = parse_comps(raw_comps)
    
    # Step 3: Enrich with Zillow (if needed)
    st.info("🔍 Enriching comps with additional data...")
    enriched_comps = enrich_comps_with_zillow(parsed_comps)
    
    # Step 4: Calculate deal metrics
    st.info("💰 Calculating deal metrics...")
    analyzer = DealAnalyzer()
    
    # Calculate ARV
    comps_data = analyzer.calculate_arv(enriched_comps)
    
    if comps_data['comp_count'] == 0:
        return {
            "error": "No valid comps found for analysis.",
            "success": False
        }
    
    # Create rehab estimate
    if rehab_cost:
        rehab_estimate = {
            'num_photos': 0,
            'damage_score': 0.5,
            'condition_rating': 3,
            'condition_description': 'Manual estimate provided',
            'est_rehab_cost': rehab_cost,
            'cost_per_sqft': round(rehab_cost / sqft, 2),
            'notes': 'Manual rehab cost provided by user'
        }
    else:
        # Use default moderate estimate
        rehab_estimate = {
            'num_photos': 0,
            'damage_score': 0.5,
            'condition_rating': 3,
            'condition_description': 'Fair - Moderate repairs needed (estimated)',
            'est_rehab_cost': 30 * sqft,
            'cost_per_sqft': 30,
            'notes': 'Default estimate - upload photos for AI analysis'
        }
    
    # Generate deal summary
    deal_summary = analyzer.generate_deal_summary(
        property_info=property_info,
        comps_data=comps_data,
        rehab_estimate=rehab_estimate,
        listing_price=listing_price
    )
    
    deal_summary['success'] = True
    deal_summary['enriched_comps'] = enriched_comps
    
    return deal_summary


def display_results(results: dict):
    """Display analysis results in a clean format."""
    
    if not results.get('success'):
        st.error(f"❌ {results.get('error', 'Analysis failed')}")
        return
    
    # ========== KEY METRICS ==========
    st.markdown("---")
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="ARV (After Repair Value)",
            value=f"${results['arv']:,.0f}",
            delta=f"{results['price_analysis']['arv_to_listing_ratio']:.1f}% of list"
        )
    
    with col2:
        st.metric(
            label="Maximum Allowable Offer",
            value=f"${results['max_allowable_offer']:,.0f}",
            delta=f"-${results['listing_price'] - results['max_allowable_offer']:,.0f}"
        )
    
    with col3:
        roi_color = "normal" if results['roi_at_mao']['roi_percent'] >= 15 else "off"
        st.metric(
            label="ROI at MAO",
            value=f"{results['roi_at_mao']['roi_percent']:.1f}%",
            delta="Target: 15%+"
        )
    
    # ========== DEAL RATING ==========
    st.markdown("---")
    st.subheader("🎯 Deal Assessment")
    
    # Color-code based on rating
    rating_colors = {
        "EXCELLENT": "🟢",
        "GOOD": "🟡",
        "FAIR": "🟠",
        "POOR": "🔴"
    }
    
    rating_icon = rating_colors.get(results['deal_rating'], "⚪")
    
    st.markdown(f"""
    <div style='background-color: white; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #2563eb;'>
        <h3 style='margin: 0; color: #1f2937;'>{rating_icon} {results['deal_rating']}</h3>
        <p style='margin: 0.5rem 0 0 0; color: #4b5563; font-size: 1.1rem;'>{results['recommendation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== OFFER RECOMMENDATIONS ==========
    st.markdown("---")
    st.subheader("💵 Offer Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recommended Offer (95% MAO)**")
        st.markdown(f"### ${results['recommended_offer']:,.0f}")
        st.caption("5% buffer for negotiations")
    
    with col2:
        st.markdown("**Listing Price**")
        st.markdown(f"### ${results['listing_price']:,.0f}")
        diff = results['listing_price'] - results['recommended_offer']
        st.caption(f"${diff:,.0f} above recommended offer")
    
    # ========== TARGET ROI SCENARIOS ==========
    st.markdown("---")
    st.subheader("🎲 Target ROI Scenarios")
    
    scenarios_cols = st.columns(4)
    
    for idx, target_roi in enumerate([10, 15, 20, 25]):
        scenario_key = f'roi_{target_roi}_percent'
        scenario = results['target_roi_scenarios'].get(scenario_key, {})
        
        with scenarios_cols[idx]:
            st.markdown(f"**{target_roi}% ROI**")
            
            if scenario.get('calculated_purchase_price'):
                purchase = scenario['calculated_purchase_price']
                st.markdown(f"### ${purchase:,.0f}")
                
                # Show if achievable
                if purchase > 0 and purchase < results['listing_price']:
                    st.success("✅ Achievable")
                else:
                    st.warning("⚠️ Needs negotiation")
            else:
                st.markdown("### N/A")
                st.error("❌ Not achievable")
    
    # ========== DETAILED BREAKDOWN ==========
    with st.expander("📋 Detailed Cost Breakdown"):
        st.markdown("**Investment Costs**")
        breakdown = results['roi_at_mao']['breakdown']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Purchase Price: ${breakdown['purchase_price']:,.2f}")
            st.write(f"Rehab Cost: ${breakdown['rehab_cost']:,.2f}")
            st.write(f"Closing Costs: ${breakdown['closing_costs']:,.2f}")
        
        with col2:
            st.write(f"Holding Costs: ${breakdown['holding_costs']:,.2f}")
            st.write(f"Selling Costs: ${breakdown['selling_costs']:,.2f}")
            st.write(f"**Total Investment: ${results['roi_at_mao']['total_investment']:,.2f}**")
        
        st.markdown("---")
        st.markdown(f"**Expected Profit:** ${results['roi_at_mao']['gross_profit']:,.2f}")
    
    # ========== COMPS DATA ==========
    with st.expander("🏘️ Comparable Properties"):
        st.markdown(f"**Found {results['comps_summary']['comp_count']} comparable properties**")
        st.write(f"Median $/sqft: ${results['comps_summary']['median_price_per_sqft']:.2f}")
        st.write(f"Price range: ${results['comps_summary']['price_range']['min']:.2f} - ${results['comps_summary']['price_range']['max']:.2f} per sqft")
        
        if results.get('enriched_comps'):
            st.markdown("---")
            for i, comp in enumerate(results['enriched_comps'][:5], 1):
                if comp['sqft'] != "N/A" and comp['price'] != "N/A":
                    price_per_sqft = float(comp['price']) / float(comp['sqft'])
                    st.write(f"{i}. **{comp['address']}**")
                    st.write(f"   ${comp['price']:,} • {comp['sqft']:,} sqft • ${price_per_sqft:.2f}/sqft")
                    st.write(f"   {comp['beds']} bed • {comp['baths']} bath")
                    st.write("")
    
    # ========== REHAB ESTIMATE ==========
    with st.expander("🔧 Rehab Cost Details"):
        rehab = results['rehab_summary']
        st.write(f"**Condition Rating:** {rehab['condition_rating']}/5")
        st.write(f"**Estimated Cost:** ${rehab['estimated_cost']:,.2f}")
        st.write(f"**Notes:** {rehab['notes']}")
    
    # ========== EXPORT ==========
    st.markdown("---")
    
    # JSON export
    json_data = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=json_data,
        file_name=f"flipfinder_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


# ========== MAIN APP ==========

def main():
    # Header
    st.title("🏠 AI FlipFinder")
    st.markdown("**Analyze real estate investment opportunities with AI-powered comps and ROI calculations**")
    
    # Check API keys
    api_issues = check_api_keys()
    if api_issues:
        st.warning("\n".join(api_issues))
        if "❌" in api_issues[0]:
            st.error("Cannot proceed without BATCHDATA_API_KEY. Set it as an environment variable.")
            st.stop()
    
    # ========== INPUT FORM ==========
    st.markdown("---")
    st.subheader("📝 Property Information")
    
    # Address inputs
    col1, col2 = st.columns([3, 1])
    
    with col1:
        address = st.text_input(
            "Street Address",
            value="155 Palmer St NE",
            help="Enter the property street address"
        )
    
    with col2:
        zip_code = st.text_input(
            "ZIP Code",
            value="49505",
            max_chars=10
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        city = st.text_input(
            "City",
            value="Grand Rapids",
            help="City name"
        )
    
    with col4:
        state = st.text_input(
            "State",
            value="MI",
            max_chars=2,
            help="Two-letter state code"
        )
    
    # Property details
    st.markdown("**Property Details**")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        sqft = st.number_input(
            "Square Feet",
            min_value=100,
            max_value=10000,
            value=1116,
            step=50,
            help="Total living area"
        )
    
    with col6:
        beds = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
    
    with col7:
        baths = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1
        )
    
    # Financial inputs
    st.markdown("**Financial Information**")
    
    col8, col9 = st.columns(2)
    
    with col8:
        listing_price = st.number_input(
            "Listing Price ($)",
            min_value=0,
            max_value=10000000,
            value=160000,
            step=5000,
            help="Current asking price"
        )
    
    with col9:
        use_manual_rehab = st.checkbox("Use manual rehab estimate", value=False)
        
        if use_manual_rehab:
            rehab_cost = st.number_input(
                "Rehab Cost ($)",
                min_value=0,
                max_value=1000000,
                value=30000,
                step=1000,
                help="Estimated repair costs"
            )
        else:
            rehab_cost = None
            st.info("Will use default estimate (30/sqft)")
    
    # ========== ANALYZE BUTTON ==========
    st.markdown("---")
    
    if st.button("🔍 Analyze Deal", use_container_width=True):
        # Validate inputs
        if not address or not city or not state or not zip_code:
            st.error("❌ Please fill in all address fields")
            return
        
        # Run analysis with spinner
        with st.spinner("🔄 Running analysis... This may take 30-60 seconds"):
            try:
                results = run_analysis(
                    address=address,
                    city=city,
                    state=state,
                    zip_code=zip_code,
                    sqft=sqft,
                    listing_price=listing_price,
                    rehab_cost=rehab_cost,
                    beds=beds,
                    baths=baths
                )
                
                # Store results in session state
                st.session_state['results'] = results
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if 'results' in st.session_state:
        display_results(st.session_state['results'])
    
    # ========== FOOTER ==========
    st.markdown("---")
    st.caption("💡 **Tip:** For more accurate results, set your APIFY_API_TOKEN to enable Zillow enrichment")
    st.caption("📖 [Learn more about the 70% rule](https://www.biggerpockets.com/blog/70-percent-rule)")


if __name__ == "__main__":
    main()