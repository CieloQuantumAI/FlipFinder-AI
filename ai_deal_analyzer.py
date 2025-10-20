"""
AI Deal Analyzer
----------------
Calculates ARV, ROI, and offer price using comps and rehab estimates.
Uses the 70% rule for house flipping analysis.
"""

import json
import statistics
from typing import List, Dict, Optional


class DealAnalyzer:
    """Analyzes real estate deals using comps and rehab costs."""
    
    def __init__(self):
        self.holding_months = 6  # Average holding period
        self.closing_cost_pct = 0.03  # 3% buyer closing costs
        self.selling_cost_pct = 0.08  # 8% selling costs (6% agent + 2% other)
        self.holding_cost_monthly = 500  # Insurance, utilities, taxes per month
    
    def calculate_arv(self, comps: List[Dict]) -> Dict:
        """
        Calculate After Repair Value (ARV) from comparable properties.
        
        Args:
            comps: List of comparable properties with price and sqft
            
        Returns:
            Dict with arv, price_per_sqft, and comp_count
        """
        valid_comps = []
        
        for comp in comps:
            price = comp.get('price')
            sqft = comp.get('sqft')
            
            # Validate data
            if price == "N/A" or sqft == "N/A":
                continue
            
            try:
                price = float(price)
                sqft = float(sqft)
                if price > 0 and sqft > 0:
                    price_per_sqft = price / sqft
                    valid_comps.append({
                        'price': price,
                        'sqft': sqft,
                        'price_per_sqft': price_per_sqft,
                        'address': comp.get('address', 'N/A')
                    })
            except (ValueError, TypeError):
                continue
        
        if not valid_comps:
            return {
                'arv': None,
                'price_per_sqft': None,
                'comp_count': 0,
                'error': 'No valid comps found'
            }
        
        # Calculate median price per sqft (more robust than mean)
        prices_per_sqft = [c['price_per_sqft'] for c in valid_comps]
        median_price_per_sqft = statistics.median(prices_per_sqft)
        
        # Also calculate mean for reference
        mean_price_per_sqft = statistics.mean(prices_per_sqft)
        
        return {
            'arv': None,  # Will be calculated when subject sqft is provided
            'median_price_per_sqft': round(median_price_per_sqft, 2),
            'mean_price_per_sqft': round(mean_price_per_sqft, 2),
            'comp_count': len(valid_comps),
            'comps_used': valid_comps,
            'price_range': {
                'min': round(min(prices_per_sqft), 2),
                'max': round(max(prices_per_sqft), 2)
            }
        }
    
    def calculate_70_percent_rule(self, arv: float, rehab_cost: float) -> Dict:
        """
        Apply the 70% rule to calculate Maximum Allowable Offer (MAO).
        
        Formula: MAO = (70% × ARV) - Rehab Cost
        
        Args:
            arv: After Repair Value
            rehab_cost: Estimated rehab/repair costs
            
        Returns:
            Dict with MAO and breakdown
        """
        mao = (0.70 * arv) - rehab_cost
        
        return {
            'max_allowable_offer': round(mao, 2),
            'arv_70_percent': round(0.70 * arv, 2),
            'rehab_cost': round(rehab_cost, 2),
            'formula': '(70% × ARV) - Rehab Cost'
        }
    
    def calculate_roi_analysis(
        self, 
        purchase_price: float,
        rehab_cost: float,
        arv: float,
        holding_months: Optional[int] = None
    ) -> Dict:
        """
        Calculate comprehensive ROI analysis.
        
        Args:
            purchase_price: Property purchase price
            rehab_cost: Estimated rehab costs
            arv: After Repair Value
            holding_months: Months to hold property (default: 6)
            
        Returns:
            Dict with ROI metrics and profit breakdown
        """
        if holding_months is None:
            holding_months = self.holding_months
        
        # Calculate all costs
        closing_costs = purchase_price * self.closing_cost_pct
        holding_costs = self.holding_cost_monthly * holding_months
        selling_costs = arv * self.selling_cost_pct
        
        total_investment = (
            purchase_price + 
            rehab_cost + 
            closing_costs + 
            holding_costs + 
            selling_costs
        )
        
        gross_profit = arv - total_investment
        roi_percent = (gross_profit / total_investment) * 100 if total_investment > 0 else 0
        
        return {
            'arv': round(arv, 2),
            'total_investment': round(total_investment, 2),
            'gross_profit': round(gross_profit, 2),
            'roi_percent': round(roi_percent, 2),
            'breakdown': {
                'purchase_price': round(purchase_price, 2),
                'rehab_cost': round(rehab_cost, 2),
                'closing_costs': round(closing_costs, 2),
                'holding_costs': round(holding_costs, 2),
                'selling_costs': round(selling_costs, 2)
            },
            'holding_months': holding_months
        }
    
    def calculate_offer_for_target_roi(
        self,
        arv: float,
        rehab_cost: float,
        target_roi_percent: float,
        holding_months: Optional[int] = None
    ) -> Dict:
        """
        Calculate the purchase price needed to achieve a target ROI.
        
        Args:
            arv: After Repair Value
            rehab_cost: Estimated rehab costs
            target_roi_percent: Target ROI percentage (e.g., 15 for 15%)
            holding_months: Months to hold property (default: 6)
            
        Returns:
            Dict with calculated purchase price and ROI analysis
        """
        if holding_months is None:
            holding_months = self.holding_months
        
        target_roi = target_roi_percent / 100
        holding_costs = self.holding_cost_monthly * holding_months
        selling_costs = arv * self.selling_cost_pct
        
        # Formula derivation:
        # ROI = (ARV - Total_Investment) / Total_Investment
        # Total_Investment = Purchase + Rehab + Closing + Holding + Selling
        # Closing = Purchase * closing_cost_pct
        # Solving for Purchase:
        # Purchase = (ARV - Rehab - Holding - Selling - (target_roi * (Rehab + Holding + Selling))) / 
        #            ((1 + closing_cost_pct) * (1 + target_roi))
        
        numerator = arv - rehab_cost - holding_costs - selling_costs - (target_roi * (rehab_cost + holding_costs + selling_costs))
        denominator = (1 + self.closing_cost_pct) * (1 + target_roi)
        
        purchase_price = numerator / denominator
        
        # Validate and calculate actual ROI
        if purchase_price <= 0:
            return {
                'target_roi_percent': target_roi_percent,
                'calculated_purchase_price': None,
                'error': f'Cannot achieve {target_roi_percent}% ROI with current ARV and costs'
            }
        
        # Calculate actual ROI with this purchase price
        roi_analysis = self.calculate_roi_analysis(purchase_price, rehab_cost, arv, holding_months)
        
        return {
            'target_roi_percent': target_roi_percent,
            'calculated_purchase_price': round(purchase_price, 2),
            'roi_analysis': roi_analysis
        }
    
    def generate_deal_summary(
        self,
        property_info: Dict,
        comps_data: Dict,
        rehab_estimate: Dict,
        listing_price: float
    ) -> Dict:
        """
        Generate comprehensive deal summary with recommendations.
        
        Args:
            property_info: Property details (address, sqft, etc.)
            comps_data: Comps analysis results
            rehab_estimate: Rehab cost estimate
            listing_price: Current listing price
            
        Returns:
            Complete deal summary with recommendations
        """
        # Calculate ARV using subject property sqft
        subject_sqft = property_info.get('sqft')
        if not subject_sqft or comps_data['comp_count'] == 0:
            return {
                'error': 'Insufficient data for analysis',
                'property': property_info
            }
        
        arv = comps_data['median_price_per_sqft'] * subject_sqft
        rehab_cost = rehab_estimate.get('est_rehab_cost', 0)
        
        # Calculate 70% rule
        mao_data = self.calculate_70_percent_rule(arv, rehab_cost)
        
        # Calculate ROI at listing price
        roi_at_listing = self.calculate_roi_analysis(
            listing_price,
            rehab_cost,
            arv
        )
        
        # Calculate ROI at MAO (ideal scenario)
        roi_at_mao = self.calculate_roi_analysis(
            mao_data['max_allowable_offer'],
            rehab_cost,
            arv
        )
        
        # Calculate offers for target ROIs: 10%, 15%, 20%, 25%
        target_roi_scenarios = {}
        for target_roi in [10, 15, 20, 25]:
            scenario = self.calculate_offer_for_target_roi(arv, rehab_cost, target_roi)
            target_roi_scenarios[f'roi_{target_roi}_percent'] = scenario
        
        # Determine deal quality
        price_difference = listing_price - mao_data['max_allowable_offer']
        price_difference_pct = (price_difference / listing_price) * 100
        
        if listing_price <= mao_data['max_allowable_offer']:
            deal_rating = "EXCELLENT"
            recommendation = "Strong Buy - Listing price is at or below MAO"
        elif price_difference_pct <= 10:
            deal_rating = "GOOD"
            recommendation = f"Good Deal - Negotiate down ${abs(price_difference):,.0f} ({abs(price_difference_pct):.1f}%)"
        elif price_difference_pct <= 20:
            deal_rating = "FAIR"
            recommendation = f"Marginal - Needs significant negotiation (${abs(price_difference):,.0f})"
        else:
            deal_rating = "POOR"
            recommendation = f"Pass - Overpriced by ${abs(price_difference):,.0f} ({abs(price_difference_pct):.1f}%)"
        
        return {
            'property': property_info,
            'arv': round(arv, 2),
            'listing_price': round(listing_price, 2),
            'max_allowable_offer': mao_data['max_allowable_offer'],
            'recommended_offer': round(mao_data['max_allowable_offer'] * 0.95, 2),  # 5% buffer
            'rehab_cost': round(rehab_cost, 2),
            'condition_rating': rehab_estimate.get('condition_rating', 'N/A'),
            'deal_rating': deal_rating,
            'recommendation': recommendation,
            'price_analysis': {
                'arv': round(arv, 2),
                'listing_price': round(listing_price, 2),
                'difference': round(price_difference, 2),
                'difference_percent': round(price_difference_pct, 2),
                'arv_to_listing_ratio': round((arv / listing_price) * 100, 2) if listing_price > 0 else 0
            },
            'roi_at_listing_price': roi_at_listing,
            'roi_at_mao': roi_at_mao,
            'target_roi_scenarios': target_roi_scenarios,
            'comps_summary': {
                'comp_count': comps_data['comp_count'],
                'median_price_per_sqft': comps_data['median_price_per_sqft'],
                'mean_price_per_sqft': comps_data['mean_price_per_sqft'],
                'price_range': comps_data['price_range']
            },
            'rehab_summary': {
                'condition_rating': rehab_estimate.get('condition_rating', 'N/A'),
                'damage_score': rehab_estimate.get('damage_score', 'N/A'),
                'estimated_cost': round(rehab_cost, 2),
                'notes': rehab_estimate.get('notes', 'N/A')
            }
        }


def format_deal_summary(summary: Dict) -> str:
    """Format deal summary for console output."""
    
    lines = [
        "\n" + "=" * 80,
        "📊 DEAL ANALYSIS SUMMARY",
        "=" * 80,
        "",
        "🏠 PROPERTY INFORMATION",
        f"   Address: {summary['property'].get('address', 'N/A')}",
        f"   Square Feet: {summary['property'].get('sqft', 'N/A'):,}",
        f"   Listing Price: ${summary['listing_price']:,.2f}",
        "",
        "💰 VALUATION",
        f"   After Repair Value (ARV): ${summary['arv']:,.2f}",
        f"   Median $/sqft from comps: ${summary['comps_summary']['median_price_per_sqft']:.2f}",
        f"   ARV to List Price Ratio: {summary['price_analysis']['arv_to_listing_ratio']:.1f}%",
        "",
        "🔧 REHAB ESTIMATE",
        f"   Condition Rating: {summary['condition_rating']}/5",
        f"   Estimated Rehab Cost: ${summary['rehab_cost']:,.2f}",
        f"   Notes: {summary['rehab_summary']['notes']}",
        "",
        "📈 70% RULE ANALYSIS",
        f"   Maximum Allowable Offer (MAO): ${summary['max_allowable_offer']:,.2f}",
        f"   Recommended Offer (95% of MAO): ${summary['recommended_offer']:,.2f}",
        f"   Listing vs MAO: ${summary['price_analysis']['difference']:,.2f} ({summary['price_analysis']['difference_percent']:.1f}%)",
        "",
        "💵 ROI ANALYSIS (At Listing Price)",
        f"   Purchase Price: ${summary['roi_at_listing_price']['breakdown']['purchase_price']:,.2f}",
        f"   Total Investment: ${summary['roi_at_listing_price']['total_investment']:,.2f}",
        f"   Gross Profit: ${summary['roi_at_listing_price']['gross_profit']:,.2f}",
        f"   ROI: {summary['roi_at_listing_price']['roi_percent']:.2f}%",
        "",
        "💵 ROI ANALYSIS (At MAO)",
        f"   Purchase Price: ${summary['roi_at_mao']['breakdown']['purchase_price']:,.2f}",
        f"   Total Investment: ${summary['roi_at_mao']['total_investment']:,.2f}",
        f"   Gross Profit: ${summary['roi_at_mao']['gross_profit']:,.2f}",
        f"   ROI: {summary['roi_at_mao']['roi_percent']:.2f}%",
        "",
    ]
    
    # Add target ROI scenarios
    lines.append("🎯 TARGET ROI SCENARIOS")
    for roi_pct in [10, 15, 20, 25]:
        scenario_key = f'roi_{roi_pct}_percent'
        if scenario_key in summary.get('target_roi_scenarios', {}):
            scenario = summary['target_roi_scenarios'][scenario_key]
            if scenario.get('calculated_purchase_price'):
                roi_data = scenario['roi_analysis']
                lines.extend([
                    f"",
                    f"   Target ROI: {roi_pct}%",
                    f"      Required Purchase Price: ${scenario['calculated_purchase_price']:,.2f}",
                    f"      Total Investment: ${roi_data['total_investment']:,.2f}",
                    f"      Gross Profit: ${roi_data['gross_profit']:,.2f}",
                    f"      Actual ROI: {roi_data['roi_percent']:.2f}%",
                ])
            else:
                lines.extend([
                    f"",
                    f"   Target ROI: {roi_pct}%",
                    f"      ⚠️ {scenario.get('error', 'Cannot achieve this ROI')}",
                ])
    
    lines.extend([
        "",
        "🎯 RECOMMENDATION",
        f"   Deal Rating: {summary['deal_rating']}",
        f"   Action: {summary['recommendation']}",
        "",
        "=" * 80,
    ])
    
    return "\n".join(lines)


# Test function
if __name__ == "__main__":
    # Example test data
    analyzer = DealAnalyzer()
    
    # Sample comps
    sample_comps = [
        {'price': 249000, 'sqft': 1250, 'address': '3 Ann St NE'},
        {'price': 182000, 'sqft': 1157, 'address': '19 Caledonia St NE'},
        {'price': 205000, 'sqft': 1200, 'address': '1738 Coit Ave NE'},
    ]
    
    # Calculate comps
    comps_data = analyzer.calculate_arv(sample_comps)
    print(f"✅ Median $/sqft: ${comps_data['median_price_per_sqft']}")
    
    # Sample property
    property_info = {
        'address': '155 Palmer St NE, Grand Rapids, MI',
        'sqft': 1116,
        'id': 'TEST001'
    }
    
    # Sample rehab estimate
    rehab_estimate = {
        'condition_rating': 3,
        'damage_score': 0.45,
        'est_rehab_cost': 33480,
        'notes': 'water damage (0.72); peeling paint (0.68)'
    }
    
    # Generate summary
    summary = analyzer.generate_deal_summary(
        property_info,
        comps_data,
        rehab_estimate,
        listing_price=160000
    )
    
    print(format_deal_summary(summary))
    
    # Save to JSON
    with open('deal_summary_test.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\n✅ Saved to deal_summary_test.json")