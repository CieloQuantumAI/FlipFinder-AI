# FlipFinder-AI 🏠

Complete AI-powered real estate deal analyzer for house flippers. Analyzes property condition, fetches comps, calculates ARV, and provides ROI analysis using the 70% rule.

## 🎯 Features

- **Automated Comps Analysis**: Fetches comparable properties from BatchData
- **Zillow Enrichment**: Fills missing data by scraping Zillow via Apify
- **AI Condition Assessment**: Analyzes property photos using CLIP to estimate rehab costs
- **ARV Calculation**: Computes After Repair Value from comp data
- **70% Rule Analysis**: Calculates Maximum Allowable Offer (MAO)
- **ROI Projections**: Complete investment analysis with profit calculations
- **Deal Ratings**: Automated deal quality assessment

## 📁 Project Structure

```
FlipFinder-AI/
├── flipfinder_ai.py           # Main integrated script
├── ai_batchdata_comps.py      # Comps fetching & enrichment
├── ai_deal_analyzer.py        # ARV, ROI, and 70% rule calculations
├── ai_rehab_estimator_test.py # Standalone rehab estimator
├── images/                    # Place property photos here
└── results/                   # Output files (auto-created)
    ├── deal_summary_*.json
    ├── comps_*.csv
    └── rehab_estimate_*.json
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install requests apify-client torch transformers pillow scikit-learn pandas numpy tqdm
```

###