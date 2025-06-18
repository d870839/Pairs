# Pricing Analysis Package

A comprehensive Python package for analyzing pricing relationships between manufacturers using sales data. This tool automatically identifies like items, builds statistical models, and provides optimal pricing recommendations.

## Features

- **Automatic Item Matching**: Finds like items between manufacturers based on size and product type
- **Statistical Modeling**: Builds multiple regression models (Linear, Ridge) with validation
- **Optimal Pricing**: Calculates revenue-maximizing prices following retail pricing rules
- **Flexible Data Input**: Works with any CSV file in the specified format
- **Manufacturer Selection**: Automatically detects available manufacturers and allows user selection
- **Quality Validation**: Only uses statistically valid models (R² ≥ 0.5, p-value ≤ 0.05)

## Data Requirements

Your CSV file must have the following structure:

| Column | Description | Example |
|--------|-------------|---------|
| `UPC+Item Description` | Product description with size | "KROGER CHICKEN BREAST 2.6 OZ" |
| `Manufacturer Description` | Manufacturer name | "KROGER", "STARKIST" |
| `Data` | Data type | "Scanned Retail $" or "Scanned Movement" |
| `Week1`, `Week2`, ... | Weekly sales data | Numeric values |

### Data Format Example:
```csv
UPC+Item Description,Manufacturer Description,Data,Week1,Week2,Week3
KROGER CHICKEN BREAST 2.6 OZ,KROGER,Scanned Retail $,150.25,145.50,160.75
KROGER CHICKEN BREAST 2.6 OZ,KROGER,Scanned Movement,125,120,135
STARKIST CHICKEN POUCH 2.6 OZ,STARKIST,Scanned Retail $,200.00,195.25,210.50
STARKIST CHICKEN POUCH 2.6 OZ,STARKIST,Scanned Movement,85,82,90
```

## Installation

1. Ensure you have Python 3.7+ installed
2. Install required packages:
```bash
pip install pandas numpy scikit-learn scipy
```

## Usage

### Quick Start
```bash
python run_analysis.py
```

### Programmatic Usage
```python
from pricing_analysis_package import PricingAnalyzer

# Create analyzer
analyzer = PricingAnalyzer('your_data.csv')

# Run analysis
results = analyzer.run_analysis(
    manufacturer1='KROGER', 
    manufacturer2='STARKIST'
)
```

## How It Works

### 1. Data Processing
- Loads CSV data and extracts product information
- Identifies size (OZ) and product type from descriptions
- Cleans and validates numeric data
- Filters outliers using IQR method

### 2. Item Matching
- Matches items between manufacturers based on:
  - **Size**: Exact ounce match (e.g., 2.6 OZ)
  - **Product Type**: Chicken, Beef, Tuna, etc.
- Creates all possible pairs for analysis

### 3. Statistical Modeling
- Builds regression models for each item:
  - **Linear Regression**: Basic price-volume relationship
  - **Ridge Regression**: Regularized model to prevent overfitting
- Validates models using:
  - **R² ≥ 0.5**: Good fit to data
  - **p-value ≤ 0.05**: Statistically significant

### 4. Price Optimization
- Calculates optimal prices for revenue maximization
- Applies retail pricing rules (prices ending in 9)
- Excludes invalid endings (.09, .19)

### 5. Recommendations
- Provides detailed pricing recommendations
- Shows model quality metrics
- Calculates price gaps between manufacturers

## Output

The analysis generates a CSV file with comprehensive results:

| Column | Description |
|--------|-------------|
| `Size_OZ` | Product size in ounces |
| `Meat_Type` | Product type (CHKN, BEEF, etc.) |
| `Mfg1_Item` | First manufacturer item description |
| `Mfg2_Item` | Second manufacturer item description |
| `Mfg1_Current_Price` | Current average price |
| `Mfg2_Current_Price` | Current average price |
| `Mfg1_Recommended_Price` | Optimal recommended price |
| `Mfg2_Recommended_Price` | Optimal recommended price |
| `Mfg1_Predicted_Volume` | Predicted sales volume |
| `Mfg2_Predicted_Volume` | Predicted sales volume |
| `Mfg1_Predicted_Revenue` | Predicted revenue |
| `Mfg2_Predicted_Revenue` | Predicted revenue |
| `Mfg1_Model_R2` | Model R-squared value |
| `Mfg2_Model_R2` | Model R-squared value |
| `Mfg1_Model_P_Value` | Model p-value |
| `Mfg2_Model_P_Value` | Model p-value |
| `Mfg1_Model_Status` | Model validation status |
| `Mfg2_Model_Status` | Model validation status |
| `Price_Gap` | Difference between recommended prices |

## Product Type Detection

The system automatically detects product types from descriptions:

- **CHKN**: Contains C, H, K (chicken)
- **BEEF**: Contains B, F (beef)
- **TUNA**: Contains "TUNA"
- **SALMON**: Contains "SALMON"
- **TURKEY**: Contains "TURKEY" or "TRKY"
- **HAM**: Contains "HAM"
- **OTHER**: All other products

## Supported Size Formats (Flexible Size Extraction)

The package supports a wide range of size formats, including multi-unit and pack formats. The extraction logic is designed to be flexible and extensible for new item formats.

### **Supported Examples:**

| Example Description                        | Interpreted As         | Extracted Size (OZ) |
|---------------------------------------------|------------------------|---------------------|
| `2.6 OZ`                                   | Single item            | 2.6                 |
| `6CT 6/3 OZ`                               | 6 units of 3 oz        | 18.0                |
| `4/4.5 OZ`                                 | 4 units of 4.5 oz      | 18.0                |
| `3PK 3/10 OZ`                              | 3 units of 10 oz       | 30.0                |
| `4 PACK 4/5 OZ`                            | 4 units of 5 oz        | 20.0                |
| `6 3 OZ`                                   | 6 units of 3 oz        | 18.0                |
| `10CT 10/1 OZ`                             | 10 units of 1 oz       | 10.0                |
| `KROGER PREMIUM CHICKEN BREAST 3PK 3/10 OZ PACKAGE` | 3 units of 10 oz | 30.0                |

- If no recognizable size is found, the value will be `None` and the item will be excluded from matching.
- The logic is easy to extend for new patterns—just add a new regex pattern to the extraction method.

## Pricing Rules

- All recommended prices end in 9 (e.g., $1.09, $1.19, $1.29)
- Excludes .09 and .19 endings
- Minimum price: $0.29
- Prices are rounded to nearest valid option

## Model Validation

Only statistically valid models are used for recommendations:

- **R² ≥ 0.5**: Model explains at least 50% of variance
- **p-value ≤ 0.05**: Relationship is statistically significant
- **Minimum 5 data points**: Sufficient data for modeling

## Example Output

```
=== ANALYZING PAIR: 2.6 OZ CHKN ===
KROGER: KROGER CHICKEN BREAST 2.6 OZ
STARKIST: STARKIST CHICKEN POUCH 2.6 OZ

KROGER Model Status: Valid model
KROGER Recommendation:
  Current Price: $1.15
  Recommended Price: $1.09
  Predicted Volume: 125.3
  Predicted Revenue: $136.58
  Model R²: 0.6234
  Model P-Value: 0.0234

STARKIST Model Status: Valid model
STARKIST Recommendation:
  Current Price: $1.85
  Recommended Price: $1.99
  Predicted Volume: 95.7
  Predicted Revenue: $190.44
  Model R²: 0.5876
  Model P-Value: 0.0187

Price Gap: $0.90
```

## Troubleshooting

### Common Issues

1. **"No matching items found"**
   - Check that both manufacturers have items with same size and type
   - Verify product descriptions contain size information (e.g., "2.6 OZ")

2. **"Model quality insufficient"**
   - Insufficient data points or weak price-volume relationships
   - Consider longer time periods or different product categories

3. **"File not found"**
   - Ensure CSV file is in the same directory as the script
   - Check file name spelling and case sensitivity

### Data Quality Tips

- Ensure consistent product naming conventions
- Include size information in product descriptions
- Provide sufficient historical data (at least 8-12 weeks)
- Clean data for missing values and outliers

## License

This package is provided as-is for educational and business analysis purposes.

## Support

For questions or issues, please check:
1. Data format requirements
2. Python package dependencies
3. File permissions and paths 