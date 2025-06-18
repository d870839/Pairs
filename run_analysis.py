#!/usr/bin/env python3
"""
Pricing Analysis Runner
======================

This script runs the comprehensive pricing analysis package.
Simply replace Data.csv with your data file and run this script.

Usage:
    python run_analysis.py
"""

from pricing_analysis_package import PricingAnalyzer

def main():
    print("=== PRICING ANALYSIS RUNNER ===")
    print("This tool analyzes pricing relationships between manufacturers.")
    print("Kroger will always be one manufacturer, you'll select the second.\n")
    
    # Get data file name
    data_file = input("Enter your data file name (default: Data.csv): ").strip()
    if not data_file:
        data_file = "Data.csv"
    
    # Create analyzer
    analyzer = PricingAnalyzer(data_file)
    
    # Load data and get manufacturers
    try:
        manufacturers = analyzer.load_data()
        print(f"\nFound {len(manufacturers)} manufacturers in the data.")
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}")
        print("Please make sure the file exists in the current directory.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Get second manufacturer
    print("\nAvailable manufacturers:")
    for i, mfg in enumerate(manufacturers):
        if mfg.upper() != 'KROGER':
            print(f"  {i}: {mfg}")
    
    while True:
        try:
            choice = input(f"\nSelect second manufacturer (0-{len(manufacturers)-1}): ").strip()
            choice_idx = int(choice)
            if 0 <= choice_idx < len(manufacturers):
                manufacturer2 = manufacturers[choice_idx]
                if manufacturer2.upper() == 'KROGER':
                    print("Kroger is already the first manufacturer. Please select a different one.")
                    continue
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\nAnalyzing: KROGER vs {manufacturer2}")
    
    # Run analysis
    try:
        results = analyzer.run_analysis(manufacturer1='KROGER', manufacturer2=manufacturer2)
        
        if results is not None:
            print(f"\n✅ Analysis complete!")
            print(f"Results saved to: Pricing_Analysis_KROGER_{manufacturer2}.csv")
            print(f"Total item pairs analyzed: {len(results)}")
            
            # Show summary of valid recommendations
            valid_results = results[results['Mfg1_Recommended_Price'].notna() & 
                                   results['Mfg2_Recommended_Price'].notna()]
            
            if len(valid_results) > 0:
                print(f"Valid recommendations: {len(valid_results)}")
                print("\nSample recommendations:")
                for _, row in valid_results.head(3).iterrows():
                    print(f"  {row['Size_OZ']} OZ {row['Meat_Type']}: "
                          f"Kroger ${row['Mfg1_Recommended_Price']:.2f} vs "
                          f"{manufacturer2} ${row['Mfg2_Recommended_Price']:.2f}")
            else:
                print("No statistically valid recommendations found.")
        else:
            print("❌ Analysis failed or no matching items found.")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please check your data format and try again.")

if __name__ == "__main__":
    main() 