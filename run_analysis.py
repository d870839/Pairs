#!/usr/bin/env python3
"""
Pricing Analysis Runner
======================

This script runs the comprehensive pricing analysis package.
Simply replace Data.csv with your data file and run this script.

The analysis compares Kroger against any other manufacturer in your data.
You'll be prompted to select which competitor to analyze.

Usage:
    python run_analysis.py
"""

from pricing_analysis_package import PricingAnalysis

def main():
    """Main function to run the pricing analysis"""
    analyzer = PricingAnalysis()
    
    # Load data with interactive file selection
    analyzer.load_data()
    
    # Run analysis with interactive manufacturer selection
    results = analyzer.run_analysis()

if __name__ == '__main__':
    main() 