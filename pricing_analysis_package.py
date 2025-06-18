import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PricingAnalyzer:
    def __init__(self, data_file='Data.csv'):
        """Initialize the pricing analyzer with a data file"""
        self.data_file = data_file
        self.df = None
        self.price_data = None
        self.manufacturers = []
        
    def load_data(self):
        """Load and prepare the data"""
        print(f"Loading data from {self.data_file}...")
        self.df = pd.read_csv(self.data_file)
        
        # Extract size and meat type
        self.df['Size_OZ'] = self.df['UPC+Item Description'].apply(self.extract_size)
        self.df['Meat_Type'] = self.df['UPC+Item Description'].apply(self.extract_meat_type)
        
        # Only keep items with valid size
        self.df = self.df[self.df['Size_OZ'].notna()]
        
        # Get available manufacturers
        self.manufacturers = sorted(self.df['Manufacturer Description'].unique())
        print(f"Available manufacturers: {self.manufacturers}")
        
        return self.manufacturers
    
    def extract_size(self, desc):
        """
        Extract total size (OZ) from description, handling multi-unit packages.
        
        Handles various formats:
        - Simple: '2.6 OZ' -> 2.6
        - Multi-unit: '6CT 6/3 OZ' -> 18.0 (6 * 3)
        - Fractional: '4/4.5 OZ' -> 18.0 (4 * 4.5)
        - Pack notation: '3PK 3/10 OZ' -> 30.0 (3 * 10)
        - Pack notation: '4 PACK 4/5 OZ' -> 20.0 (4 * 5)
        """
        desc = str(desc).upper()
        
        # Pattern 1: Multi-unit with CT (count) like "6CT 6/3 OZ" or "10CT 10/1 OZ"
        ct_pattern = re.compile(r'(\d+)CT\s+(\d+)/(\d+(?:\.\d+)?)\s*OZ', re.IGNORECASE)
        ct_match = ct_pattern.search(desc)
        if ct_match:
            units = int(ct_match.group(2))
            size_per_unit = float(ct_match.group(3))
            return units * size_per_unit
        
        # Pattern 2: Multi-unit with PK (pack) like "3PK 3/10 OZ"
        pk_pattern = re.compile(r'(\d+)PK\s+(\d+)/(\d+(?:\.\d+)?)\s*OZ', re.IGNORECASE)
        pk_match = pk_pattern.search(desc)
        if pk_match:
            units = int(pk_match.group(2))
            size_per_unit = float(pk_match.group(3))
            return units * size_per_unit
        
        # Pattern 3: Multi-unit with PACK like "4 PACK 4/5 OZ"
        pack_pattern = re.compile(r'(\d+)\s+PACK\s+(\d+)/(\d+(?:\.\d+)?)\s*OZ', re.IGNORECASE)
        pack_match = pack_pattern.search(desc)
        if pack_match:
            units = int(pack_match.group(2))
            size_per_unit = float(pack_match.group(3))
            return units * size_per_unit
        
        # Pattern 4: Simple fraction like "4/4.5 OZ" (assumes 1 pack)
        fraction_pattern = re.compile(r'(\d+)/(\d+(?:\.\d+)?)\s*OZ', re.IGNORECASE)
        fraction_match = fraction_pattern.search(desc)
        if fraction_match:
            units = int(fraction_match.group(1))
            size_per_unit = float(fraction_match.group(2))
            return units * size_per_unit
        
        # Pattern 5: Multi-unit without explicit pack notation like "6 3 OZ"
        multi_pattern = re.compile(r'(\d+)\s+(\d+(?:\.\d+)?)\s*OZ', re.IGNORECASE)
        multi_match = multi_pattern.search(desc)
        if multi_match:
            units = int(multi_match.group(1))
            size_per_unit = float(multi_match.group(2))
            return units * size_per_unit
        
        # Pattern 6: Simple size like "2.6 OZ" (fallback)
        simple_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*OZ', re.IGNORECASE)
        simple_match = simple_pattern.search(desc)
        if simple_match:
            return float(simple_match.group(1))
        
        return None
    
    def extract_meat_type(self, desc):
        """Extract meat type from description"""
        desc = str(desc).upper()
        # Check for chicken: contains C, H, K in any combination
        if 'C' in desc and 'H' in desc and 'K' in desc:
            return 'CHKN'
        # Check for beef: contains B and F
        elif 'B' in desc and 'F' in desc:
            return 'BEEF'
        elif 'TUNA' in desc:
            return 'TUNA'
        elif 'SALMON' in desc:
            return 'SALMON'
        elif 'TURKEY' in desc or 'TRKY' in desc:
            return 'TURKEY'
        elif 'HAM' in desc:
            return 'HAM'
        else:
            return 'OTHER'
    
    def select_manufacturers(self, manufacturer1='KROGER', manufacturer2=None):
        """Select manufacturers for analysis"""
        if manufacturer2 is None:
            print(f"Please select the second manufacturer from: {self.manufacturers}")
            for i, mfg in enumerate(self.manufacturers):
                if mfg.upper() != manufacturer1.upper():
                    print(f"  {i}: {mfg}")
            return manufacturer1, None
        
        return manufacturer1.upper(), manufacturer2.upper()
    
    def prepare_weekly_data(self, manufacturer1, manufacturer2):
        """Prepare weekly data for analysis"""
        print("Preparing weekly data...")
        
        # Filter for selected manufacturers
        filtered_df = self.df[self.df['Manufacturer Description'].str.upper().isin([manufacturer1, manufacturer2])]
        
        # Identify week/date columns
        week_cols = filtered_df.columns.tolist()
        week_start = week_cols.index('Data') + 1
        week_cols = week_cols[week_start:]
        
        # Clean numeric data
        def clean_numeric(x):
            try:
                return float(str(x).replace('$', '').replace(',', '').replace('nan', '0'))
            except Exception:
                return 0.0
        
        for col in week_cols:
            filtered_df[col] = filtered_df[col].apply(clean_numeric)
        
        # Create weekly data for analysis
        weekly_data = []
        for _, row in filtered_df.iterrows():
            item_desc = row['UPC+Item Description']
            manufacturer = row['Manufacturer Description']
            size = row['Size_OZ']
            meat_type = row['Meat_Type']
            data_type = row['Data']
            
            for week_col in week_cols:
                value = row[week_col]
                if data_type == 'Scanned Retail $':
                    weekly_data.append({
                        'Item': item_desc,
                        'Manufacturer': manufacturer,
                        'Size_OZ': size,
                        'Meat_Type': meat_type,
                        'Week': week_col,
                        'Sales_Dollars': value,
                        'Data_Type': 'Sales'
                    })
                elif data_type == 'Scanned Movement':
                    # Find corresponding sales row
                    sales_row = filtered_df[(filtered_df['UPC+Item Description'] == item_desc) & 
                                          (filtered_df['Data'] == 'Scanned Retail $')]
                    if not sales_row.empty:
                        sales_value = sales_row[week_col].iloc[0]
                        if value > 0:  # Avoid division by zero
                            avg_price = sales_value / value
                        else:
                            avg_price = 0
                    else:
                        avg_price = 0
                        
                    weekly_data.append({
                        'Item': item_desc,
                        'Manufacturer': manufacturer,
                        'Size_OZ': size,
                        'Meat_Type': meat_type,
                        'Week': week_col,
                        'Units': value,
                        'Avg_Price': avg_price,
                        'Data_Type': 'Movement'
                    })
        
        weekly_df = pd.DataFrame(weekly_data)
        
        # Filter out zero units and calculate price per unit
        self.price_data = weekly_df[weekly_df['Data_Type'] == 'Movement'].copy()
        self.price_data = self.price_data[self.price_data['Units'] > 0]
        
        # Filter outliers (> 1.5 standard deviations)
        self.price_data = self.filter_outliers(self.price_data, 'Units')
        self.price_data = self.filter_outliers(self.price_data, 'Avg_Price')
        
        print(f"Data points after filtering: {len(self.price_data)}")
        return self.price_data
    
    def filter_outliers(self, df, column):
        """Filter outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def find_like_items(self):
        """Find like items between manufacturers based on size and meat type"""
        print("Finding like items...")
        
        # Get unique items by manufacturer
        mfg1_items = self.price_data[self.price_data['Manufacturer'].str.upper() == self.manufacturer1]['Item'].unique()
        mfg2_items = self.price_data[self.price_data['Manufacturer'].str.upper() == self.manufacturer2]['Item'].unique()
        
        print(f"{self.manufacturer1} items: {len(mfg1_items)}")
        print(f"{self.manufacturer2} items: {len(mfg2_items)}")
        
        # Find common size + meat type combinations
        mfg1_combos = set(self.price_data[self.price_data['Manufacturer'].str.upper() == self.manufacturer1][['Size_OZ', 'Meat_Type']].apply(tuple, axis=1))
        mfg2_combos = set(self.price_data[self.price_data['Manufacturer'].str.upper() == self.manufacturer2][['Size_OZ', 'Meat_Type']].apply(tuple, axis=1))
        common_combos = mfg1_combos.intersection(mfg2_combos)
        
        print(f"Common size + meat type combinations: {len(common_combos)}")
        
        # Create all possible pairs
        matches = []
        for size, meat_type in sorted(common_combos):
            mfg1_matches = self.price_data[(self.price_data['Manufacturer'].str.upper() == self.manufacturer1) & 
                                         (self.price_data['Size_OZ'] == size) & 
                                         (self.price_data['Meat_Type'] == meat_type)]['Item'].unique()
            mfg2_matches = self.price_data[(self.price_data['Manufacturer'].str.upper() == self.manufacturer2) & 
                                         (self.price_data['Size_OZ'] == size) & 
                                         (self.price_data['Meat_Type'] == meat_type)]['Item'].unique()
            
            for mfg1_item in mfg1_matches:
                for mfg2_item in mfg2_matches:
                    matches.append({
                        'Size_OZ': size,
                        'Meat_Type': meat_type,
                        'Mfg1_Item': mfg1_item,
                        'Mfg2_Item': mfg2_item
                    })
        
        return matches
    
    def get_valid_prices(self, min_price, max_price):
        """Generate valid prices ending in 9"""
        valid_prices = []
        for price in np.arange(min_price, max_price + 0.01, 0.01):
            price_rounded = round(price, 2)
            # Check if price ends in 9 (last digit is 9)
            if str(price_rounded).endswith('9'):
                valid_prices.append(price_rounded)
        return valid_prices
    
    def round_to_valid_price(self, price):
        """Round to nearest valid price ending in 9"""
        rounded = round(price, 2)
        
        # Find the nearest valid price ending in 9
        valid_prices = self.get_valid_prices(max(0.29, rounded - 0.20), rounded + 0.20)
        if not valid_prices:
            # If no valid prices in range, create some ending in 9
            base = int(rounded * 10) / 10
            valid_prices = [base + 0.09, base + 0.19, base + 0.29, base + 0.39, base + 0.49, 
                           base + 0.59, base + 0.69, base + 0.79, base + 0.89, base + 0.99]
        
        # Find closest valid price
        closest = min(valid_prices, key=lambda x: abs(x - rounded))
        return closest
    
    def calculate_p_value(self, X, y, model):
        """Calculate p-value for regression model"""
        try:
            # Fit the model
            model.fit(X, y)
            
            # Get predictions
            y_pred = model.predict(X)
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Calculate degrees of freedom
            n = len(X)
            p = X.shape[1]  # number of features
            df = n - p - 1
            
            # Calculate standard error
            mse = np.sum(residuals**2) / df
            se = np.sqrt(mse)
            
            # Calculate t-statistic for the coefficient
            if hasattr(model, 'coef_') and len(model.coef_) > 0:
                coef = model.coef_[0]
                # Calculate standard error of coefficient
                X_with_intercept = np.column_stack([np.ones(n), X])
                XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
                coef_se = np.sqrt(XtX_inv[1, 1] * mse)
                
                # Calculate t-statistic
                t_stat = coef / coef_se
                
                # Calculate p-value (two-tailed test)
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                
                return p_value
            else:
                return 1.0  # Return high p-value if no coefficient
        except:
            return 1.0  # Return high p-value if calculation fails
    
    def build_item_model(self, item_data, item_name):
        """Build regression model for a specific item with statistical validation"""
        if len(item_data) < 5:
            return None, None, None, "Insufficient data"
        
        X = item_data[['Avg_Price']].values
        y = item_data['Units'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {}
        results = {}
        
        # Linear Regression
        try:
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            r2_lr = r2_score(y_test, y_pred_lr)
            p_value_lr = self.calculate_p_value(X_train, y_train, lr)
            
            models['Linear'] = lr
            results['Linear'] = {
                'R2': r2_lr,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
                'Coefficient': lr.coef_[0],
                'Intercept': lr.intercept_,
                'P_Value': p_value_lr
            }
        except:
            pass
        
        # Ridge Regression
        try:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred_ridge = ridge.predict(X_test)
            r2_ridge = r2_score(y_test, y_pred_ridge)
            p_value_ridge = self.calculate_p_value(X_train, y_train, ridge)
            
            models['Ridge'] = ridge
            results['Ridge'] = {
                'R2': r2_ridge,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
                'Coefficient': ridge.coef_[0],
                'Intercept': ridge.intercept_,
                'P_Value': p_value_ridge
            }
        except:
            pass
        
        if not results:
            return None, None, None, "No valid models"
        
        # Find best model with validation
        valid_models = {}
        for model_name, metrics in results.items():
            if metrics['R2'] >= 0.5 and metrics['P_Value'] <= 0.05:
                valid_models[model_name] = metrics
        
        if not valid_models:
            return None, None, None, f"Model quality insufficient (R² < 0.5 or p > 0.05). Best R²: {max([m['R2'] for m in results.values()]):.4f}, Best p-value: {min([m['P_Value'] for m in results.values()]):.4f}"
        
        # Find best valid model
        best_model_name = max(valid_models.items(), key=lambda x: x[1]['R2'])[0]
        best_model = models[best_model_name]
        best_results = valid_models[best_model_name]
        
        return best_model, best_results, best_model_name, "Valid model"
    
    def predict_optimal_price(self, model, model_name, price_range, item_name):
        """Predict optimal price for revenue maximization"""
        predictions = []
        
        for price in price_range:
            try:
                pred = model.predict([[price]])[0]
                predictions.append(max(0, pred))
            except:
                predictions.append(0)
        
        # Calculate revenue
        revenues = [price * pred for price, pred in zip(price_range, predictions)]
        
        if not revenues or max(revenues) == 0:
            return None, None, None
        
        # Find optimal price
        max_revenue_idx = np.argmax(revenues)
        optimal_price = price_range[max_revenue_idx]
        max_revenue = revenues[max_revenue_idx]
        optimal_volume = predictions[max_revenue_idx]
        
        return optimal_price, optimal_volume, max_revenue
    
    def analyze_pair(self, mfg1_item, mfg2_item, size, meat_type):
        """Analyze a specific item pair"""
        print(f"\n=== ANALYZING PAIR: {size} OZ {meat_type} ===")
        print(f"{self.manufacturer1}: {mfg1_item}")
        print(f"{self.manufacturer2}: {mfg2_item}")
        
        # Get data for each item
        mfg1_data = self.price_data[(self.price_data['Item'] == mfg1_item) & 
                                   (self.price_data['Manufacturer'].str.upper() == self.manufacturer1)]
        mfg2_data = self.price_data[(self.price_data['Item'] == mfg2_item) & 
                                   (self.price_data['Manufacturer'].str.upper() == self.manufacturer2)]
        
        print(f"{self.manufacturer1} data points: {len(mfg1_data)}")
        print(f"{self.manufacturer2} data points: {len(mfg2_data)}")
        
        # Build models for each item
        mfg1_model, mfg1_results, mfg1_model_name, mfg1_status = self.build_item_model(mfg1_data, mfg1_item)
        mfg2_model, mfg2_results, mfg2_model_name, mfg2_status = self.build_item_model(mfg2_data, mfg2_item)
        
        # Define price ranges based on current prices
        mfg1_current_avg = mfg1_data['Avg_Price'].mean() if len(mfg1_data) > 0 else 1.15
        mfg2_current_avg = mfg2_data['Avg_Price'].mean() if len(mfg2_data) > 0 else 1.50
        
        mfg1_price_range = np.linspace(max(0.29, mfg1_current_avg * 0.7), mfg1_current_avg * 1.3, 20)
        mfg2_price_range = np.linspace(max(0.29, mfg2_current_avg * 0.7), mfg2_current_avg * 1.3, 20)
        
        # Get valid prices ending in 9
        mfg1_valid_prices = self.get_valid_prices(mfg1_price_range.min(), mfg1_price_range.max())
        mfg2_valid_prices = self.get_valid_prices(mfg2_price_range.min(), mfg2_price_range.max())
        
        mfg1_recommendation = None
        mfg2_recommendation = None
        
        # Generate recommendations
        if mfg1_model is not None:
            mfg1_optimal_price, mfg1_optimal_volume, mfg1_optimal_revenue = self.predict_optimal_price(
                mfg1_model, mfg1_model_name, mfg1_valid_prices, mfg1_item
            )
            if mfg1_optimal_price is not None:
                mfg1_recommendation = {
                    'Optimal_Price': self.round_to_valid_price(mfg1_optimal_price),
                    'Predicted_Volume': mfg1_optimal_volume,
                    'Predicted_Revenue': mfg1_optimal_revenue,
                    'Model_R2': mfg1_results['R2'],
                    'Model_P_Value': mfg1_results['P_Value'],
                    'Current_Avg_Price': mfg1_current_avg
                }
        
        if mfg2_model is not None:
            mfg2_optimal_price, mfg2_optimal_volume, mfg2_optimal_revenue = self.predict_optimal_price(
                mfg2_model, mfg2_model_name, mfg2_valid_prices, mfg2_item
            )
            if mfg2_optimal_price is not None:
                mfg2_recommendation = {
                    'Optimal_Price': self.round_to_valid_price(mfg2_optimal_price),
                    'Predicted_Volume': mfg2_optimal_volume,
                    'Predicted_Revenue': mfg2_optimal_revenue,
                    'Model_R2': mfg2_results['R2'],
                    'Model_P_Value': mfg2_results['P_Value'],
                    'Current_Avg_Price': mfg2_current_avg
                }
        
        # Create recommendation entry
        recommendation = {
            'Size_OZ': size,
            'Meat_Type': meat_type,
            'Mfg1_Item': mfg1_item,
            'Mfg2_Item': mfg2_item,
            'Mfg1_Current_Price': mfg1_current_avg,
            'Mfg2_Current_Price': mfg2_current_avg,
            'Mfg1_Recommended_Price': mfg1_recommendation['Optimal_Price'] if mfg1_recommendation else None,
            'Mfg2_Recommended_Price': mfg2_recommendation['Optimal_Price'] if mfg2_recommendation else None,
            'Mfg1_Predicted_Volume': mfg1_recommendation['Predicted_Volume'] if mfg1_recommendation else None,
            'Mfg2_Predicted_Volume': mfg2_recommendation['Predicted_Volume'] if mfg2_recommendation else None,
            'Mfg1_Predicted_Revenue': mfg1_recommendation['Predicted_Revenue'] if mfg1_recommendation else None,
            'Mfg2_Predicted_Revenue': mfg2_recommendation['Predicted_Revenue'] if mfg2_recommendation else None,
            'Mfg1_Model_R2': mfg1_recommendation['Model_R2'] if mfg1_recommendation else None,
            'Mfg2_Model_R2': mfg2_recommendation['Model_R2'] if mfg2_recommendation else None,
            'Mfg1_Model_P_Value': mfg1_recommendation['Model_P_Value'] if mfg1_recommendation else None,
            'Mfg2_Model_P_Value': mfg2_recommendation['Model_P_Value'] if mfg2_recommendation else None,
            'Mfg1_Model_Status': mfg1_status,
            'Mfg2_Model_Status': mfg2_status,
            'Price_Gap': (mfg2_recommendation['Optimal_Price'] - mfg1_recommendation['Optimal_Price']) if (mfg2_recommendation and mfg1_recommendation) else None
        }
        
        # Print recommendations
        print(f"{self.manufacturer1} Model Status: {mfg1_status}")
        if mfg1_recommendation:
            print(f"{self.manufacturer1} Recommendation:")
            print(f"  Current Price: ${mfg1_current_avg:.2f}")
            print(f"  Recommended Price: ${mfg1_recommendation['Optimal_Price']:.2f}")
            print(f"  Predicted Volume: {mfg1_recommendation['Predicted_Volume']:.1f}")
            print(f"  Predicted Revenue: ${mfg1_recommendation['Predicted_Revenue']:.2f}")
            print(f"  Model R²: {mfg1_recommendation['Model_R2']:.4f}")
            print(f"  Model P-Value: {mfg1_recommendation['Model_P_Value']:.4f}")
        else:
            print(f"{self.manufacturer1}: No valid recommendation - {mfg1_status}")
        
        print(f"{self.manufacturer2} Model Status: {mfg2_status}")
        if mfg2_recommendation:
            print(f"{self.manufacturer2} Recommendation:")
            print(f"  Current Price: ${mfg2_current_avg:.2f}")
            print(f"  Recommended Price: ${mfg2_recommendation['Optimal_Price']:.2f}")
            print(f"  Predicted Volume: {mfg2_recommendation['Predicted_Volume']:.1f}")
            print(f"  Predicted Revenue: ${mfg2_recommendation['Predicted_Revenue']:.2f}")
            print(f"  Model R²: {mfg2_recommendation['Model_R2']:.4f}")
            print(f"  Model P-Value: {mfg2_recommendation['Model_P_Value']:.4f}")
        else:
            print(f"{self.manufacturer2}: No valid recommendation - {mfg2_status}")
        
        if mfg1_recommendation and mfg2_recommendation:
            price_gap = mfg2_recommendation['Optimal_Price'] - mfg1_recommendation['Optimal_Price']
            print(f"Price Gap: ${price_gap:.2f}")
        
        return recommendation
    
    def run_analysis(self, manufacturer1='KROGER', manufacturer2=None):
        """Run the complete pricing analysis"""
        print("=== PRICING ANALYSIS PACKAGE ===")
        
        # Load data
        manufacturers = self.load_data()
        
        # Select manufacturers
        self.manufacturer1, self.manufacturer2 = self.select_manufacturers(manufacturer1, manufacturer2)
        if self.manufacturer2 is None:
            print("No second manufacturer selected. Exiting.")
            return None
        
        print(f"Analyzing: {self.manufacturer1} vs {self.manufacturer2}")
        
        # Prepare data
        self.prepare_weekly_data(self.manufacturer1, self.manufacturer2)
        
        # Find like items
        matches = self.find_like_items()
        
        if not matches:
            print("No matching items found between manufacturers.")
            return None
        
        print(f"Found {len(matches)} item pairs to analyze")
        
        # Analyze each pair
        recommendations = []
        for match in matches:
            recommendation = self.analyze_pair(
                match['Mfg1_Item'], 
                match['Mfg2_Item'], 
                match['Size_OZ'], 
                match['Meat_Type']
            )
            recommendations.append(recommendation)
        
        # Save results
        recommendations_df = pd.DataFrame(recommendations)
        output_file = f'Pricing_Analysis_{self.manufacturer1}_{self.manufacturer2}.csv'
        recommendations_df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n=== SUMMARY ===")
        print(f"Total item pairs analyzed: {len(recommendations)}")
        print(f"Results saved to: {output_file}")
        
        valid_recommendations = [r for r in recommendations if r['Mfg1_Recommended_Price'] is not None and r['Mfg2_Recommended_Price'] is not None]
        print(f"Valid recommendations: {len(valid_recommendations)}")
        
        if valid_recommendations:
            avg_price_gap = np.mean([r['Price_Gap'] for r in valid_recommendations])
            avg_mfg1_r2 = np.mean([r['Mfg1_Model_R2'] for r in valid_recommendations if r['Mfg1_Model_R2'] is not None])
            avg_mfg2_r2 = np.mean([r['Mfg2_Model_R2'] for r in valid_recommendations if r['Mfg2_Model_R2'] is not None])
            
            print(f"Average Price Gap: ${avg_price_gap:.2f}")
            print(f"Average {self.manufacturer1} Model R²: {avg_mfg1_r2:.4f}")
            print(f"Average {self.manufacturer2} Model R²: {avg_mfg2_r2:.4f}")
        
        return recommendations_df

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = PricingAnalyzer('Data.csv')
    
    # Run analysis (user will be prompted to select second manufacturer)
    results = analyzer.run_analysis(manufacturer1='KROGER', manufacturer2=None) 