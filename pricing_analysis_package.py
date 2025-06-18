import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def jaro_winkler_similarity(s1, s2):
    """
    Calculate Jaro-Winkler similarity between two strings.
    Returns a value between 0 and 1, where 1 is identical.
    """
    def jaro_similarity(s1, s2):
        if s1 == s2:
            return 1.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # Maximum distance for matching characters
        match_distance = (max(len1, len2) // 2) - 1
        if match_distance < 0:
            match_distance = 0
        
        # Find matching characters
        s1_matches = [False] * len1
        s2_matches = [False] * len2
        
        matches = 0
        transpositions = 0
        
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            
            for j in range(start, end):
                if s2_matches[j]:
                    continue
                if s1[i] != s2[j]:
                    continue
                s1_matches[i] = s2_matches[j] = True
                matches += 1
                break
        
        if matches == 0:
            return 0.0
        
        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1
        
        transpositions //= 2
        
        return (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3
    
    # Calculate Jaro similarity
    jaro_sim = jaro_similarity(s1, s2)
    
    # Calculate Winkler modification
    prefix = 0
    for i in range(min(len(s1), len(s2), 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    
    # Winkler modification factor
    winkler_factor = 0.1
    
    return jaro_sim + (prefix * winkler_factor * (1 - jaro_sim))

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
        
        # Extract size
        self.df['Size_OZ'] = self.df['UPC+Item Description'].apply(self.extract_size)
        
        # Only keep items with valid size
        self.df = self.df[self.df['Size_OZ'].notna()]
        
        # Get available manufacturers
        self.manufacturers = sorted(self.df['Manufacturer Description'].unique())
        print(f"Available manufacturers: {self.manufacturers}")
        
        return self.manufacturers
    
    def extract_size(self, desc):
        """
        Extract total size (OZ/FO) from description, handling multi-unit packages.
        
        Handles various formats:
        - Simple: '2.6 OZ' or '16 FO' -> 2.6 or 16.0
        - Multi-unit: '6CT 6/3 OZ' -> 18.0 (6 * 3)
        - Fractional: '4/4.5 OZ' -> 18.0 (4 * 4.5)
        - Pack notation: '3PK 3/10 OZ' -> 30.0 (3 * 10)
        - Pack notation: '4 PACK 4/5 OZ' -> 20.0 (4 * 5)
        - Fluid ounces: '4/16 FO' -> 64.0 (4 * 16)
        """
        desc = str(desc).upper()
        
        # Pattern 1: Multi-unit with CT (count) like "6CT 6/3 OZ" or "10CT 10/1 FO"
        ct_pattern = re.compile(r'(\d+)CT\s+(\d+)/(\d+(?:\.\d+)?)\s*(OZ|FO)', re.IGNORECASE)
        ct_match = ct_pattern.search(desc)
        if ct_match:
            units = int(ct_match.group(2))
            size_per_unit = float(ct_match.group(3))
            return units * size_per_unit
        
        # Pattern 2: Multi-unit with PK (pack) like "3PK 3/10 OZ" or "4PK 4/16 FO"
        pk_pattern = re.compile(r'(\d+)PK\s+(\d+)/(\d+(?:\.\d+)?)\s*(OZ|FO)', re.IGNORECASE)
        pk_match = pk_pattern.search(desc)
        if pk_match:
            units = int(pk_match.group(2))
            size_per_unit = float(pk_match.group(3))
            return units * size_per_unit
        
        # Pattern 3: Multi-unit with PACK like "4 PACK 4/5 OZ" or "4 PACK 4/16 FO"
        pack_pattern = re.compile(r'(\d+)\s+PACK\s+(\d+)/(\d+(?:\.\d+)?)\s*(OZ|FO)', re.IGNORECASE)
        pack_match = pack_pattern.search(desc)
        if pack_match:
            units = int(pack_match.group(2))
            size_per_unit = float(pack_match.group(3))
            return units * size_per_unit
        
        # Pattern 4: Simple fraction like "4/4.5 OZ" or "4/16 FO" (assumes 1 pack)
        fraction_pattern = re.compile(r'(\d+)/(\d+(?:\.\d+)?)\s*(OZ|FO)', re.IGNORECASE)
        fraction_match = fraction_pattern.search(desc)
        if fraction_match:
            units = int(fraction_match.group(1))
            size_per_unit = float(fraction_match.group(2))
            return units * size_per_unit
        
        # Pattern 5: Multi-unit without explicit pack notation like "6 3 OZ" or "6 16 FO"
        multi_pattern = re.compile(r'(\d+)\s+(\d+(?:\.\d+)?)\s*(OZ|FO)', re.IGNORECASE)
        multi_match = multi_pattern.search(desc)
        if multi_match:
            units = int(multi_match.group(1))
            size_per_unit = float(multi_match.group(2))
            return units * size_per_unit
        
        # Pattern 6: Simple size like "2.6 OZ" or "16 FO" (fallback)
        simple_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(OZ|FO)', re.IGNORECASE)
        simple_match = simple_pattern.search(desc)
        if simple_match:
            return float(simple_match.group(1))
        
        return None
    
    def select_manufacturers(self, manufacturer1='KROGER', manufacturer2=None):
        """Select manufacturers for analysis"""
        if manufacturer2 is None:
            print(f"Please select the second manufacturer from: {self.manufacturers}")
            available_competitors = [mfg for mfg in self.manufacturers if mfg.upper() != manufacturer1.upper()]
            for i, mfg in enumerate(available_competitors):
                print(f"  {i}: {mfg}")
            print(f"\nNote: {len(self.manufacturers)} manufacturers have items with valid sizes.")
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
            data_type = row['Data']
            
            for week_col in week_cols:
                value = row[week_col]
                if data_type == 'Scanned Retail $':
                    weekly_data.append({
                        'Item': item_desc,
                        'Manufacturer': manufacturer,
                        'Size_OZ': size,
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
    
    def find_like_items(self, similarity_threshold=0.6):
        """
        Find like items between manufacturers using Jaro-Winkler similarity and size matching.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) for item matching
        """
        print("Finding like items using string similarity and size matching...")
        
        # Get unique items by manufacturer
        mfg1_items = self.price_data[self.price_data['Manufacturer'].str.upper() == self.manufacturer1]['Item'].unique()
        mfg2_items = self.price_data[self.price_data['Manufacturer'].str.upper() == self.manufacturer2]['Item'].unique()
        
        print(f"{self.manufacturer1} items: {len(mfg1_items)}")
        print(f"{self.manufacturer2} items: {len(mfg2_items)}")
        
        # Get size data for each item
        mfg1_sizes = {}
        mfg2_sizes = {}
        
        for item in mfg1_items:
            item_data = self.price_data[self.price_data['Item'] == item]
            if len(item_data) > 0:
                mfg1_sizes[item] = item_data['Size_OZ'].iloc[0]
        
        for item in mfg2_items:
            item_data = self.price_data[self.price_data['Item'] == item]
            if len(item_data) > 0:
                mfg2_sizes[item] = item_data['Size_OZ'].iloc[0]
        
        # Find matches using similarity and size
        matches = []
        total_comparisons = 0
        
        for mfg1_item, mfg1_size in mfg1_sizes.items():
            for mfg2_item, mfg2_size in mfg2_sizes.items():
                total_comparisons += 1
                
                # Size must match exactly
                if mfg1_size != mfg2_size:
                    continue
                
                # Calculate string similarity
                similarity = jaro_winkler_similarity(mfg1_item, mfg2_item)
                
                # Check if similarity meets threshold
                if similarity >= similarity_threshold:
                    matches.append({
                        'Size_OZ': mfg1_size,
                        'Mfg1_Item': mfg1_item,
                        'Mfg2_Item': mfg2_item,
                        'Similarity_Score': similarity
                    })
        
        # Sort matches by similarity score (highest first)
        matches.sort(key=lambda x: x['Similarity_Score'], reverse=True)
        
        print(f"Total item comparisons: {total_comparisons}")
        print(f"Matches found with similarity ≥ {similarity_threshold}: {len(matches)}")
        
        if matches:
            print(f"Top 3 matches:")
            for i, match in enumerate(matches[:3]):
                print(f"  {i+1}. Similarity: {match['Similarity_Score']:.3f}")
                print(f"     {self.manufacturer1}: {match['Mfg1_Item']}")
                print(f"     {self.manufacturer2}: {match['Mfg2_Item']}")
                print(f"     Size: {match['Size_OZ']} OZ")
        
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
        except Exception:
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
        
        # 1. Linear Regression
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
                'P_Value': p_value_lr,
                'Type': 'Linear'
            }
        except Exception:
            pass
        
        # 2. Ridge Regression
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
                'P_Value': p_value_ridge,
                'Type': 'Linear'
            }
        except Exception:
            pass
        
        # 3. Log-Linear Model (log of units vs price)
        try:
            # Ensure positive values for log transformation
            y_train_log = np.log(np.maximum(y_train, 0.1))
            y_test_log = np.log(np.maximum(y_test, 0.1))
            
            lr_log = LinearRegression()
            lr_log.fit(X_train, y_train_log)
            y_pred_log = lr_log.predict(X_test)
            
            # Convert back to original scale for R² calculation
            y_pred_original = np.exp(y_pred_log)
            r2_log = r2_score(y_test, y_pred_original)
            p_value_log = self.calculate_p_value(X_train, y_train_log, lr_log)
            
            models['Log_Linear'] = lr_log
            results['Log_Linear'] = {
                'R2': r2_log,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_original)),
                'Coefficient': lr_log.coef_[0],
                'Intercept': lr_log.intercept_,
                'P_Value': p_value_log,
                'Type': 'Log_Linear'
            }
        except Exception:
            pass
        
        # 4. Polynomial Model (Quadratic)
        try:
            # Create polynomial features
            X_train_poly = np.column_stack([X_train, X_train**2])
            X_test_poly = np.column_stack([X_test, X_test**2])
            
            poly_model = LinearRegression()
            poly_model.fit(X_train_poly, y_train)
            y_pred_poly = poly_model.predict(X_test_poly)
            r2_poly = r2_score(y_test, y_pred_poly)
            p_value_poly = self.calculate_p_value(X_train_poly, y_train, poly_model)
            
            models['Polynomial'] = poly_model
            results['Polynomial'] = {
                'R2': r2_poly,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_poly)),
                'Coefficient': poly_model.coef_[0],
                'Intercept': poly_model.intercept_,
                'P_Value': p_value_poly,
                'Type': 'Polynomial'
            }
        except Exception:
            pass
        
        # 5. Exponential Model (log of price vs units)
        try:
            # Ensure positive prices for log transformation
            X_train_log = np.log(np.maximum(X_train, 0.1))
            X_test_log = np.log(np.maximum(X_test, 0.1))
            
            exp_model = LinearRegression()
            exp_model.fit(X_train_log, y_train)
            y_pred_exp = exp_model.predict(X_test_log)
            r2_exp = r2_score(y_test, y_pred_exp)
            p_value_exp = self.calculate_p_value(X_train_log, y_train, exp_model)
            
            models['Exponential'] = exp_model
            results['Exponential'] = {
                'R2': r2_exp,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_exp)),
                'Coefficient': exp_model.coef_[0],
                'Intercept': exp_model.intercept_,
                'P_Value': p_value_exp,
                'Type': 'Exponential'
            }
        except Exception:
            pass
        
        # 6. Power Model (log-log transformation)
        try:
            # Ensure positive values for log transformation
            X_train_log = np.log(np.maximum(X_train, 0.1))
            X_test_log = np.log(np.maximum(X_test, 0.1))
            y_train_log = np.log(np.maximum(y_train, 0.1))
            y_test_log = np.log(np.maximum(y_test, 0.1))
            
            power_model = LinearRegression()
            power_model.fit(X_train_log, y_train_log)
            y_pred_log_power = power_model.predict(X_test_log)
            
            # Convert back to original scale for R² calculation
            y_pred_power = np.exp(y_pred_log_power)
            r2_power = r2_score(y_test, y_pred_power)
            p_value_power = self.calculate_p_value(X_train_log, y_train_log, power_model)
            
            models['Power'] = power_model
            results['Power'] = {
                'R2': r2_power,
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_power)),
                'Coefficient': power_model.coef_[0],
                'Intercept': power_model.intercept_,
                'P_Value': p_value_power,
                'Type': 'Power'
            }
        except Exception:
            pass
        
        if not results:
            return None, None, None, "No valid models"
        
        # Find best model with validation
        valid_models = {}
        for model_name, metrics in results.items():
            if metrics['R2'] >= 0.25 and metrics['P_Value'] <= 0.05:
                valid_models[model_name] = metrics
        
        if not valid_models:
            return None, None, None, f"Model quality insufficient (R² < 0.25 or p > 0.05). Best R²: {max([m['R2'] for m in results.values()]):.4f}, Best p-value: {min([m['P_Value'] for m in results.values()]):.4f}"
        
        # Find best valid model (highest R²)
        best_model_name = max(valid_models.items(), key=lambda x: x[1]['R2'])[0]
        best_model = models[best_model_name]
        best_results = valid_models[best_model_name]
        
        return best_model, best_results, best_model_name, f"Valid {best_results['Type']} model"
    
    def predict_optimal_price(self, model, model_name, price_range, item_name):
        """Predict optimal price for revenue maximization"""
        predictions = []
        
        for price in price_range:
            try:
                if model_name == 'Linear' or model_name == 'Ridge':
                    # Standard linear prediction
                    pred = model.predict([[price]])[0]
                
                elif model_name == 'Log_Linear':
                    # Log-linear: predict log(units), then convert back
                    log_pred = model.predict([[price]])[0]
                    pred = np.exp(log_pred)
                
                elif model_name == 'Polynomial':
                    # Polynomial: include quadratic term
                    pred = model.predict([[price, price**2]])[0]
                
                elif model_name == 'Exponential':
                    # Exponential: log of price
                    log_price = np.log(max(price, 0.1))
                    pred = model.predict([[log_price]])[0]
                
                elif model_name == 'Power':
                    # Power: log-log transformation
                    log_price = np.log(max(price, 0.1))
                    log_pred = model.predict([[log_price]])[0]
                    pred = np.exp(log_pred)
                
                else:
                    # Fallback to linear
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
    
    def analyze_pair(self, mfg1_item, mfg2_item, size, similarity_score):
        """Analyze a specific item pair"""
        print(f"\n=== ANALYZING PAIR: {size} OZ ===")
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
            'Mfg1_Item': mfg1_item,
            'Mfg2_Item': mfg2_item,
            'Similarity_Score': similarity_score,
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
        print(f"\n{self.manufacturer1} Model Status: {mfg1_status}")
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
                match['Similarity_Score']
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
    
    def calculate_cross_price_elasticity(self, mfg1_data, mfg2_data, mfg1_item, mfg2_item):
        """
        Calculate cross-price elasticity between two competing items.
        
        Cross-price elasticity measures how a 1% change in one product's price
        affects the demand for the competing product.
        
        Returns:
        - mfg1_to_mfg2_elasticity: How mfg1's price affects mfg2's demand
        - mfg2_to_mfg1_elasticity: How mfg2's price affects mfg1's demand
        - significance levels and R² values
        """
        if len(mfg1_data) < 5 or len(mfg2_data) < 5:
            return None, None, "Insufficient data for cross-price analysis"
        
        # Align data by week
        mfg1_weekly = mfg1_data.groupby('Week').agg({
            'Avg_Price': 'mean',
            'Units': 'sum'
        }).reset_index()
        
        mfg2_weekly = mfg2_data.groupby('Week').agg({
            'Avg_Price': 'mean',
            'Units': 'sum'
        }).reset_index()
        
        # Merge on week to get aligned price and demand data
        combined_data = pd.merge(mfg1_weekly, mfg2_weekly, on='Week', suffixes=('_mfg1', '_mfg2'))
        
        if len(combined_data) < 5:
            return None, None, "Insufficient aligned data points"
        
        # Calculate cross-price elasticities
        results = {}
        
        # Model 1: How mfg1's price affects mfg2's demand
        try:
            X1 = combined_data[['Avg_Price_mfg1']].values
            y1 = combined_data['Units_mfg2'].values
            
            # Add constant for regression
            X1_with_const = np.column_stack([np.ones(len(X1)), X1])
            
            # Calculate elasticity at mean values
            mean_price_mfg1 = combined_data['Avg_Price_mfg1'].mean()
            mean_units_mfg2 = combined_data['Units_mfg2'].mean()
            
            # Linear regression
            beta1 = np.linalg.lstsq(X1_with_const, y1, rcond=None)[0][1]
            elasticity1 = beta1 * (mean_price_mfg1 / mean_units_mfg2)
            
            # Calculate R² and p-value
            y_pred1 = X1_with_const @ np.linalg.lstsq(X1_with_const, y1, rcond=None)[0]
            r2_1 = 1 - np.sum((y1 - y_pred1)**2) / np.sum((y1 - np.mean(y1))**2)
            
            # Calculate p-value
            residuals1 = y1 - y_pred1
            n1 = len(X1)
            p1 = X1_with_const.shape[1]
            df1 = n1 - p1
            mse1 = np.sum(residuals1**2) / df1
            XtX_inv1 = np.linalg.inv(X1_with_const.T @ X1_with_const)
            se_beta1 = np.sqrt(XtX_inv1[1, 1] * mse1)
            t_stat1 = beta1 / se_beta1
            p_value1 = 2 * (1 - stats.t.cdf(abs(t_stat1), df1))
            
            results['mfg1_to_mfg2'] = {
                'elasticity': elasticity1,
                'r2': r2_1,
                'p_value': p_value1,
                'significant': p_value1 <= 0.05
            }
        except Exception:
            results['mfg1_to_mfg2'] = {
                'elasticity': None,
                'r2': None,
                'p_value': None,
                'significant': False
            }
        
        # Model 2: How mfg2's price affects mfg1's demand
        try:
            X2 = combined_data[['Avg_Price_mfg2']].values
            y2 = combined_data['Units_mfg1'].values
            
            # Add constant for regression
            X2_with_const = np.column_stack([np.ones(len(X2)), X2])
            
            # Calculate elasticity at mean values
            mean_price_mfg2 = combined_data['Avg_Price_mfg2'].mean()
            mean_units_mfg1 = combined_data['Units_mfg1'].mean()
            
            # Linear regression
            beta2 = np.linalg.lstsq(X2_with_const, y2, rcond=None)[0][1]
            elasticity2 = beta2 * (mean_price_mfg2 / mean_units_mfg1)
            
            # Calculate R² and p-value
            y_pred2 = X2_with_const @ np.linalg.lstsq(X2_with_const, y2, rcond=None)[0]
            r2_2 = 1 - np.sum((y2 - y_pred2)**2) / np.sum((y2 - np.mean(y2))**2)
            
            # Calculate p-value
            residuals2 = y2 - y_pred2
            n2 = len(X2)
            p2 = X2_with_const.shape[1]
            df2 = n2 - p2
            mse2 = np.sum(residuals2**2) / df2
            XtX_inv2 = np.linalg.inv(X2_with_const.T @ X2_with_const)
            se_beta2 = np.sqrt(XtX_inv2[1, 1] * mse2)
            t_stat2 = beta2 / se_beta2
            p_value2 = 2 * (1 - stats.t.cdf(abs(t_stat2), df2))
            
            results['mfg2_to_mfg1'] = {
                'elasticity': elasticity2,
                'r2': r2_2,
                'p_value': p_value2,
                'significant': p_value2 <= 0.05
            }
        except Exception:
            results['mfg2_to_mfg1'] = {
                'elasticity': None,
                'r2': None,
                'p_value': None,
                'significant': False
            }
        
        return results, "Cross-price elasticity calculated", None

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = PricingAnalyzer('Data.csv')
    
    # Run analysis (user will be prompted to select second manufacturer)
    results = analyzer.run_analysis(manufacturer1='KROGER', manufacturer2=None) 