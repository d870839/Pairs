import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import os
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

class PricingAnalysis:
    """
    A class for analyzing pricing relationships between manufacturers.
    Supports finding like items, analyzing price elasticity, and generating recommendations.
    """
    def __init__(self):
        """Initialize the pricing analyzer"""
        self.df = None
        self.price_data = None
        self.manufacturer1 = None
        self.manufacturer2 = None
        
    def load_data(self, filename=None):
        """Load and prepare the data for analysis"""
        if filename is None:
            filename = self.select_data_file()
            if filename is None:
                return
                
        print(f"Loading data from {filename}...")
        try:
            self.df = pd.read_csv(filename)
            
            # Create consolidated item name based on RBP grouping
            def create_item_name(row):
                rbp_code = row['Rules Based Pricing Code']
                rbp_desc = str(row['Rules Based Pricing Description']).strip()
                upc_desc = str(row['UPC+Item Description']).strip()
                
                # If RBP Code is null/empty/NaN, use UPC+Item Description
                # Otherwise, use RBP Description
                if pd.isna(rbp_code) or str(rbp_code).strip() == '' or str(rbp_code).strip().lower() == 'nan':
                    return upc_desc
                else:
                    return rbp_desc
            
            # Create consolidated item name
            self.df['Consolidated_Item'] = self.df.apply(create_item_name, axis=1)
            
            # Extract size information
            self.df['Size_OZ'] = self.df['UPC+Item Description'].apply(self.extract_size)
            
            print(f"Data loaded: {len(self.df)} rows")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
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
    
    def select_manufacturers(self, manufacturer1=None, manufacturer2=None):
        """Let user select manufacturers to compare"""
        # Get unique manufacturers
        available_manufacturers = sorted(self.df['Manufacturer Description'].unique())
        print("\nAvailable manufacturers:", available_manufacturers)
        
        # Select first manufacturer if not provided
        if manufacturer1 is None:
            print("\nSelect first manufacturer:")
            for i, mfg in enumerate(available_manufacturers, 1):
                print(f"{i}. {mfg}")
                
            while True:
                try:
                    choice = input("Enter manufacturer number (or press Enter for KROGER): ").strip()
                    if not choice:  # Default to KROGER
                        manufacturer1 = 'KROGER'
                        break
                        
                    choice = int(choice)
                    if 1 <= choice <= len(available_manufacturers):
                        manufacturer1 = available_manufacturers[choice - 1]
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        
        # Select second manufacturer if not provided
        if manufacturer2 is None:
            print(f"\nSelect competitor to compare with {manufacturer1}:")
            for i, mfg in enumerate(available_manufacturers, 1):
                if mfg != manufacturer1:  # Don't show first manufacturer
                    print(f"{i}. {mfg}")
                    
            while True:
                try:
                    choice = input("Enter manufacturer number: ").strip()
                    choice = int(choice)
                    if 1 <= choice <= len(available_manufacturers):
                        manufacturer2 = available_manufacturers[choice - 1]
                        if manufacturer2 != manufacturer1:
                            break
                        else:
                            print("Cannot compare manufacturer with itself. Please select a different one.")
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        
        return manufacturer1, manufacturer2
    
    def filter_discontinued_items(self, df, weeks_to_check=12, threshold_percent=10):
        """
        Filter out items that appear to be discontinued or significantly underperforming.
        
        Args:
            df: DataFrame containing weekly sales data
            weeks_to_check: Number of most recent weeks to check for activity
            threshold_percent: Items with recent sales below this percent of median will be filtered out
            
        Returns:
            DataFrame with only active items
        """
        # Get all week columns
        week_pattern = re.compile(r'^\\d{4}\\s+PD\\s+\\d{2}\\s+WK\\s+\\d+\\s+\\(\\d{2}\\)$')
        week_cols = [col for col in df.columns if week_pattern.match(col)]
        
        # Sort week columns to get most recent weeks
        week_cols.sort(reverse=True)
        recent_weeks = week_cols[:weeks_to_check]
        
        # Calculate median sales by manufacturer (excluding zeros)
        manufacturer_medians = {}
        for manufacturer in df['Manufacturer Description'].unique():
            mfg_data = df[df['Manufacturer Description'] == manufacturer]
            all_sales = []
            for week in week_cols:
                sales = pd.to_numeric(mfg_data[week].str.replace(',', ''), errors='coerce')
                all_sales.extend(sales[sales > 0])  # Only include non-zero sales
            if all_sales:
                manufacturer_medians[manufacturer] = np.median(all_sales)
            else:
                manufacturer_medians[manufacturer] = 0
        
        # For each item, check recent sales against manufacturer median
        active_items = []
        for item in df['UPC+Item Description'].unique():
            item_data = df[df['UPC+Item Description'] == item]
            manufacturer = item_data['Manufacturer Description'].iloc[0]
            mfg_median = manufacturer_medians[manufacturer]
            
            if mfg_median == 0:
                continue
                
            # Calculate average recent sales
            recent_sales = []
            for week in recent_weeks:
                sales = pd.to_numeric(item_data[week].str.replace(',', ''), errors='coerce')
                if not sales.empty:
                    recent_sales.append(sales.iloc[0])
            
            avg_recent_sales = np.mean(recent_sales) if recent_sales else 0
            
            # If average recent sales are above threshold, keep the item
            if avg_recent_sales >= (mfg_median * threshold_percent / 100):
                active_items.append(item)
        
        # Filter dataframe to keep only active items
        active_df = df[df['UPC+Item Description'].isin(active_items)]
        
        # Print removal statistics
        removed_count = len(df['UPC+Item Description'].unique()) - len(active_df['UPC+Item Description'].unique())
        print(f"Removed {removed_count} items (sales below {threshold_percent}% of manufacturer median in last {weeks_to_check} weeks)")
        print(f"Data points after filtering: {len(active_df)}")
        
        return active_df

    def prepare_weekly_data(self, manufacturer1, manufacturer2):
        """
        Prepare weekly data for analysis.
        
        Args:
            manufacturer1: First manufacturer to compare
            manufacturer2: Second manufacturer to compare
            
        Returns:
            DataFrame with weekly data for both manufacturers
        """
        # Filter for selected manufacturers
        filtered_df = self.df[
            (self.df['Manufacturer Description'].str.upper() == manufacturer1) |
            (self.df['Manufacturer Description'].str.upper() == manufacturer2)
        ]
        
        # Filter out discontinued items
        filtered_df = self.filter_discontinued_items(filtered_df)
        
        # Get all week columns
        week_pattern = re.compile(r'^\\d{4}\\s+PD\\s+\\d{2}\\s+WK\\s+\\d+\\s+\\(\\d{2}\\)$')
        week_cols = [col for col in filtered_df.columns if week_pattern.match(col)]
        
        # Create weekly data
        weekly_data = []
        
        for _, row in filtered_df.iterrows():
            item = row['UPC+Item Description']
            manufacturer = row['Manufacturer Description']
            size_oz = row['Size_OZ']
            
            for week in week_cols:
                try:
                    sales = float(str(row[week]).replace(',', ''))
                    if sales > 0:  # Only include weeks with sales
                        weekly_data.append({
                            'Item': item,
                            'Manufacturer': manufacturer,
                            'Size_OZ': size_oz,
                            'Week': week,
                            'Units': sales
                        })
                except (ValueError, TypeError):
                    continue
        
        # Convert to DataFrame
        weekly_df = pd.DataFrame(weekly_data)
        
        # Calculate average price per unit
        weekly_df['Avg_Price'] = weekly_df.groupby(['Item', 'Week'])['Units'].transform('mean')
        
        return weekly_df
    
    def filter_outliers(self, df, column):
        """Filter outliers using IQR method"""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def find_like_items(self, similarity_threshold=0.8):
        """
        Find like items by first matching at individual item level, then grouping by RBP.
        If any items between two RBP groups match with high similarity, the entire groups are considered matched.
        
        Args:
            similarity_threshold: Minimum similarity score (0-1) for item matching
        """
        print("Finding like items using string similarity and size matching...")
        
        # Debug: Print available columns in both DataFrames
        print("Original DataFrame columns:", self.df.columns.tolist())
        print("Price data columns:", self.price_data.columns.tolist())
        
        # Get items by manufacturer
        mfg1_data = self.df[self.df['Manufacturer Description'].str.upper() == self.manufacturer1]
        mfg2_data = self.df[self.df['Manufacturer Description'].str.upper() == self.manufacturer2]
        
        # Create dictionaries to store RBP group information
        mfg1_rbp_groups = {}  # RBP code -> {items: [], sizes: [], description: str}
        mfg2_rbp_groups = {}
        
        # Group items by RBP for manufacturer 1
        for _, row in mfg1_data.drop_duplicates(['Consolidated_Item', 'Size_OZ']).iterrows():
            rbp_code = row['Rules Based Pricing Code']
            if pd.notna(rbp_code):
                if rbp_code not in mfg1_rbp_groups:
                    mfg1_rbp_groups[rbp_code] = {
                        'items': [],
                        'sizes': set(),
                        'description': row['Rules Based Pricing Description']
                    }
                mfg1_rbp_groups[rbp_code]['items'].append(row['Consolidated_Item'])
                mfg1_rbp_groups[rbp_code]['sizes'].add(row['Size_OZ'])
        
        # Group items by RBP for manufacturer 2
        for _, row in mfg2_data.drop_duplicates(['Consolidated_Item', 'Size_OZ']).iterrows():
            rbp_code = row['Rules Based Pricing Code']
            if pd.notna(rbp_code):
                if rbp_code not in mfg2_rbp_groups:
                    mfg2_rbp_groups[rbp_code] = {
                        'items': [],
                        'sizes': set(),
                        'description': row['Rules Based Pricing Description']
                    }
                mfg2_rbp_groups[rbp_code]['items'].append(row['Consolidated_Item'])
                mfg2_rbp_groups[rbp_code]['sizes'].add(row['Size_OZ'])
        
        print(f"{self.manufacturer1} RBP groups: {len(mfg1_rbp_groups)}")
        print(f"{self.manufacturer2} RBP groups: {len(mfg2_rbp_groups)}")
        
        # Find matches between RBP groups based on item-level similarity
        matches = []
        total_comparisons = 0
        
        for rbp1_code, rbp1_info in mfg1_rbp_groups.items():
            for rbp2_code, rbp2_info in mfg2_rbp_groups.items():
                total_comparisons += 1
                
                # Find any matching sizes between the groups
                common_sizes = rbp1_info['sizes'].intersection(rbp2_info['sizes'])
                if not common_sizes:
                    continue
                
                # Compare all items between the groups
                best_similarity = 0
                best_item1 = None
                best_item2 = None
                
                for item1 in rbp1_info['items']:
                    for item2 in rbp2_info['items']:
                        # Calculate string similarity
                        similarity = jaro_winkler_similarity(str(item1), str(item2))
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_item1 = item1
                            best_item2 = item2
                
                # If any items match above threshold, consider the RBP groups matched
                if best_similarity >= similarity_threshold:
                    for size in common_sizes:
                        matches.append({
                            'Size_OZ': size,
                            'Mfg1_RBP_Code': rbp1_code,
                            'Mfg2_RBP_Code': rbp2_code,
                            'Mfg1_RBP_Desc': rbp1_info['description'],
                            'Mfg2_RBP_Desc': rbp2_info['description'],
                            'Best_Match_Item1': best_item1,
                            'Best_Match_Item2': best_item2,
                            'Similarity_Score': best_similarity
                        })
        
        # Sort matches by similarity score (highest first)
        matches.sort(key=lambda x: x['Similarity_Score'], reverse=True)
        
        print(f"Total RBP group comparisons: {total_comparisons}")
        print(f"Matches found with similarity ≥ {similarity_threshold}: {len(matches)}")
        
        if matches:
            print(f"Top 3 matches:")
            for i, match in enumerate(matches[:3]):
                print(f"  {i+1}. Similarity: {match['Similarity_Score']:.3f}")
                print(f"     {self.manufacturer1} RBP: {match['Mfg1_RBP_Desc']}")
                print(f"     {self.manufacturer2} RBP: {match['Mfg2_RBP_Desc']}")
                print(f"     Best matching items:")
                print(f"       {self.manufacturer1}: {match['Best_Match_Item1']}")
                print(f"       {self.manufacturer2}: {match['Best_Match_Item2']}")
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
        """
        Build and evaluate regression models for an RBP group.
        Returns a dictionary with model results if successful, None if not.
        """
        if len(item_data) < 10:
            print(f"Insufficient data points for {item_name}")
            return None
            
        X = item_data['Avg_Price'].values.reshape(-1, 1)
        y = item_data['Units'].values
        
        # Try different model types
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Log-Linear': LinearRegression(),  # Will transform data
            'Polynomial': Pipeline([
                ('poly', PolynomialFeatures(degree=2)),
                ('linear', LinearRegression())
            ]),
            'Exponential': LinearRegression(),  # Will transform data
            'Power': LinearRegression()  # Will transform data
        }
        
        best_model = None
        best_r2 = 0
        best_p_value = 1
        best_type = None
        best_X = None
        best_y = None
        
        for model_type, model in models.items():
            try:
                # Transform data based on model type
                if model_type == 'Log-Linear':
                    model_X = np.log(X)
                    model_y = y
                elif model_type == 'Exponential':
                    model_X = X
                    model_y = np.log(y + 1)  # Add 1 to handle zeros
                elif model_type == 'Power':
                    model_X = np.log(X)
                    model_y = np.log(y + 1)  # Add 1 to handle zeros
                else:
                    model_X = X
                    model_y = y
                
                # Fit model
                model.fit(model_X, model_y)
                
                # Calculate R² and p-value
                r2 = model.score(model_X, model_y)
                p_value = self.calculate_p_value(model_X, model_y, model)
                
                # Update best model if this one is better
                if r2 > best_r2 and r2 >= 0.25 and p_value <= 0.05:
                    best_model = model
                    best_r2 = r2
                    best_p_value = p_value
                    best_type = model_type
                    best_X = model_X
                    best_y = model_y
            
            except Exception as e:
                print(f"Error fitting {model_type} model: {str(e)}")
                continue
        
        if best_model is None:
            print(f"Model quality insufficient (R² < 0.25 or p > 0.05). Best R²: {best_r2:.4f}, Best p-value: {best_p_value:.4f}")
            return None
            
        # Calculate current average price and predict optimal price
        current_price = item_data['Avg_Price'].mean()
        price_range = np.linspace(max(0.29, current_price * 0.7), current_price * 1.3, 20)
        valid_prices = self.get_valid_prices(price_range.min(), price_range.max())
        
        # Transform prices based on best model type
        if best_type == 'Log-Linear':
            model_prices = np.log(valid_prices).reshape(-1, 1)
            predictions = best_model.predict(model_prices)
        elif best_type == 'Exponential':
            model_prices = np.array(valid_prices).reshape(-1, 1)
            predictions = np.exp(best_model.predict(model_prices)) - 1
        elif best_type == 'Power':
            model_prices = np.log(valid_prices).reshape(-1, 1)
            predictions = np.exp(best_model.predict(model_prices)) - 1
        elif best_type == 'Polynomial':
            model_prices = best_model.named_steps['poly'].transform(np.array(valid_prices).reshape(-1, 1))
            predictions = best_model.predict(model_prices)
        else:
            model_prices = np.array(valid_prices).reshape(-1, 1)
            predictions = best_model.predict(model_prices)
        
        # Find optimal price
        revenues = np.array(valid_prices) * predictions
        optimal_idx = np.argmax(revenues)
        optimal_price = valid_prices[optimal_idx]
        optimal_volume = predictions[optimal_idx]
        optimal_revenue = revenues[optimal_idx]
        
        # Print results
        print(f"{item_name} Model Status: Valid {best_type} model")
        print(f"{item_name} Recommendation:")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Recommended Price: ${optimal_price:.2f}")
        print(f"  Predicted Volume: {optimal_volume:.1f}")
        print(f"  Predicted Revenue: ${optimal_revenue:.2f}")
        print(f"  Model R²: {best_r2:.4f}")
        print(f"  Model P-Value: {best_p_value:.4f}")
        
        return {
            'current_price': current_price,
            'recommended_price': optimal_price,
            'predicted_volume': optimal_volume,
            'predicted_revenue': optimal_revenue,
            'r2': best_r2,
            'p_value': best_p_value,
            'model_type': best_type,
            'model': best_model,  # Include the trained model
            'baseline_volume': item_data['Units'].mean()  # Include baseline volume
        }
    
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
    
    def optimize_joint_prices(self, mfg1_data, mfg2_data, mfg1_model, mfg2_model, max_volume_loss_pct=10, price_step=0.10):
        """
        Find optimal prices that maximize combined revenue while limiting volume loss.
        
        Args:
            mfg1_data: DataFrame with first manufacturer's data
            mfg2_data: DataFrame with second manufacturer's data
            mfg1_model: Dict containing first manufacturer's model info
            mfg2_model: Dict containing second manufacturer's model info
            max_volume_loss_pct: Maximum allowed volume loss as a percentage
            price_step: Step size for price grid search
            
        Returns:
            Dictionary with optimal prices and predicted metrics
        """
        if not mfg1_model or not mfg2_model:
            return None
            
        # Get current metrics
        current_price1 = mfg1_model['current_price']
        current_price2 = mfg2_model['current_price']
        
        # Get baseline volumes from model info
        baseline_volume1 = mfg1_model['baseline_volume']
        baseline_volume2 = mfg2_model['baseline_volume']
        total_baseline_volume = baseline_volume1 + baseline_volume2
        min_allowed_volume = total_baseline_volume * (1 - max_volume_loss_pct/100)
        
        # Calculate current revenues
        current_revenue1 = baseline_volume1 * current_price1
        current_revenue2 = baseline_volume2 * current_price2
        
        # Generate price ranges (±30% from current, ending in 9)
        def get_price_range(current_price):
            min_price = max(0.29, current_price * 0.7)
            max_price = current_price * 1.3
            prices = []
            price = min_price
            while price <= max_price:
                rounded = self.round_to_valid_price(price)
                if rounded not in prices:
                    prices.append(rounded)
                price += price_step
            return sorted(prices)
        
        price_range1 = get_price_range(current_price1)
        price_range2 = get_price_range(current_price2)
        
        # Try all price combinations
        best_result = {
            'combined_revenue': current_revenue1 + current_revenue2,
            'price1': current_price1,
            'price2': current_price2,
            'volume1': baseline_volume1,
            'volume2': baseline_volume2,
            'revenue1': current_revenue1,
            'revenue2': current_revenue2,
            'total_volume': total_baseline_volume,
            'volume_change_pct': 0.0
        }
        
        for price1 in price_range1:
            for price2 in price_range2:
                # Predict volumes at these prices
                try:
                    if mfg1_model['model_type'] == 'Log-Linear':
                        volume1 = np.exp(mfg1_model['model'].predict([[np.log(price1)]]))
                    elif mfg1_model['model_type'] == 'Exponential':
                        volume1 = np.exp(mfg1_model['model'].predict([[price1]])) - 1
                    elif mfg1_model['model_type'] == 'Power':
                        volume1 = np.exp(mfg1_model['model'].predict([[np.log(price1)]])) - 1
                    else:
                        volume1 = mfg1_model['model'].predict([[price1]])
                        
                    if mfg2_model['model_type'] == 'Log-Linear':
                        volume2 = np.exp(mfg2_model['model'].predict([[np.log(price2)]]))
                    elif mfg2_model['model_type'] == 'Exponential':
                        volume2 = np.exp(mfg2_model['model'].predict([[price2]])) - 1
                    elif mfg2_model['model_type'] == 'Power':
                        volume2 = np.exp(mfg2_model['model'].predict([[np.log(price2)]])) - 1
                    else:
                        volume2 = mfg2_model['model'].predict([[price2]])
                    
                    # Ensure volumes are positive
                    volume1 = max(0, volume1)
                    volume2 = max(0, volume2)
                    
                    # Check if total volume meets minimum requirement
                    total_volume = volume1 + volume2
                    if total_volume < min_allowed_volume:
                        continue
                    
                    # Calculate revenues
                    revenue1 = volume1 * price1
                    revenue2 = volume2 * price2
                    combined_revenue = revenue1 + revenue2
                    
                    # Update best result if this combination is better
                    if combined_revenue > best_result['combined_revenue']:
                        best_result = {
                            'combined_revenue': combined_revenue,
                            'price1': price1,
                            'price2': price2,
                            'volume1': volume1,
                            'volume2': volume2,
                            'revenue1': revenue1,
                            'revenue2': revenue2,
                            'total_volume': total_volume,
                            'volume_change_pct': ((total_volume - total_baseline_volume) / total_baseline_volume) * 100
                        }
                
                except Exception as e:
                    continue
        
        return best_result
    
    def analyze_pair(self, match):
        """
        Analyze a matched pair of RBP groups.
        
        Args:
            match: Dictionary containing RBP group match information
        """
        print(f"\n=== ANALYZING PAIR: {match['Size_OZ']} OZ ===")
        print(f"{self.manufacturer1} RBP: {match['Mfg1_RBP_Desc']}")
        print(f"{self.manufacturer2} RBP: {match['Mfg2_RBP_Desc']}")
        
        # Get data for each RBP group using the best matching items
        mfg1_data = self.price_data[
            (self.price_data['Manufacturer'].str.upper() == self.manufacturer1) &
            (self.price_data['Item'] == match['Best_Match_Item1']) &
            (self.price_data['Size_OZ'] == match['Size_OZ'])
        ]
        
        mfg2_data = self.price_data[
            (self.price_data['Manufacturer'].str.upper() == self.manufacturer2) &
            (self.price_data['Item'] == match['Best_Match_Item2']) &
            (self.price_data['Size_OZ'] == match['Size_OZ'])
        ]
        
        print(f"{self.manufacturer1} data points: {len(mfg1_data)}")
        print(f"{self.manufacturer2} data points: {len(mfg2_data)}")
        print()
        
        if len(mfg1_data) < 10:
            print(f"Insufficient data points for {match['Mfg1_RBP_Desc']}")
            mfg1_model = None
        else:
            mfg1_model = self.build_item_model(mfg1_data, match['Mfg1_RBP_Desc'])
            
        if len(mfg2_data) < 10:
            print(f"Insufficient data points for {match['Mfg2_RBP_Desc']}")
            mfg2_model = None
        else:
            mfg2_model = self.build_item_model(mfg2_data, match['Mfg2_RBP_Desc'])
        
        # If we have valid models for both items, perform joint optimization
        if mfg1_model and mfg2_model:
            print("\nJoint Optimization Results:")
            joint_result = self.optimize_joint_prices(mfg1_data, mfg2_data, mfg1_model, mfg2_model)
            
            if joint_result:
                current_combined = joint_result['revenue1'] + joint_result['revenue2']
                revenue_increase = ((joint_result['combined_revenue'] - current_combined) / current_combined) * 100
                
                print(f"Current Combined Revenue: ${current_combined:.2f}")
                print(f"Optimal Combined Revenue: ${joint_result['combined_revenue']:.2f}")
                print(f"Revenue Increase: {revenue_increase:.1f}%")
                print(f"Volume Change: {joint_result['volume_change_pct']:.1f}%")
                print(f"\nOptimal Prices:")
                print(f"  {match['Mfg1_RBP_Desc']}: ${joint_result['price1']:.2f} (current: ${mfg1_model['current_price']:.2f})")
                print(f"  {match['Mfg2_RBP_Desc']}: ${joint_result['price2']:.2f} (current: ${mfg2_model['current_price']:.2f})")
                print(f"\nPredicted Volumes:")
                print(f"  {match['Mfg1_RBP_Desc']}: {joint_result['volume1']:.1f}")
                print(f"  {match['Mfg2_RBP_Desc']}: {joint_result['volume2']:.1f}")
                print(f"\nPredicted Revenues:")
                print(f"  {match['Mfg1_RBP_Desc']}: ${joint_result['revenue1']:.2f}")
                print(f"  {match['Mfg2_RBP_Desc']}: ${joint_result['revenue2']:.2f}")
        
        # Store results
        result = {
            'Size_OZ': match['Size_OZ'],
            'Mfg1_RBP': match['Mfg1_RBP_Desc'],
            'Mfg2_RBP': match['Mfg2_RBP_Desc'],
            'Best_Match_Item1': match['Best_Match_Item1'],
            'Best_Match_Item2': match['Best_Match_Item2'],
            'Similarity_Score': match['Similarity_Score']
        }
        
        # Add model results if available
        if mfg1_model:
            result.update({
                'Mfg1_Current_Price': mfg1_model['current_price'],
                'Mfg1_Individual_Recommended_Price': mfg1_model['recommended_price'],
                'Mfg1_Individual_Predicted_Volume': mfg1_model['predicted_volume'],
                'Mfg1_Individual_Predicted_Revenue': mfg1_model['predicted_revenue'],
                'Mfg1_Model_R2': mfg1_model['r2'],
                'Mfg1_Model_Type': mfg1_model['model_type']
            })
        
        if mfg2_model:
            result.update({
                'Mfg2_Current_Price': mfg2_model['current_price'],
                'Mfg2_Individual_Recommended_Price': mfg2_model['recommended_price'],
                'Mfg2_Individual_Predicted_Volume': mfg2_model['predicted_volume'],
                'Mfg2_Individual_Predicted_Revenue': mfg2_model['predicted_revenue'],
                'Mfg2_Model_R2': mfg2_model['r2'],
                'Mfg2_Model_Type': mfg2_model['model_type']
            })
            
        # Add joint optimization results if available
        if mfg1_model and mfg2_model and joint_result:
            result.update({
                'Joint_Optimal_Price1': joint_result['price1'],
                'Joint_Optimal_Price2': joint_result['price2'],
                'Joint_Predicted_Volume1': joint_result['volume1'],
                'Joint_Predicted_Volume2': joint_result['volume2'],
                'Joint_Predicted_Revenue1': joint_result['revenue1'],
                'Joint_Predicted_Revenue2': joint_result['revenue2'],
                'Joint_Combined_Revenue': joint_result['combined_revenue'],
                'Joint_Volume_Change_Pct': joint_result['volume_change_pct']
            })
        
        return result
    
    def run_analysis(self, manufacturer1='KROGER', manufacturer2=None):
        """
        Run the pricing analysis and return results.
        """
        # Load data and select manufacturers if not already done
        if not hasattr(self, 'price_data'):
            self.load_data()
            self.manufacturer1, self.manufacturer2 = self.select_manufacturers(manufacturer1, manufacturer2)
            print(f"Analyzing: {self.manufacturer1} vs {self.manufacturer2}")
        
        # Prepare weekly data
        print("Preparing weekly data...")
        self.price_data = self.prepare_weekly_data(self.manufacturer1, self.manufacturer2)
        
        # Find like items
        print("Finding like items using string similarity and size matching...")
        matches = self.find_like_items()
        
        # Print top matches
        print("Top 3 matches:")
        for i, match in enumerate(matches[:3], 1):
            print(f"  {i}. Similarity: {match['Similarity_Score']:.3f}")
            print(f"     KROGER RBP: {match['Mfg1_RBP_Desc']}")
            print(f"     {self.manufacturer2} RBP: {match['Mfg2_RBP_Desc']}")
            print(f"     Best matching items:")
            print(f"       KROGER: {match['Best_Match_Item1']}")
            print(f"       {self.manufacturer2}: {match['Best_Match_Item2']}")
            print(f"     Size: {match['Size_OZ']} OZ")
        
        # Analyze each matched pair
        print(f"Found {len(matches)} item pairs to analyze")
        recommendations = []
        
        for match in matches:
            recommendation = self.analyze_pair(match)
            if recommendation:
                recommendations.append(recommendation)
        
        # Calculate summary statistics
        valid_recommendations = 0
        total_price_gap = 0
        total_revenue_increase = 0
        total_volume_change = 0
        
        for recommendation in recommendations:
            if recommendation.get('Joint_Optimal_Price1') and recommendation.get('Joint_Optimal_Price2'):
                valid_recommendations += 1
                total_price_gap += abs(recommendation['Joint_Optimal_Price1'] - recommendation['Joint_Optimal_Price2'])
                
                # Calculate revenue increase percentage
                current_revenue = recommendation['Mfg1_Current_Price'] * recommendation['Mfg1_Individual_Predicted_Volume'] + \
                                recommendation['Mfg2_Current_Price'] * recommendation['Mfg2_Individual_Predicted_Volume']
                optimal_revenue = recommendation['Joint_Combined_Revenue']
                revenue_increase = ((optimal_revenue - current_revenue) / current_revenue) * 100
                total_revenue_increase += revenue_increase
                
                # Add volume change
                total_volume_change += recommendation['Joint_Volume_Change_Pct']
        
        if valid_recommendations > 0:
            print("\nSummary Statistics:")
            print(f"Valid Recommendations: {valid_recommendations}")
            print(f"Average Price Gap: ${total_price_gap/valid_recommendations:.2f}")
            print(f"Average Revenue Increase: {total_revenue_increase/valid_recommendations:.1f}%")
            print(f"Average Volume Change: {total_volume_change/valid_recommendations:.1f}%")
        
        return recommendations
    
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

    def select_data_file(self):
        """Let user select the data file to analyze"""
        # Get list of CSV files in current directory
        csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        
        if not csv_files:
            print("No CSV files found in current directory.")
            return None
            
        # Print available files
        print("\nAvailable data files:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
            
        while True:
            try:
                choice = input("\nSelect a file number (or press Enter for default 'SWater.csv'): ").strip()
                if not choice:  # Default to SWater.csv
                    return 'SWater.csv'
                    
                choice = int(choice)
                if 1 <= choice <= len(csv_files):
                    return csv_files[choice - 1]
                else:
                    print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = PricingAnalysis()
    
    # Run analysis (user will be prompted to select second manufacturer)
    results = analyzer.run_analysis() 