import pandas as pd
import argparse
from pathlib import Path

def engineer_features(input_path, output_path):
    """
    Engineers customer-level features (RFM, AOV, Tenure) from cleaned transactions.
    Creates a single feature table with one row per customer.
    """
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"Error: Could not find '{input_file.resolve()}'. Run Module 2 first.")
        return
        
    print(f"Loading cleaned data from {input_file.resolve()}...")
    df = pd.read_csv(input_file, parse_dates=['InvoiceDate'])
    
    # Calculate Revenue per line item
    df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    
    # Determine the Analysis Date
    # We set the analysis date to exactly 1 day after the most recent transaction in the dataset.
    # This prevents any customer from having a Recency of 0 days (which could be misleading)
    # and provides a fixed point in time to measure against.
    max_date = df['InvoiceDate'].max()
    analysis_date = max_date + pd.Timedelta(days=1)
    
    print(f"\n--- Feature Engineering Details ---")
    print(f"Dataset Date Range: {df['InvoiceDate'].min().date()} to {max_date.date()}")
    print(f"Analysis Date set to: {analysis_date.date()} (1 day after the latest purchase)")
    print("Calculating Recency, Frequency, Monetary, AOV, and Tenure features...")
    
    # Aggregate to customer level
    customer_features = df.groupby('CustomerID').agg(
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('LineTotal', 'sum'),
        FirstPurchase=('InvoiceDate', 'min'),
        LastPurchase=('InvoiceDate', 'max')
    ).reset_index()
    
    # 1. Recency: Days since most recent purchase (Analysis Date - Last Purchase Date)
    customer_features['Recency'] = (analysis_date - customer_features['LastPurchase']).dt.days
    
    # 2. Average Order Value: Total Spend / Number of Invoices
    customer_features['AvgOrderValue'] = customer_features['Monetary'] / customer_features['Frequency']
    
    # 3. Tenure: Days between the first and last purchase
    customer_features['Tenure'] = (customer_features['LastPurchase'] - customer_features['FirstPurchase']).dt.days
    
    # Reorder columns and drop the raw date columns to keep it clean
    feature_cols = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 'Tenure']
    customer_features = customer_features[feature_cols]
    
    print("\n--- Validation Checks ---")
    # Check for missing values
    missing = customer_features.isnull().sum()
    if missing.sum() == 0:
        print("Missing values check: PASSED (0 missing values)")
    else:
        print(f"Missing values check: FAILED\n{missing[missing > 0]}")
        
    # Check for impossible values
    impossible_recency = (customer_features['Recency'] <= 0).sum() # Must be at least 1 since analysis date is max_date + 1 day
    impossible_frequency = (customer_features['Frequency'] <= 0).sum()
    impossible_monetary = (customer_features['Monetary'] <= 0).sum() # Because we cleaned out non-positive quantities and prices
    impossible_tenure = (customer_features['Tenure'] < 0).sum()
    
    print(f"Rows with impossible Recency (<= 0): {impossible_recency}")
    print(f"Rows with impossible Frequency (<= 0): {impossible_frequency}")
    print(f"Rows with impossible Monetary value (<= 0): {impossible_monetary}")
    print(f"Rows with impossible Tenure (< 0): {impossible_tenure}")
    
    # Check for duplicate rows
    duplicate_customers = customer_features.duplicated(subset=['CustomerID']).sum()
    print(f"Duplicate Customer rows check: {duplicate_customers} duplicates found.")
    
    # Calculate how many have a Tenure of 0
    zero_tenure = (customer_features['Tenure'] == 0).sum()
    pct_zero_tenure = (zero_tenure / len(customer_features)) * 100
    
    print("\n--- Limitations & Interpretation ---")
    print(f"1. Short Histories: {zero_tenure} customers ({pct_zero_tenure:.1f}%) have a Tenure of 0 days.")
    print("   This means all their purchases occurred on a single day. The dataset only covers 1 year,")
    print("   so we lack historical depth. We cannot tell if they are truly 'one-time buyers' or")
    print("   if they purchased before/after our 1-year data collection window.")
    print("2. Prediction Warning: Features like Tenure and Frequency must be handled carefully")
    print("   if we train a model later to prevent data leakage from the future prediction window.")
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving customer feature table to '{out_path.resolve()}'...")
    customer_features.to_csv(out_path, index=False)
    print("Success! Features saved.")
    
    print("\n--- Feature Table Preview ---")
    print(customer_features.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Customer-Level Features")
    parser.add_argument("--input_path", type=str, default="../data/processed/cleaned_transactions.csv")
    parser.add_argument("--output_path", type=str, default="../data/processed/customer_features.csv")
    
    args = parser.parse_args()
    script_dir = Path(__file__).parent
    
    in_path = Path(args.input_path) if Path(args.input_path).is_absolute() else (script_dir / args.input_path).resolve()
    out_path = Path(args.output_path) if Path(args.output_path).is_absolute() else (script_dir / args.output_path).resolve()
    
    engineer_features(in_path, out_path)
