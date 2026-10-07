import pandas as pd
import argparse
import sys
from pathlib import Path

def clean_data(input_path, output_path):
    """
    Loads raw data, performs necessary validations and cleaning steps, 
    and saves the cleaned dataset.
    """
    path = Path(input_path)
    
    if not path.exists() or not path.is_file():
        print(f"Error: The file at '{path.resolve()}' does not exist.")
        sys.exit(1)
        
    print(f"Loading data from '{path.resolve()}'...")
    try:
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path, encoding='utf-8', encoding_errors='ignore')
        elif path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(path)
        else:
            print("Error: Unsupported file format.")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
        
    initial_rows = len(df)
    print(f"\n--- Initial State ---")
    print(f"Initial row count: {initial_rows}")
    print(f"Columns found: {list(df.columns)}")
    
    # 1. Report Missing Values
    print("\n--- Missing Values Report ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if not missing[missing > 0].empty else "No missing values found.")
    
    # 2. Report Duplicate Rows
    duplicate_count = df.duplicated().sum()
    print(f"\nExact duplicate rows found: {duplicate_count}")
    
    # Define standard column names for the UCI dataset to handle logic safely
    required_cols = ['InvoiceNo', 'Quantity', 'UnitPrice', 'CustomerID', 'InvoiceDate']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"\nError: Expected columns not found in dataset: {missing_cols}")
        sys.exit(1)

    # 3. Report Cancelled Invoices and Invalid Values before dropping
    # Cancelled invoices typically start with 'C' in InvoiceNo
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    cancelled_mask = df['InvoiceNo'].str.startswith('C', na=False)
    cancelled_count = cancelled_mask.sum()
    
    # Use coerce to catch any weird formatting
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
    
    invalid_qty_count = (df['Quantity'] <= 0).sum()
    invalid_price_count = (df['UnitPrice'] <= 0).sum()
    missing_customer_count = df['CustomerID'].isnull().sum()
    
    print(f"\n--- Data Quality Issues Detected ---")
    print(f"Cancelled invoices (InvoiceNo starts with 'C'): {cancelled_count}")
    print(f"Rows with non-positive Quantity (<= 0): {invalid_qty_count}")
    print(f"Rows with non-positive UnitPrice (<= 0): {invalid_price_count}")
    print(f"Rows with missing CustomerID: {missing_customer_count}")
    
    # --- Cleaning Steps ---
    print("\n--- Starting Data Cleaning ---")
    df_cleaned = df.copy()
    
    # Remove exact duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    print(f"Removed {duplicate_count} duplicate rows.")
    
    # Exclude cancelled transactions
    # Note: we recalculate the drops at each step to give an accurate report on the residual dataset
    canc_drop = df_cleaned['InvoiceNo'].str.startswith('C', na=False).sum()
    df_cleaned = df_cleaned[~df_cleaned['InvoiceNo'].str.startswith('C', na=False)]
    print(f"Excluded {canc_drop} cancelled transactions.")
    
    # Exclude non-positive quantity and price
    qty_drop = (df_cleaned['Quantity'] <= 0).sum()
    df_cleaned = df_cleaned[df_cleaned['Quantity'] > 0]
    print(f"Excluded {qty_drop} remaining rows with non-positive Quantity.")
    
    price_drop = (df_cleaned['UnitPrice'] <= 0).sum()
    df_cleaned = df_cleaned[df_cleaned['UnitPrice'] > 0]
    print(f"Excluded {price_drop} remaining rows with non-positive UnitPrice.")
    
    # Handle missing CustomerID
    cust_drop = df_cleaned['CustomerID'].isnull().sum()
    df_cleaned = df_cleaned.dropna(subset=['CustomerID'])
    print(f"Excluded {cust_drop} remaining rows with missing CustomerID.")
    print("Note: Rows without a CustomerID were dropped because they cannot be attributed to a specific customer for segmentation and repeat purchase analysis.")
    
    # --- Data Type Conversion ---
    print("\n--- Data Type Conversions ---")
    df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])
    
    # Convert CustomerID to integer first to drop '.0' if parsed as float, then to string
    df_cleaned['CustomerID'] = df_cleaned['CustomerID'].astype(int).astype(str)
    
    print("Converted 'InvoiceDate' to datetime objects.")
    print("Converted 'CustomerID' to string (treated as a categorical identifier without silent imputation).")
    
    # No columns are dropped. 
    # E.g. "Description" or "Country" might be useful for EDA later.
    print("\nNote: No columns were dropped from the dataset to preserve all available metadata.")

    final_rows = len(df_cleaned)
    print("\n--- Concise Cleaning Summary ---")
    print(f"Rows before cleaning: {initial_rows}")
    print(f"Rows after cleaning:  {final_rows}")
    print(f"Total rows removed:   {initial_rows - final_rows}")
    print(f"Percentage of data retained: {(final_rows/initial_rows)*100:.2f}%")
    
    # Save cleaned data
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving cleaned data to '{out_path.resolve()}'...")
    df_cleaned.to_csv(out_path, index=False)
    print("Success! Cleaned data saved separately.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and validate raw transaction data.")
    parser.add_argument(
        "--input_path", 
        type=str, 
        default="../data/raw/Online Retail.xlsx",
        help="Path to the raw dataset file."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="../data/processed/cleaned_transactions.csv",
        help="Path to save the cleaned data."
    )
    
    args = parser.parse_args()
    script_dir = Path(__file__).parent
    
    in_path = Path(args.input_path) if Path(args.input_path).is_absolute() else (script_dir / args.input_path).resolve()
    out_path = Path(args.output_path) if Path(args.output_path).is_absolute() else (script_dir / args.output_path).resolve()
        
    clean_data(in_path, out_path)
