import pandas as pd
import argparse
import sys
from pathlib import Path

def load_and_inspect_data(file_path):
    """
    Loads the transaction data from the specified path and prints basic information.
    Does not modify the raw data.
    """
    path = Path(file_path)
    
    # Handle missing or incorrect file path
    if not path.exists():
        print(f"Error: The file at '{path.resolve()}' does not exist.")
        print("Please ensure you have downloaded the dataset and placed it in the correct location.")
        sys.exit(1)
        
    if not path.is_file():
        print(f"Error: The path '{path.resolve()}' is not a file.")
        sys.exit(1)

    print(f"Loading data from '{path.resolve()}'...")
    
    try:
        # Load the file based on its extension
        if path.suffix.lower() == '.csv':
            df = pd.read_csv(path, encoding='utf-8', encoding_errors='ignore')
        elif path.suffix.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(path)
        else:
            print(f"Error: Unsupported file format '{path.suffix}'. Please provide a .csv or .xlsx file.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error loading the data: {e}")
        sys.exit(1)

    # Print data information as requested
    print("\n--- Data Inspection ---")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    print("\n--- Column Names and Data Types ---")
    print(df.dtypes)
    
    print("\n--- Small Preview (First 5 rows) ---")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # Allows configuring the file path via command line
    parser = argparse.ArgumentParser(description="Load and inspect raw transaction data.")
    parser.add_argument(
        "--file_path", 
        type=str, 
        default="../data/raw/Online Retail.xlsx",
        help="Path to the raw dataset file (relative to this script or absolute)."
    )
    
    args = parser.parse_args()
    
    # Resolve the path relative to the script directory to handle being run from anywhere
    script_dir = Path(__file__).parent
    
    if Path(args.file_path).is_absolute():
        data_path = Path(args.file_path)
    else:
        data_path = (script_dir / args.file_path).resolve()
        
    # Execute the loading and inspection
    load_and_inspect_data(data_path)
