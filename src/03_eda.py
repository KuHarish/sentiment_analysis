import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path

def run_eda(input_path, output_dir):
    """
    Performs Exploratory Data Analysis on the cleaned dataset.
    Generates charts and a summary text file with key findings.
    """
    # Setup paths
    input_file = Path(input_path)
    out_dir = Path(output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "eda_summary.txt"
    
    print(f"Loading cleaned data from {input_file}...")
    df = pd.read_csv(input_file, parse_dates=['InvoiceDate'])
    
    # Calculate LineTotal (Revenue per item line)
    df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    
    # 1. Date Coverage & Overall Metrics
    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    
    n_customers = df['CustomerID'].nunique()
    n_invoices = df['InvoiceNo'].nunique()
    total_sales = df['LineTotal'].sum()
    
    # 2. Aggregations
    # Invoice-level aggregation
    invoice_df = df.groupby(['InvoiceNo', 'CustomerID']).agg(
        OrderValue=('LineTotal', 'sum'),
        TotalItems=('Quantity', 'sum')
    ).reset_index()
    
    # Customer-level aggregation
    customer_df = invoice_df.groupby('CustomerID').agg(
        OrderCount=('InvoiceNo', 'nunique'),
        TotalSpend=('OrderValue', 'sum')
    ).reset_index()
    customer_df['AvgOrderValue'] = customer_df['TotalSpend'] / customer_df['OrderCount']
    
    # Monthly Aggregations
    df['MonthYear'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_sales = df.groupby('MonthYear')['LineTotal'].sum().reset_index()
    monthly_invoices = df.groupby('MonthYear')['InvoiceNo'].nunique().reset_index()
    
    # 3. Chart Generation
    print("Generating charts...")
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Sales over time
    plt.figure(figsize=(12, 6))
    sns.barplot(x='MonthYear', y='LineTotal', data=monthly_sales, color='skyblue')
    plt.title('Monthly Sales Revenue', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Revenue (£)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / 'monthly_sales.png')
    plt.close()
    
    # Chart 2: Transaction volume over time
    plt.figure(figsize=(12, 6))
    sns.barplot(x='MonthYear', y='InvoiceNo', data=monthly_invoices, color='lightgreen')
    plt.title('Monthly Transaction Volume (Number of Invoices)', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Invoices', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_dir / 'monthly_transactions.png')
    plt.close()
    
    # Chart 3: Distribution of Customer Spending (log scale)
    plt.figure(figsize=(10, 6))
    # Log10 to handle extreme skewness
    sns.histplot(np.log10(customer_df['TotalSpend'] + 1), bins=50, color='coral')
    plt.title('Distribution of Customer Total Spending (Log10 Scale)', fontsize=16)
    plt.xlabel('Log10(Total Spend in £)', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    
    median_spend = customer_df['TotalSpend'].median()
    plt.axvline(np.log10(median_spend + 1), color='red', linestyle='dashed', linewidth=2, 
                label=f"Median Spend: £{median_spend:,.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'customer_spending_dist.png')
    plt.close()
    
    # Chart 4: Order Frequency Distribution
    plt.figure(figsize=(10, 6))
    # Clip at 20 to prevent a massive long tail from squeezing the chart
    sns.histplot(customer_df['OrderCount'].clip(upper=20), bins=20, discrete=True, color='purple')
    plt.title('Distribution of Order Frequency per Customer (Capped at 20)', fontsize=16)
    plt.xlabel('Number of Orders', fontsize=12)
    plt.ylabel('Number of Customers', fontsize=12)
    
    median_orders = customer_df['OrderCount'].median()
    plt.axvline(median_orders, color='red', linestyle='dashed', linewidth=2, 
                label=f"Median Orders: {median_orders}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'order_frequency_dist.png')
    plt.close()
    
    # 4. One-time vs Repeat Customers Analysis
    customer_df['CustomerType'] = np.where(customer_df['OrderCount'] == 1, 'One-Time', 'Repeat')
    type_counts = customer_df['CustomerType'].value_counts()
    spend_by_type = customer_df.groupby('CustomerType')['TotalSpend'].agg(['count', 'mean', 'median'])
    
    # 5. Statistical Test
    onetime_aov = customer_df[customer_df['CustomerType'] == 'One-Time']['AvgOrderValue']
    repeat_aov = customer_df[customer_df['CustomerType'] == 'Repeat']['AvgOrderValue']
    
    # Mann-Whitney U test because spending/AOV distributions are heavily skewed right
    stat, p_value = stats.mannwhitneyu(onetime_aov, repeat_aov, alternative='two-sided')
    
    # 6. Generate Summary Text
    summary = f"""=========================================================
EXPLORATORY DATA ANALYSIS SUMMARY
=========================================================

--- Overall Metrics ---
Date Coverage: {min_date.date()} to {max_date.date()}
Total Customers: {n_customers:,}
Total Invoices: {n_invoices:,}
Total Sales Revenue: £{total_sales:,.2f}

--- Customer Behavior Metrics ---
Average Spend per Customer: £{customer_df['TotalSpend'].mean():,.2f}
Median Spend per Customer: £{customer_df['TotalSpend'].median():,.2f}
Average Orders per Customer: {customer_df['OrderCount'].mean():.2f}
Median Orders per Customer: {customer_df['OrderCount'].median():.2f}
Overall Average Order Value (AOV): £{invoice_df['OrderValue'].mean():,.2f}
Overall Median Order Value: £{invoice_df['OrderValue'].median():,.2f}

--- One-Time vs Repeat Customers ---
One-Time Customers: {type_counts.get('One-Time', 0):,} ({type_counts.get('One-Time', 0)/n_customers*100:.1f}%)
Repeat Customers:   {type_counts.get('Repeat', 0):,} ({type_counts.get('Repeat', 0)/n_customers*100:.1f}%)

Spending Breakdown (Total Spend):
{spend_by_type.to_string()}

--- Statistical Test: Mann-Whitney U Test ---
Comparison: Average Order Value (AOV) between One-Time and Repeat customers.
Reasoning: Financial data (like Order Values) is highly right-skewed, meaning the mean is pulled upwards by extreme outliers. The Mann-Whitney U test is a non-parametric test that compares the medians/distributions without assuming a normal distribution, making it robust to these outliers.
Null Hypothesis: There is no difference in the underlying distribution of AOV between one-time and repeat customers.

Result: U-statistic = {stat:,.2f}, p-value = {p_value:.4e}
"""
    
    if p_value < 0.05:
        summary += "\nConclusion: The p-value is strictly less than 0.05, so we reject the null hypothesis. There is a statistically significant difference in the average order value between one-time and repeat customers."
    else:
        summary += "\nConclusion: The p-value is greater than 0.05, so we fail to reject the null hypothesis. There is no statistically significant difference in the average order value between the two groups."
        
    summary += "\n\nLimitations: This test simply observes a difference in distributions; it does NOT prove causation. It does not mean that forcing someone to buy a second time will inherently change how much they spend per order. Additionally, the definition of a 'one-time' customer is constrained by our 1-year data window; they may have purchased before or after this period."
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
        
    print(summary)
    print(f"\nAll charts saved to: {fig_dir.resolve()}")
    print(f"Summary report saved to: {summary_file.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned transaction data.")
    parser.add_argument("--input_path", type=str, default="../data/processed/cleaned_transactions.csv")
    parser.add_argument("--output_dir", type=str, default="../reports")
    
    args = parser.parse_args()
    script_dir = Path(__file__).parent
    
    in_path = Path(args.input_path) if Path(args.input_path).is_absolute() else (script_dir / args.input_path).resolve()
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (script_dir / args.output_dir).resolve()
    
    run_eda(in_path, out_dir)
