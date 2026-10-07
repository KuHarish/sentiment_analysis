import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import argparse
from pathlib import Path
import warnings

def run_segmentation(input_path, output_path, reports_dir):
    """
    Performs Customer Segmentation using K-Means Clustering on RFM features.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: {input_file.resolve()} not found. Run Module 4 first.")
        return
        
    print(f"Loading features from {input_file.resolve()}...")
    df = pd.read_csv(input_file)
    
    # --- Feature Selection ---
    # We select Recency, Frequency, and Monetary (RFM).
    # Exclusions explained:
    # 1. CustomerID is an identifier, not a behavioral feature.
    # 2. AvgOrderValue is redundant (collinear) since it is just Monetary / Frequency.
    # 3. Tenure contains many 0s and is heavily tied to Frequency for this specific 1-year dataset. 
    #    Keeping it simple with standard RFM yields the most interpretable clusters.
    features = ['Recency', 'Frequency', 'Monetary']
    X = df[features].copy()
    
    print("\n--- Preprocessing ---")
    print("1. Handling Skewness: RFM values are heavily skewed. We apply a log1p transformation (log(1+x)) to pull extreme outliers closer to the center, creating a more normal distribution.")
    X_log = np.log1p(X)
    
    print("2. Scaling: K-means calculates Euclidean distance. We must scale the features (StandardScaler) so that Monetary (in thousands) doesn't completely overpower Recency (in hundreds) and Frequency (in tens).")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    
    print("\n--- Evaluating Cluster Counts (K=2 to 6) ---")
    # Suppress UserWarning about memory leaks in KMeans on Windows
    warnings.filterwarnings("ignore", category=UserWarning)
    
    best_k = 2
    best_score = -1
    scores = {}
    
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
        # Ranges from -1 to 1. Higher is better.
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"K={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            
    print(f"\n=> Choosing K={best_k} as it has the highest Silhouette Score ({best_score:.4f}).")
    print("Disclaimer: Clustering is not objectively 'correct'. K-means simply finds the most mathematically compact groups based on Euclidean distance. While K=2 or K=3 might mathematically score best, a business might prefer K=4 or K=5 for more granular marketing. We will use the mathematically optimal one for this portfolio.")
    
    # --- Fit the final model ---
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    print(f"\n--- Segment Profiles (K={best_k}) ---")
    # Summarize using medians. We do not use means because un-logged monetary/frequency values are still skewed.
    segment_summary = df.groupby('Cluster').agg(
        CustomerCount=('CustomerID', 'count'),
        MedianRecency=('Recency', 'median'),
        MedianFrequency=('Frequency', 'median'),
        MedianMonetary=('Monetary', 'median')
    ).reset_index()
    
    segment_summary['% of Total'] = (segment_summary['CustomerCount'] / len(df) * 100).round(1)
    print(segment_summary.to_string(index=False))
    
    print("\nNote: We intentionally do not give these clusters business names (like 'VIPs' or 'Churned') until you review their actual median behaviors above.")
    
    # --- Save Outputs ---
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nCustomer-to-cluster assignments saved to: {out_path.resolve()}")
    
    # Charts
    rep_dir = Path(reports_dir)
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Segment Sizes
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x='Cluster', y='CustomerCount', data=segment_summary, palette='Set2')
    plt.title(f'Customer Segment Sizes (K={best_k})', fontsize=14)
    plt.xlabel('Cluster (0-Indexed)')
    plt.ylabel('Number of Customers')
    for i, row in segment_summary.iterrows():
        plt.text(row.name, row.CustomerCount, f"{row['% of Total']}%", color='black', ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(fig_dir / 'segmentation_sizes.png')
    plt.close()
    
    # Chart 2: Recency vs Monetary Profile Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x='Recency', 
        y='Monetary', 
        hue='Cluster', 
        data=df, 
        palette='Set2', 
        alpha=0.6,
        edgecolor=None
    )
    plt.yscale('log') # Log scale for monetary on Y axis to see the cluster separation clearly
    plt.title(f'Segment Profiles (K={best_k}): Recency vs. Monetary (Log Scale)', fontsize=14)
    plt.xlabel('Recency (Days since last purchase)')
    plt.ylabel('Monetary Value (£) - Log Scale')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(fig_dir / 'segmentation_scatter.png')
    plt.close()
    
    print(f"Segment charts saved to: {fig_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment Customers using K-Means")
    parser.add_argument("--input_path", type=str, default="../data/processed/customer_features.csv")
    parser.add_argument("--output_path", type=str, default="../data/processed/customer_segments.csv")
    parser.add_argument("--reports_dir", type=str, default="../reports")
    
    args = parser.parse_args()
    script_dir = Path(__file__).parent
    
    in_path = Path(args.input_path) if Path(args.input_path).is_absolute() else (script_dir / args.input_path).resolve()
    out_path = Path(args.output_path) if Path(args.output_path).is_absolute() else (script_dir / args.output_path).resolve()
    rep_dir = Path(args.reports_dir) if Path(args.reports_dir).is_absolute() else (script_dir / args.reports_dir).resolve()
    
    run_segmentation(in_path, out_path, rep_dir)
