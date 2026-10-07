import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import joblib
import warnings

def run_prediction(input_path, output_dir):
    """
    Trains a Logistic Regression baseline to predict 90-day repeat purchases.
    Strictly uses time-based splitting to prevent data leakage.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Error: {input_file.resolve()} not found. Please run Module 2 to generate cleaned transactions.")
        return
        
    print(f"Loading cleaned transaction data from {input_file.resolve()}...")
    df = pd.read_csv(input_file, parse_dates=['InvoiceDate'])
    df['LineTotal'] = df['Quantity'] * df['UnitPrice']
    
    # --- 1. Temporal Setup & Leakage Prevention ---
    min_date = df['InvoiceDate'].min()
    max_date = df['InvoiceDate'].max()
    
    # We want a 90-day prediction window
    target_window_days = 90
    cutoff_date = max_date - pd.Timedelta(days=target_window_days)
    
    print("\n--- Temporal Setup (Preventing Data Leakage) ---")
    print("Rule: Features are built using ONLY transactions ON OR BEFORE the cutoff date.")
    print("Rule: Target labels are built using ONLY transactions AFTER the cutoff date.")
    print(f"Available Date Range: {min_date.date()} to {max_date.date()}")
    print(f"Chosen Cutoff Date: {cutoff_date.date()}")
    print(f"Historical Feature Window: {min_date.date()} to {cutoff_date.date()}")
    print(f"90-Day Target Window: {(cutoff_date + pd.Timedelta(days=1)).date()} to {max_date.date()}")
    
    historical_days = (cutoff_date - min_date).days
    if historical_days < 90:
        print(f"\nLIMITATION: There are only {historical_days} days of historical data before the cutoff.")
        print("We cannot build reliable features with such a short history. Training aborted.")
        return
        
    print(f"Historical data spans {historical_days} days. The data supports modeling. Proceeding...")
    
    # --- 2. Split Transactions Chronologically ---
    feature_data = df[df['InvoiceDate'] <= cutoff_date]
    target_data = df[df['InvoiceDate'] > cutoff_date]
    
    # We only care about predicting for customers who were active BEFORE the cutoff date.
    valid_customers = feature_data['CustomerID'].unique()
    
    # --- 3. Build Predictor Features ---
    print("\nBuilding predictor features (Recency, Frequency, Monetary, Tenure)...")
    customer_features = feature_data.groupby('CustomerID').agg(
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('LineTotal', 'sum'),
        FirstPurchase=('InvoiceDate', 'min'),
        LastPurchase=('InvoiceDate', 'max')
    ).reset_index()
    
    # Recency calculated against the cutoff_date, NOT the dataset max date
    customer_features['Recency'] = (cutoff_date - customer_features['LastPurchase']).dt.days
    customer_features['Tenure'] = (customer_features['LastPurchase'] - customer_features['FirstPurchase']).dt.days
    
    # Note: Cluster labels are explicitly NOT used as features here because they were built 
    # using the full dataset (Module 5), which contains post-cutoff info (Data Leakage).
    
    # --- 4. Build Target Labels ---
    print("Building target labels (1 = Purchased in next 90 days, 0 = Did not purchase)...")
    purchased_in_target = target_data['CustomerID'].unique()
    customer_features['Target_90_Days'] = customer_features['CustomerID'].isin(purchased_in_target).astype(int)
    
    # --- 5. Report Valid Customers and Class Balance ---
    n_customers = len(customer_features)
    positives = customer_features['Target_90_Days'].sum()
    negatives = n_customers - positives
    pct_positive = (positives / n_customers) * 100
    
    print("\n--- Target Label Distribution ---")
    print(f"Number of customers with usable features: {n_customers:,}")
    print(f"Positive Labels (Purchased again): {positives:,} ({pct_positive:.1f}%)")
    print(f"Negative Labels (Did not purchase): {negatives:,} ({100 - pct_positive:.1f}%)")
    
    # --- 6. Train-Test Split (Customer Level) ---
    features = ['Recency', 'Frequency', 'Monetary', 'Tenure']
    X = customer_features[features]
    y = customer_features['Target_90_Days']
    
    # We split 80% train, 20% test, stratifying to maintain the target balance.
    # Because all features share the exact same temporal cutoff window, randomly splitting customers here is correct.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 7. Preprocessing and Modeling ---
    print("\n--- Modeling: Logistic Regression Baseline ---")
    print("Preprocessing: Applying log1p to heavily skewed features, then StandardScaling.")
    
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)
    
    # Train Logistic Regression, using class_weight='balanced' to handle the class imbalance safely
    model = LogisticRegression(random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # --- 8. Evaluation ---
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- Evaluation on Unseen Test Set ---")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}  (When the model predicts a repeat purchase, it is correct {precision*100:.1f}% of the time)")
    print(f"Recall:    {recall:.3f}  (The model successfully identified {recall*100:.1f}% of all actual repeat purchasers)")
    print(f"F1 Score:  {f1:.3f}  (The harmonic mean of Precision and Recall)")
    
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {cm[0, 0]} | False Positives: {cm[0, 1]}")
    print(f"False Negatives: {cm[1, 0]} | True Positives:  {cm[1, 1]}")
    
    print("\n--- Why Accuracy Alone is Misleading ---")
    print(f"If we built a 'dumb' model that blindly guessed NO ONE would buy again, its accuracy would be {100 - pct_positive:.1f}% (the majority class size). High accuracy in imbalanced datasets often hides the fact that the model completely fails to identify the minority class. Precision, Recall, and F1 are much better indicators of true model performance.")
    
    print("\n--- Disclaimer on Intervention Causation ---")
    print("This predictive model identifies customers who have a high probability of repurchasing based on their past habits. However, this is purely predictive, NOT causal. If the model says a customer is 90% likely to return, sending them a 20% discount coupon might simply waste your profit margin, because they were likely going to buy anyway. Predictive performance does NOT prove that an intervention will actually change a customer's behavior.")
    
    # --- 9. Save Models ---
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(scaler, out_dir / 'scaler.pkl')
    joblib.dump(model, out_dir / 'logistic_regression_model.pkl')
    customer_features.to_csv(out_dir / 'prediction_features_target.csv', index=False)
    
    print(f"\nSuccess! Scaler, Model, and modeling dataset saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict 90-Day Repeat Purchases")
    parser.add_argument("--input_path", type=str, default="../data/processed/cleaned_transactions.csv")
    parser.add_argument("--output_dir", type=str, default="../models")
    
    args = parser.parse_args()
    script_dir = Path(__file__).parent
    
    in_path = Path(args.input_path) if Path(args.input_path).is_absolute() else (script_dir / args.input_path).resolve()
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else (script_dir / args.output_dir).resolve()
    
    run_prediction(in_path, out_dir)
