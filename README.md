# Customer Purchase Behavior Analysis, Segmentation, and Repeat-Purchase Prediction

## Project Scope and Objective
This project analyzes customer purchasing behavior to segment customers and predict the likelihood of repeat purchases within a 90-day window. It is built as a data science portfolio project demonstrating data cleaning, exploratory data analysis (EDA), feature engineering, K-Means clustering, predictive modeling, and building an interactive Streamlit dashboard.

## Dataset Source and Limitations
- **Source:** [UCI Machine Learning Repository - Online Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Timeframe:** 2010-12-01 to 2011-12-09 (~1 year).
- **Limitations:** The dataset strictly contains transactional data (invoices, stock codes, quantities, prices). It does **not** contain demographic information, website visits, or app interaction data. Consequently, all segments and predictions are based entirely on past purchasing habits.

## Setup and Run Instructions
To run this project locally, follow these exact steps:

1. **Install Dependencies:**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Download the Data:**
   Download the `Online Retail.xlsx` file from the UCI repository and place it in the `data/raw/` directory.
3. **Execute the Pipeline:**
   Run the modules sequentially to generate all datasets and models:
   ```bash
   python src/01_data_loading.py
   python src/02_data_cleaning.py
   python src/03_eda.py
   python src/04_feature_engineering.py
   python src/05_segmentation.py
   python src/06_prediction.py
   ```
4. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 1. Cleaning Decisions
- Exact duplicate rows were removed.
- Cancelled transactions (InvoiceNo starting with 'C') and rows with non-positive quantities or prices were dropped.
- Rows missing a `CustomerID` (approx. 135,000 rows) were dropped. No imputation was performed, as predicting customer-level behavior requires definitive customer IDs.

## 2. EDA Findings
- **Revenue & Volume:** The dataset contains 4,338 unique customers generating ~£8.89 million in total sales across 18,532 invoices.
- **Skewed Behavior:** Customer spending is heavily right-skewed. The *average* spend is £2,048, but the *median* spend is much lower at £668.
- **Repeat vs. One-Time:** 65.6% of customers are repeat buyers, while 34.4% purchased only once in the 1-year window. A Mann-Whitney U test confirmed a statistically significant difference in Average Order Value between one-time and repeat buyers.

## 3. Feature Definitions (RFM)
Customer-level features were engineered for clustering and prediction:
- **Recency:** Days since the customer's most recent purchase, measured against a fixed analysis date.
- **Frequency:** Total number of distinct invoices.
- **Monetary:** Total spending by the customer.
- **Tenure:** Days between the customer's first and last purchase.

## 4. Segmentation Method & Profiles
- **Method:** Customers were grouped using **K-Means Clustering**. Because RFM values are heavily right-skewed, a `log1p` transformation and `StandardScaler` were applied prior to clustering. The algorithm evaluated cluster counts (K=2 to 6) and optimized using the Silhouette Score.
- **Segment Profiles:** (Exact median metrics available via the dashboard). The algorithm reliably identifies mathematically distinct groups based on their recency and spend, effectively separating high-value loyal customers from low-value churned ones.

## 5. Prediction Setup and Metrics
- **Objective:** Predict if a customer will make a purchase in the final 90 days of the dataset.
- **Leakage Prevention:** Features (X) were built strictly using data *on or before* the 90-day cutoff. The target (Y) was defined strictly using data *after* the cutoff. Cluster labels were explicitly excluded as features because they contained future data.
- **Model:** A Logistic Regression baseline with balanced class weights.
- **Metrics:** Precision, Recall, and F1 Score were used to evaluate the model on an unseen 20% test set. Accuracy was explicitly deemed misleading due to the natural class imbalance (the majority of customers do not buy in a specific 90-day window). Exact test-set metrics are displayed in the dashboard.

## 6. Practical Recommendations
Based on the data, here are three strategies to test:
1. **Target the "Golden Middle":** Since 65.6% of customers repeat, but the median order is only £302, test a "Spend £400, get 10% off" campaign specifically targeting recent, mid-tier customers to pull their basket size up.
2. **Re-engage One-Time Buyers Safely:** Over 34% of customers buy once and stop. Test a low-margin "Welcome Back" discount on this group at the 60-day mark. Ensure a control group is used to measure actual lift.
3. **Protect the High-Value Cluster:** K-Means identifies a cluster of highly frequent, high-monetary customers. Instead of discounts (which waste margin on guaranteed sales), test VIP perks like early access to new products or free expedited shipping to maintain loyalty.

## 7. Limitations and Next Steps
- **Short History Limitation:** The 1-year window means customers with a "0-day tenure" might have years of invisible purchase history before the dataset began.
- **Prediction != Causation:** The logistic regression model identifies customers *likely* to buy again. Intervening (e.g., sending a coupon) to these specific customers does not guarantee their behavior will change. It may simply cannibalize organic revenue.
- **Next Steps:** Implement an A/B testing framework to measure the true causal impact of marketing campaigns on the predicted customer segments.
