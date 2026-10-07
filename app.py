import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import joblib

# --- Configuration ---
st.set_page_config(page_title="Customer Behavior Dashboard", layout="wide", page_icon="🛒")

st.title("🛒 Customer Purchase Behavior Analysis")
st.markdown("A data science portfolio project analyzing customer segments and predicting 90-day repeat purchases using the UCI Online Retail dataset.")

# --- Paths ---
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

TRANSACTIONS_PATH = DATA_DIR / "cleaned_transactions.csv"
SEGMENTS_PATH = DATA_DIR / "customer_segments.csv"
PREDICTION_PATH = MODELS_DIR / "prediction_features_target.csv"
MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# --- Data Loading ---
@st.cache_data
def load_data():
    transactions = pd.read_csv(TRANSACTIONS_PATH, parse_dates=['InvoiceDate']) if TRANSACTIONS_PATH.exists() else None
    if transactions is not None:
        transactions['LineTotal'] = transactions['Quantity'] * transactions['UnitPrice']
        
    segments = pd.read_csv(SEGMENTS_PATH) if SEGMENTS_PATH.exists() else None
    prediction_data = pd.read_csv(PREDICTION_PATH) if PREDICTION_PATH.exists() else None
    return transactions, segments, prediction_data

try:
    transactions, segments, prediction_data = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

if transactions is None:
    st.warning("⚠️ Cleaned transaction data not found. Please run Module 2 to generate the data.")
    st.stop()

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs(["Overview & Sales", "Customer Segments", "Prediction Model", "Limitations"])

# --- Tab 1: Overview & Sales ---
with tab1:
    st.header("Overall Summary")
    
    total_sales = transactions['LineTotal'].sum()
    total_customers = transactions['CustomerID'].nunique()
    total_invoices = transactions['InvoiceNo'].nunique()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales Revenue", f"£{total_sales:,.2f}")
    col2.metric("Total Unique Customers", f"{total_customers:,}")
    col3.metric("Total Invoices", f"{total_invoices:,}")
    
    st.header("Sales Trend Over Time")
    # Group by month for charting
    transactions['MonthYear'] = transactions['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_sales = transactions.groupby('MonthYear')['LineTotal'].sum().reset_index()
    
    fig = px.bar(monthly_sales, x='MonthYear', y='LineTotal', title='Monthly Sales Revenue (£)', 
                 labels={'MonthYear': 'Month', 'LineTotal': 'Revenue (£)'})
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Customer Segments ---
with tab2:
    st.header("Customer Segmentation Profiles")
    
    if segments is None:
        st.info("ℹ️ Customer segments data not found. Run Module 5 to unlock this tab.")
    else:
        cluster_counts = segments['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Customer Count']
        cluster_counts['Cluster'] = cluster_counts['Cluster'].astype(str)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Segment Sizes")
            fig_pie = px.pie(cluster_counts, names='Cluster', values='Customer Count', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col2:
            st.subheader("Explore Segment Behaviors")
            selected_cluster = st.selectbox("Select a Cluster to view its median profile:", options=sorted(segments['Cluster'].unique()))
            
            cluster_data = segments[segments['Cluster'] == selected_cluster]
            med_recency = cluster_data['Recency'].median()
            med_frequency = cluster_data['Frequency'].median()
            med_monetary = cluster_data['Monetary'].median()
            
            st.markdown(f"**Cluster {selected_cluster} Profile (Medians):**")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Recency (Days)", f"{med_recency:.1f}")
            metric_cols[1].metric("Frequency (Orders)", f"{med_frequency:.1f}")
            metric_cols[2].metric("Monetary (Spend)", f"£{med_monetary:,.2f}")
            
        st.subheader("Recency vs. Monetary Distribution")
        segments_plot = segments.copy()
        segments_plot['Cluster'] = segments_plot['Cluster'].astype(str)
        fig_scatter = px.scatter(segments_plot, x='Recency', y='Monetary', color='Cluster',
                                 log_y=True, opacity=0.7,
                                 labels={'Monetary': 'Monetary Value (£) - Log Scale'})
        st.plotly_chart(fig_scatter, use_container_width=True)

# --- Tab 3: Prediction Model ---
with tab3:
    st.header("90-Day Repeat Purchase Prediction")
    
    if prediction_data is None or not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.info("ℹ️ Prediction model not found. Run Module 6 to unlock this tab.")
    else:
        st.markdown("We trained a **Logistic Regression** model to predict whether a customer will make a purchase in the next 90 days, strictly using data from *before* a chronological cutoff date to prevent data leakage.")
        
        try:
            from sklearn.model_selection import train_test_split
            
            X = prediction_data[['Recency', 'Frequency', 'Monetary', 'Tenure']]
            y = prediction_data['Target_90_Days']
            
            # Recreate the exact test set from Module 6
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            X_test_scaled = scaler.transform(np.log1p(X_test))
            y_pred = model.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            st.subheader("Evaluation on Unseen Test Set")
            m_cols = st.columns(4)
            m_cols[0].metric("Accuracy", f"{acc:.3f}")
            m_cols[1].metric("Precision", f"{prec:.3f}")
            m_cols[2].metric("Recall", f"{rec:.3f}")
            m_cols[3].metric("F1 Score", f"{f1:.3f}")
            
            col_cm, col_txt = st.columns([1, 2])
            with col_cm:
                st.write("**Confusion Matrix**")
                cm_df = pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Pred No', 'Pred Yes'])
                st.dataframe(cm_df)
                
            with col_txt:
                st.info("""
                **Why not just look at Accuracy?** 
                In imbalanced datasets where the majority class is 'No', a dummy model guessing 'No' for everyone would still have high accuracy. 
                **Precision** tells us how often the model is correct when it says 'Yes'. **Recall** tells us how many of the true 'Yes' customers it successfully found.
                """)
            
        except Exception as e:
            st.error(f"Error evaluating model: {e}")

# --- Tab 4: Limitations ---
with tab4:
    st.header("Project Limitations")
    st.markdown("""
    When reviewing this analysis, please keep the following limitations in mind:
    
    1. **Transactional Data Only**: This dataset strictly contains invoice transaction lines. It **does not** contain demographic data (age, gender, location), website interactions (clicks, time on page), or app usage. All segments and predictions are based purely on past purchasing habits.
    2. **Short History**: The dataset spans only ~1 year. Customers with a 'Tenure' of 0 days (buying all items on a single day) might be true one-time buyers, or they might have years of history prior to our data collection window that we cannot see.
    3. **Prediction vs. Causation**: Our Logistic Regression model predicts who is *highly likely* to repurchase. However, prediction is not causation. Sending a 20% discount coupon to someone who is already 95% likely to buy on their own might simply waste profit margin. Predictive modeling identifies patterns, but it does not prove that a specific intervention will successfully change behavior.
    """)
