"""
Bank Churn Prediction Streamlit App
Predicts customer churn likelihood using a trained XGBoost model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — dark-mode compatible
st.markdown("""
    <style>
    /* ── Color tokens (dark-mode aware) ───────────────────── */
    :root {
        --primary:   #60a5fa;   /* bright blue — readable on dark */
        --secondary: #34d399;   /* emerald */
        --accent:    #f87171;   /* soft red */
        --success:   #4ade80;
        --warning:   #fb923c;
        --card-bg:   rgba(255,255,255,0.06);
        --card-border: rgba(255,255,255,0.12);
        --text-muted: rgba(255,255,255,0.55);
    }

    /* ── Generic metric card (used in misc sections) ──────── */
    .metric-card {
        background: var(--card-bg);
        backdrop-filter: blur(6px);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid var(--primary);
        box-shadow: 0 1px 6px rgba(0,0,0,0.35);
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--primary);
        margin: 10px 0;
    }

    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--text-muted);
        font-weight: 600;
    }

    /* ── Prediction result cards ──────────────────────────── */
    .churn-risk-high {
        background: rgba(220, 38, 38, 0.18);
        border: 2px solid #ef4444;
        color: #fca5a5;
    }

    .churn-risk-low {
        background: rgba(22, 163, 74, 0.18);
        border: 2px solid #22c55e;
        color: #86efac;
    }

    .churn-risk-medium {
        background: rgba(234, 88, 12, 0.18);
        border: 2px solid #f97316;
        color: #fdba74;
    }

    /* ── Headers ──────────────────────────────────────────── */
    h1 {
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    h2 {
        border-bottom: 3px solid var(--secondary);
        padding-bottom: 10px;
        margin-top: 30px;
    }

    /* ── Input section ────────────────────────────────────── */
    .input-section {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }

    /* ── Footer ───────────────────────────────────────────── */
    .footer-text {
        color: var(--text-muted);
        font-size: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# LOAD MODEL & DATA
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = r'D:/Material/ITI/Python Visualization/Final Project/Predection_Model/churn_model.pkl'
    """Load the trained model and encoders from pickle file"""
    try:
        model_data = joblib.load(model)
        return model_data
    except FileNotFoundError:
        st.error("❌ Model file 'churn_model.pkl' not found. Please upload it to the app directory.")
        st.stop()

model_data = load_model()
model = model_data['model']
encoders = model_data['encoders']
feature_names = model_data['feature_names']
model_name = model_data['model_name']
cap_values = model_data['cap_values']

# ──────────────────────────────────────────────────────────────
# NAVIGATION SIDEBAR
# ──────────────────────────────────────────────────────────────
st.sidebar.markdown("# 🏦 Bank Churn Predictor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "❓ Business Questions", "🔮 Make Prediction", "💡 Insights", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
    - Model Type: XGBoost
    - ROC-AUC Score: 0.9919
    - Accuracy: 96.84%
    - Training Date: 2024
""")

# ──────────────────────────────────────────────────────────────
# PAGE 1: DASHBOARD
# ──────────────────────────────────────────────────────────────

# Professional Color Palette
COLORS = {
    "primary": "#1E3A5F",      # Deep Navy
    "secondary": "#2E5A88",    # Steel Blue
    "success": "#2E7D32",      # Forest Green
    "danger": "#C62828",       # Deep Red
    "warning": "#F57C00",      # Orange
    "info": "#00838F",         # Teal
    "dark_bg": "#F5F7FA",     # Light Gray
    "light_bg": "#1a1a2e",      # Dark Navy
    "existing": "#2E7D32",     # Green for existing
    "attrited": "#C62828",     # Red for attrited
    "gradient": ["#1E3A5F", "#2E5A88", "#3A7CA5"]
}

# ──────────────────────────────────────────────────────────────
# PAGE 1: DASHBOARD
# ──────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.markdown("# 🏦 Bank Customer Churn Analysis")
    st.markdown("### Data-Driven Insights Dashboard")

    @st.cache_data
    def load_data():
        df = pd.read_csv(r"D:/Material/ITI/Python Visualization/Final Project/Raw_Data/BankChurners.csv")
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
        df['Churned'] = df['Attrition_Flag']
        df['Attrition_Label'] = df['Attrition_Flag'].map({0: 'Existing', 1: 'Attrited'})
        df['Gender_Label'] = df['Gender'].map({'M': 'Male', 'F': 'Female'})
        return df

    df = load_data()

    # ─────────────────────────────
    # KPI CARDS (Row 1)
    # ─────────────────────────────
    total_customers = len(df)
    churn_rate = (df['Attrition_Flag'] == 1).mean() * 100
    avg_age = df['Customer_Age'].mean()
    avg_transactions = df['Total_Trans_Ct'].mean()
    avg_credit = df['Credit_Limit'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    # Shared card style — equal height, left-aligned label, emoji after text
    CARD_H = "min-height:110px; display:flex; flex-direction:column; justify-content:center;"

    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
                    padding: 18px 16px; border-radius: 12px; color: white; {CARD_H}'>
            <div style='font-size:12px; font-weight:600; opacity:0.85; letter-spacing:0.5px;
                        text-transform:uppercase; text-align:left;'>TOTAL CUSTOMERS &nbsp;👥</div>
            <div style='font-size:30px; font-weight:800; margin-top:8px;
                        text-align:left;'>{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS["danger"]}, #e57373);
                    padding: 18px 16px; border-radius: 12px; color: white; {CARD_H}'>
            <div style='font-size:12px; font-weight:600; opacity:0.85; letter-spacing:0.5px;
                        text-transform:uppercase; text-align:left;'>CHURN RATE &nbsp;📉</div>
            <div style='font-size:30px; font-weight:800; margin-top:8px;
                        text-align:left;'>{churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS["info"]}, #26a69a);
                    padding: 18px 16px; border-radius: 12px; color: white; {CARD_H}'>
            <div style='font-size:12px; font-weight:600; opacity:0.85; letter-spacing:0.5px;
                        text-transform:uppercase; text-align:left;'>AVG AGE &nbsp;🎂</div>
            <div style='font-size:30px; font-weight:800; margin-top:8px;
                        text-align:left;'>{avg_age:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS["success"]}, #66bb6a);
                    padding: 18px 16px; border-radius: 12px; color: white; {CARD_H}'>
            <div style='font-size:12px; font-weight:600; opacity:0.85; letter-spacing:0.5px;
                        text-transform:uppercase; text-align:left;'>AVG TRANSACTIONS &nbsp;💳</div>
            <div style='font-size:30px; font-weight:800; margin-top:8px;
                        text-align:left;'>{avg_transactions:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {COLORS["warning"]}, #ffb74d);
                    padding: 18px 16px; border-radius: 12px; color: white; {CARD_H}'>
            <div style='font-size:12px; font-weight:600; opacity:0.85; letter-spacing:0.5px;
                        text-transform:uppercase; text-align:left;'>AVG CREDIT &nbsp;💰</div>
            <div style='font-size:30px; font-weight:800; margin-top:8px;
                        text-align:left;'>${avg_credit:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ─────────────────────────────
    # ROW 2: Churn Distribution + Age Distribution
    # ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Churn Distribution")
        churn_counts = df['Attrition_Label'].value_counts()
        fig1 = go.Figure(data=[go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            hole=0.5,
            marker_colors=[COLORS["existing"], COLORS["attrited"]],
            textinfo='percent+label',
            textposition='auto',
            showlegend=False
        )])
        fig1.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 👤 Age Distribution")
        existing_age = df[df['Attrition_Flag'] == 0]['Customer_Age']
        attrited_age = df[df['Attrition_Flag'] == 1]['Customer_Age']
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=existing_age, name='Existing', 
                                     marker_color=COLORS["existing"], opacity=0.7))
        fig2.add_trace(go.Histogram(x=attrited_age, name='Attrited', 
                                     marker_color=COLORS["attrited"], opacity=0.7))
        fig2.update_layout(barmode='overlay', height=375, margin=dict(t=20, b=20),
                          xaxis_title="Age", yaxis_title="Count", legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig2, use_container_width=True)

    # ─────────────────────────────
    # ROW 3: Transactions + Credit Limit
    # ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💳 Transaction Count")
        fig3 = go.Figure()
        fig3.add_trace(go.Box(x=df[df['Attrition_Flag'] == 0]['Total_Trans_Ct'], 
                               name='Existing', marker_color=COLORS["existing"], boxmean='sd'))
        fig3.add_trace(go.Box(x=df[df['Attrition_Flag'] == 1]['Total_Trans_Ct'], 
                               name='Attrited', marker_color=COLORS["attrited"], boxmean='sd'))
        fig3.update_layout(height=375, margin=dict(t=20, b=20), 
                          xaxis_title="Customer Status", yaxis_title="Transaction Count")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.markdown("#### 💰 Credit Limit")
        fig4 = go.Figure()
        fig4.add_trace(go.Box(x=df[df['Attrition_Flag'] == 0]['Credit_Limit'], 
                               name='Existing', marker_color=COLORS["existing"], boxmean='sd'))
        fig4.add_trace(go.Box(x=df[df['Attrition_Flag'] == 1]['Credit_Limit'], 
                               name='Attrited', marker_color=COLORS["attrited"], boxmean='sd'))
        fig4.update_layout(height=375, margin=dict(t=20, b=20),
                          xaxis_title="Customer Status", yaxis_title="Credit Limit ($)")
        st.plotly_chart(fig4, use_container_width=True)

    # ─────────────────────────────
    # ROW 4: Gender + Income Category
    # ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👥 Churn by Gender")
        gender_churn = df.groupby('Gender_Label')['Attrition_Flag'].mean() * 100
        fig5 = go.Figure(data=[go.Bar(
            x=gender_churn.index,
            y=gender_churn.values,
            marker_color=[COLORS["info"], COLORS["secondary"]],
            text=gender_churn.round(1).astype(str) + '%',
            textposition='outside'
        )])
        fig5.update_layout(height=440, margin=dict(t=20, b=20),
                          xaxis_title="Gender", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.markdown("#### 💵 Churn by Income Category")
        income_churn = df.groupby('Income_Category')['Attrition_Flag'].mean() * 100
        income_churn = income_churn.sort_values(ascending=False)
        colors = px.colors.sequential.Oranges_r[:len(income_churn)]
        fig6 = go.Figure(data=[go.Bar(
            x=income_churn.index,
            y=income_churn.values,
            marker_color=colors,
            text=income_churn.round(1).astype(str) + '%',
            textposition='outside'
        )])
        fig6.update_layout(height=440, margin=dict(t=20, b=60),
                          xaxis_title="Income Category", yaxis_title="Churn Rate (%)",
                          xaxis_tickangle=-30)
        st.plotly_chart(fig6, use_container_width=True)

    # ─────────────────────────────
    # ROW 5: Inactive Months + Contacts
    # ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⏳ Inactive Months Impact")
        inactive_churn = df.groupby('Months_Inactive_12_mon')['Attrition_Flag'].mean() * 100
        colors_inactive = px.colors.sequential.Reds_r[:len(inactive_churn)]
        fig7 = go.Figure(data=[go.Bar(
            x=inactive_churn.index.astype(str),
            y=inactive_churn.values,
            marker_color=colors_inactive,
            text=inactive_churn.round(1).astype(str) + '%',
            textposition='outside'
        )])
        fig7.update_layout(height=375, margin=dict(t=20, b=20),
                          xaxis_title="Months Inactive (Last 12)", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig7, use_container_width=True)

    with col2:
        st.markdown("#### 📞 Customer Service Contacts")
        contacts_churn = df.groupby('Contacts_Count_12_mon')['Attrition_Flag'].mean() * 100
        colors_contacts = px.colors.sequential.Blues_r[:len(contacts_churn)]
        fig8 = go.Figure(data=[go.Bar(
            x=contacts_churn.index.astype(str),
            y=contacts_churn.values,
            marker_color=colors_contacts,
            text=contacts_churn.round(1).astype(str) + '%',
            textposition='outside'
        )])
        fig8.update_layout(height=375, margin=dict(t=20, b=20),
                          xaxis_title="Number of Contacts (Last 12)", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig8, use_container_width=True)

    # ─────────────────────────────
    # ROW 6: Products + Card Category
    # ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏦 Products Held vs Churn")
        product_churn = df.groupby('Total_Relationship_Count')['Attrition_Flag'].mean() * 100
        colors_product = px.colors.sequential.Greens_r[:len(product_churn)]
        fig9 = go.Figure(data=[go.Bar(
            x=product_churn.index.astype(str),
            y=product_churn.values,
            marker_color=colors_product,
            text=product_churn.round(1).astype(str) + '%',
            textposition='outside'
        )])
        fig9.update_layout(height=375, margin=dict(t=20, b=20),
                          xaxis_title="Number of Products", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig9, use_container_width=True)

    with col2:
        st.markdown("#### 💳 Churn by Card Category")
        card_churn = df.groupby('Card_Category')['Attrition_Flag'].mean() * 100
        card_churn = card_churn.sort_values(ascending=False)
        colors_card = px.colors.sequential.Purples_r[:len(card_churn)]
        fig10 = go.Figure(data=[go.Bar(
            x=card_churn.index,
            y=card_churn.values,
            marker_color=colors_card,
            text=card_churn.round(1).astype(str) + '%',
            textposition='outside'
        )])
        fig10.update_layout(height=375, margin=dict(t=20, b=20),
                           xaxis_title="Card Category", yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig10, use_container_width=True)

    # ─────────────────────────────
    # ROW 7: Transaction Amount + Utilization Ratio
    # ─────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💵 Transaction Amount")
        fig11 = go.Figure()
        fig11.add_trace(go.Histogram(x=df[df['Attrition_Flag'] == 0]['Total_Trans_Amt'], 
                                      name='Existing', marker_color=COLORS["existing"], opacity=0.6))
        fig11.add_trace(go.Histogram(x=df[df['Attrition_Flag'] == 1]['Total_Trans_Amt'], 
                                      name='Attrited', marker_color=COLORS["attrited"], opacity=0.6))
        fig11.update_layout(barmode='overlay', height=375, margin=dict(t=20, b=20),
                           xaxis_title="Transaction Amount ($)", yaxis_title="Count",
                           legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig11, use_container_width=True)

    with col2:
        st.markdown("#### 📈 Utilization Ratio")
        fig12 = go.Figure()
        fig12.add_trace(go.Histogram(x=df[df['Attrition_Flag'] == 0]['Avg_Utilization_Ratio'], 
                                      name='Existing', marker_color=COLORS["existing"], opacity=0.6))
        fig12.add_trace(go.Histogram(x=df[df['Attrition_Flag'] == 1]['Avg_Utilization_Ratio'], 
                                      name='Attrited', marker_color=COLORS["attrited"], opacity=0.6))
        fig12.update_layout(barmode='overlay', height=375, margin=dict(t=20, b=20),
                           xaxis_title="Avg Utilization Ratio", yaxis_title="Count",
                           legend=dict(orientation='h', y=1.05))
        st.plotly_chart(fig12, use_container_width=True)

    # ─────────────────────────────
    # ROW 8: Correlation Heatmap
    # ─────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔥 Feature Correlation Heatmap")
    
    numeric_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                    'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Credit_Limit',
                    'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Avg_Utilization_Ratio',
                    'Attrition_Flag']
    
    corr_matrix = df[numeric_cols].corr()
    
    fig13 = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate='%{text}',
        textfont={"size": 9}
    ))
    fig13.update_layout(height=550, margin=dict(t=30, b=30))
    st.plotly_chart(fig13, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# PAGE 2: DETAILED ANALYSIS
# ──────────────────────────────────────────────────────────────
elif page == "📈 Detailed Analysis":
    st.markdown("# 📈 Detailed Statistical Analysis")
    
    @st.cache_data
    def load_data():
        df = pd.read_csv(r"D:/Material/ITI/Python Visualization/Final Project/Raw_Data/BankChurners.csv")
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
        df['Attrition_Label'] = df['Attrition_Flag'].map({0: 'Existing', 1: 'Attrited'})
        return df
    
    df = load_data()
    
    st.subheader("📊 Summary Statistics by Churn Status")
    numeric_cols = ['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct',
                    'Avg_Utilization_Ratio', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon']
    
    summary = df.groupby('Attrition_Label')[numeric_cols].agg(['mean', 'median', 'std'])
    st.dataframe(summary.round(2), use_container_width=True)

# ──────────────────────────────────────────────────────────────
# PAGE: INSIGHTS
# ──────────────────────────────────────────────────────────────

elif page == "💡 Insights":
    st.markdown("# 💡 Insights & Recommendations")

    # Hero banner
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {COLORS["primary"]}, {COLORS["secondary"]});
                padding: 30px; border-radius: 16px; color: white; text-align: center; margin-bottom: 30px;'>
        <h2 style='margin:0;'>📌 Key Findings & Strategic Recommendations</h2>
        <p style='font-size: 18px; opacity: 0.95; margin:8px 0 0;'>Data-driven insights to reduce customer churn by up to 20%</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Insights cards ───────────────────────────────────
    st.markdown("### 🔑 Key Insights")
    insights = [
        (COLORS["danger"],   "📉", "Churn rate is <strong>16.07%</strong> — 1 in 6 customers leaves"),
        (COLORS["primary"],  "💳", "Low transaction activity is the <strong>#1 churn signal</strong> (31.7% model importance)"),
        (COLORS["warning"],  "😴", "Inactivity of <strong>3+ months</strong> sharply increases churn"),
        (COLORS["success"],  "🛒", "Customers with <strong>1–2 products</strong> churn at 27% vs 10% with 6 products"),
        (COLORS["info"],     "💵", "<strong>Lowest income group</strong> shows the highest churn rate"),
        (COLORS["secondary"],"🟦", "<strong>Blue card holders</strong> (93%) represent the largest churn volume"),
    ]

    col_a, col_b = st.columns(2)
    for i, (color, emoji, text) in enumerate(insights):
        target = col_a if i % 2 == 0 else col_b
        target.markdown(f"""
        <div style='background:{COLORS["light_bg"]}; padding:16px 20px; border-radius:10px;
                    border-left:5px solid {color}; margin-bottom:14px;'>
            <span style='font-size:20px;'>{emoji}</span>
            <span style='font-size:15px; margin-left:10px;'>{text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Actionable Recommendations ───────────────────────────
    st.markdown("### 🎯 Actionable Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='background:{COLORS["light_bg"]}; padding:20px; border-radius:12px;
                    border-left:4px solid {COLORS["danger"]}; margin-bottom:20px;'>
            <h3 style='color:{COLORS["danger"]}; margin-top:0;'>⚠️ High Priority</h3>
            <ul style='font-size:15px; line-height:1.8;'>
                <li><strong>Transaction Decline Alerts</strong> — Early-warning system for declining activity
                    <span style='color:{COLORS["success"]};'> ↑ ~20% reduction</span></li>
                <li><strong>Inactive Customer Re-engagement</strong> — Activate after 2 months of inactivity
                    <span style='color:{COLORS["success"]};'> ↑ ~15% reduction</span></li>
                <li><strong>Low Transaction Intervention</strong> — Targeted offers for &lt;50 transactions/year</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{COLORS["light_bg"]}; padding:20px; border-radius:12px;
                    border-left:4px solid {COLORS["warning"]}; margin-bottom:20px;'>
            <h3 style='color:{COLORS["warning"]}; margin-top:0;'>🟡 Medium Priority</h3>
            <ul style='font-size:15px; line-height:1.8;'>
                <li><strong>Cross-Sell Products</strong> — Target 1–2 product customers
                    <span style='color:{COLORS["success"]};'> ↑ ~10% reduction</span></li>
                <li><strong>Credit Limit Review</strong> — Increase limits for loyal customers
                    <span style='color:{COLORS["success"]};'> ↑ ~8% reduction</span></li>
                <li><strong>Card Category Upgrade</strong> — Promote engaged Blue cardholders to Silver/Gold</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:{COLORS["light_bg"]}; padding:20px; border-radius:12px;
                    border-left:4px solid {COLORS["success"]}; margin-bottom:20px;'>
            <h3 style='color:{COLORS["success"]}; margin-top:0;'>🟢 Quick Wins</h3>
            <ul style='font-size:15px; line-height:1.8;'>
                <li><strong>Low-Income Loyalty Programs</strong> — Targeted rewards for &lt;$40K income
                    <span style='color:{COLORS["success"]};'> ↑ ~5% reduction</span></li>
                <li><strong>Customer Service Training</strong> — Reduce unnecessary contact escalations</li>
                <li><strong>Welcome Back Offers</strong> — Special incentives for returning inactive customers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:{COLORS["light_bg"]}; padding:20px; border-radius:12px;
                    border-left:4px solid {COLORS["info"]};'>
            <h3 style='color:{COLORS["info"]}; margin-top:0;'>📊 Key Metrics to Monitor</h3>
            <ul style='font-size:15px; line-height:1.8;'>
                <li><strong>Transaction Count</strong> — Monthly change tracking</li>
                <li><strong>Inactivity Duration</strong> — 30 / 60 / 90 day milestones</li>
                <li><strong>Product Adoption</strong> — Cross-sell conversion rates</li>
                <li><strong>Credit Limit Utilization</strong> — Usage patterns by segment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Recommendations Summary Table ────────────────────────
    st.markdown("### 📋 Recommendations Summary")
    recs = pd.DataFrame({
        '#': [1, 2, 3, 4, 5, 6],
        'Recommendation': [
            'Early-warning alerts for declining transaction count',
            'Re-engage customers inactive 2+ months with offers',
            'Cross-sell extra products to 1–2 product holders',
            'Offer credit limit increases to loyal low-limit users',
            'Loyalty programs targeting sub-$40K income bracket',
            'Upgrade engaged Blue cardholders to Silver/Gold',
        ],
        'Target Segment': [
            'All customers', 'Months_Inactive ≥ 2',
            'Products = 1–2', 'Credit_Limit < median',
            'Income < $40K', 'Blue card power users',
        ],
        'Expected Impact': [
            'Catch churn early', 'Reduce churn ~20%',
            'Cut churn 27% → 10%', 'Boost loyalty',
            'Reduce bracket churn', 'Upsell revenue',
        ],
        'Priority': ['🔴 High', '🔴 High', '🟡 Medium', '🟡 Medium', '🟢 Quick Win', '🟢 Quick Win'],
    })
    st.dataframe(recs, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Customer Risk Distribution Chart ─────────────────────
    st.markdown("### 🎯 Customer Risk Score Distribution")
    st.caption("Risk score based on: Inactive ≥3 months (+30), Products ≤2 (+25), Trans Ct <60 (+25), Credit Limit <$5K (+20)")

    @st.cache_data
    def load_risk_data():
        df_r = pd.read_csv(r"D:/Material/ITI/Python Visualization/Final Project/Raw_Data/BankChurners.csv")
        df_r['Risk_Score'] = (
            (df_r['Months_Inactive_12_mon'] >= 3) * 30 +
            (df_r['Total_Relationship_Count'] <= 2) * 25 +
            (df_r['Total_Trans_Ct'] < 60) * 25 +
            (df_r['Credit_Limit'] < 5000) * 20
        )
        return df_r

    df_risk = load_risk_data()
    risk_counts = df_risk['Risk_Score'].value_counts().sort_index()
    risk_colors = {0: COLORS["success"], 25: COLORS["success"], 30: COLORS["warning"],
                   45: COLORS["warning"], 50: COLORS["warning"], 55: COLORS["danger"],
                   75: COLORS["danger"], 80: "#7b1fa2", 100: "#7b1fa2"}
    bar_colors = [risk_colors.get(k, COLORS["info"]) for k in risk_counts.index]

    fig_risk = go.Figure(data=[go.Bar(
        x=risk_counts.index, y=risk_counts.values,
        marker_color=bar_colors,
        text=risk_counts.values, textposition='outside'
    )])
    fig_risk.update_layout(
        title="Customer Risk Distribution by Composite Score",
        height=420, xaxis_title="Risk Score", yaxis_title="Number of Customers",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)"),
        title_font=dict(color="rgba(255,255,255,0.95)"),
        xaxis=dict(title_font=dict(color="rgba(255,255,255,0.85)"), tickfont=dict(color="rgba(255,255,255,0.75)"), gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title_font=dict(color="rgba(255,255,255,0.85)"), tickfont=dict(color="rgba(255,255,255,0.75)"), gridcolor="rgba(255,255,255,0.1)")
    )
    st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {COLORS["success"]}, #66bb6a);
                padding: 18px 24px; border-radius: 12px; color: white; text-align: center; margin-top: 10px;'>
        ✅ Use the <strong>Make Prediction</strong> page to identify at-risk customers before they leave.
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PAGE: MAKE PREDICTION
# ──────────────────────────────────────────────────────────────
elif page == "🔮 Make Prediction":
    st.markdown("# 🔮 Predict Customer Churn")
    st.write("Enter customer details below to predict their churn risk")
    
    st.markdown("---")
    
    # Create input form with organized sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Demographics")
        customer_age = st.slider("Age", 18, 100, 45)
        gender = st.selectbox("Gender", ["M", "F"])
        dependent_count = st.slider("Number of Dependents", 0, 5, 2)
        marital_status = st.selectbox(
            "Marital Status",
            ["Unknown", "Single", "Married", "Divorced"]
        )
        education_level = st.selectbox(
            "Education Level",
            ["Unknown", "Uneducated", "High School", "College", "Graduate", "Post-Graduate", "Doctorate"]
        )
    
    with col2:
        st.markdown("### 💳 Account Information")
        months_on_book = st.slider("Months on Book", 6, 56, 24)
        credit_limit = st.number_input("Credit Limit ($)", 1000, 50000, 10000, step=500)
        card_category = st.selectbox(
            "Card Category",
            ["Blue", "Silver", "Gold", "Platinum"]
        )
        income_category = st.selectbox(
            "Income Category",
            ["Unknown", "Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +"]
        )
    
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 💰 Transaction Activity")
        total_trans_ct = st.slider("Total Transaction Count", 0, 150, 60)
        total_trans_amt = st.number_input("Total Transaction Amount ($)", 0, 200000, 50000, step=1000)
        avg_utilization_ratio = st.slider("Avg Utilization Ratio", 0.0, 1.0, 0.35, 0.01)
        total_revolving_bal = st.number_input("Total Revolving Balance ($)", 0, 30000, 1000, step=100)
    
    with col4:
        st.markdown("### 📊 Recent Changes & Contact")
        total_amt_chng_q4_q1 = st.number_input("Qty Trans Amount Change Q4-Q1", -0.5, 3.0, 0.5, 0.05)
        total_ct_chng_q4_q1 = st.number_input("Qty Trans Count Change Q4-Q1", -0.5, 3.0, 0.5, 0.05)
        contacts_count_12_mon = st.slider("Contacts Count 12 Months", 0, 10, 2)
        months_inactive_12_mon = st.slider("Months Inactive 12 Months", 0, 12, 2)
        total_relationship_count = st.slider("Total Relationship Count", 1, 6, 3)
        avg_open_to_buy = st.number_input("Avg Open to Buy ($)", 0, 50000, 5000, step=500)
    
    st.markdown("---")
    
    if st.button("🎯 Predict Churn", use_container_width=True):
        # Prepare input data
        input_data = {
            'Customer_Age': customer_age,
            'Gender': 0 if gender == 'M' else 1,
            'Dependent_count': dependent_count,
            'Months_on_book': months_on_book,
            'Credit_Limit': credit_limit,
            'Total_Revolving_Bal': total_revolving_bal,
            'Avg_Open_To_Buy': avg_open_to_buy,
            'Avg_Utilization_Ratio': avg_utilization_ratio,
            'Months_Inactive_12_mon': months_inactive_12_mon,
            'Contacts_Count_12_mon': contacts_count_12_mon,
            'Total_Relationship_Count': total_relationship_count,
            'Total_Trans_Ct': total_trans_ct,
            'Total_Trans_Amt': total_trans_amt,
            'Total_Ct_Chng_Q4_Q1': total_ct_chng_q4_q1,
            'Total_Amt_Chng_Q4_Q1': total_amt_chng_q4_q1,
            'Education_Level': encoders['Education_Level'].index(education_level),
            'Income_Category': encoders['Income_Category'].index(income_category),
            'Card_Category': encoders['Card_Category'].index(card_category),
            'Marital_Status': encoders['Marital_Status'].index(marital_status),
        }
        
        # Create dataframe with correct column order
        X_new = pd.DataFrame([input_data])[feature_names]
        
        # Make prediction
        prediction = model.predict(X_new)[0]
        probability = model.predict_proba(X_new)[0]
        
        # Display results
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        col_pred1, col_pred2 = st.columns([2, 1])
        
        with col_pred1:
            if prediction == 1:
                risk_level = "HIGH"
                risk_color = "🔴"
                box_class = "churn-risk-high"
                churn_prob = probability[1]
            else:
                if probability[1] > 0.35:
                    risk_level = "MEDIUM"
                    risk_color = "🟡"
                    box_class = "churn-risk-medium"
                else:
                    risk_level = "LOW"
                    risk_color = "🟢"
                    box_class = "churn-risk-low"
                churn_prob = probability[1]
            
            st.markdown(f"""
            <div class="{box_class}" style="padding: 30px; border-radius: 12px; text-align: center; margin: 20px 0;">
                <h2 style="margin: 0; font-size: 40px;">{risk_color} {risk_level}</h2>
                <p style="margin: 10px 0; font-size: 18px;">Churn Risk: <strong>{churn_prob*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_pred2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=churn_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk %"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "lightyellow"},
                        {'range': [66, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📈 Probability Distribution")
        
        fig_prob = go.Figure(data=[
            go.Bar(
                x=['Existing Customer', 'Attrited Customer'],
                y=[probability[0]*100, probability[1]*100],
                marker_color=['#16a34a', '#dc2626'],
                text=[f'{probability[0]*100:.1f}%', f'{probability[1]*100:.1f}%'],
                textposition='auto',
            )
        ])
        fig_prob.update_layout(
            title='Prediction Probability',
            yaxis_title='Probability (%)',
            height=300
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ──────────────────────────────────────────────────────────────
# PAGE 4: ABOUT
# ──────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("# ℹ️ About This Application")
    
    st.markdown("""
    ## 🏦 Bank Churn Prediction System
    
    This application uses machine learning to predict the likelihood of customers 
    leaving their bank. It helps financial institutions identify at-risk customers 
    and implement proactive retention strategies.
    
    ### 📚 Dataset Overview
    - **Source:** Bank Churners Dataset
    - **Total Records:** 10,127 customers
    - **Churn Rate:** 16.07%
    - **Features:** 23 customer attributes
    
    ### 🤖 Model Details
    
    **Selected Model:** XGBoost Classifier
    
    **Why XGBoost?**
    - Best overall performance (ROC-AUC: 99.19%)
    - Balanced precision-recall trade-off
    - Handles class imbalance effectively
    - Fast prediction time for real-time applications
    
    **Training Setup:**
    - Train-Test Split: 80-20
    - Class Weight: Balanced to handle 84-16 imbalance
    - Hyperparameters: Optimized for churn detection
    - Cross-Validation: Stratified K-Fold
    
    ### 📊 Performance Metrics
    
    | Metric | Score |
    |--------|-------|
    | Accuracy | 96.84% |
    | Precision | 90.40% |
    | Recall | 89.85% |
    | F1-Score | 90.12% |
    | ROC-AUC | 99.19% |
    
    ### 🎯 Use Cases
    
    1. **Retention Campaign Targeting** - Identify customers most likely to leave
    2. **Resource Allocation** - Focus retention efforts on high-risk segments
    3. **Customer Segmentation** - Understand churn drivers by customer type
    4. **Product Development** - Use insights to improve customer satisfaction
    
    ### 🔐 Data Privacy
    This application does not store or transmit customer data. All predictions 
    are made locally and user inputs are not saved.
    
    """)

# ──────────────────────────────────────────────────────────────
# PAGE: BUSINESS QUESTIONS (FORCED DARK STYLE)
# ──────────────────────────────────────────────────────────────
elif page == "❓ Business Questions":

    # ---------- FORCED DARK CSS OVERRIDES ----------
    st.markdown("""
    <style>

    /* Full app background */
    .stApp {
        background-color: #0e1117 !important;
    }

    /* Containers */
    .block-container {
        background-color: #0e1117 !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e1e1e !important;
        color: white !important;
    }

    .streamlit-expanderContent {
        background-color: #1e1e1e !important;
        color: white !important;
    }

    /* Text */
    h1, h2, h3, h4, h5, h6, p, span {
        color: white !important;
    }

    /* Dataframe */
    .stDataFrame {
        background-color: #1e1e1e !important;
        color: white !important;
    }

    /* Remove white cards globally */
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent !important;
    }

    </style>
    """, unsafe_allow_html=True)
    # ---------- DATA LOADING ----------
    # Make sure the file path is correct
    # df = pd.read_csv("BankChurners.csv")  
    df= pd.read_csv(r"D:/Material/ITI/Python Visualization/Final Project/Raw_Data/BankChurners.csv") # or use your full path
    df['Churned'] = (df['Attrition_Flag'] == 'Attrited Customer').astype(int)

    # ---------- Q1: Overall churn rate ----------
    with st.expander('📊 Q1 — What is the overall churn rate?', expanded=True):
        rate = df['Churned'].mean() * 100

        col1, col2 = st.columns([1, 2])

        # col2.metric('Churn Rate', f'{rate:.2f}%')
        # col2.metric('Total Churned', f'{df["Churned"].sum():,}')

        col1.markdown(f"""
        Out of **{len(df):,}** customers,  
        **{df["Churned"].sum():,}** have churned.  

        👉 That represents **{rate:.2f}%** of the total customer base.
        """)
    # ---------- Q2: Income vs churn ----------
    with st.expander('💰 Q2 — Which income group churns the most?'):
        
        # Calculate stats
        grouped = df.groupby('Income_Category').agg(
            churn_rate=('Churned', 'mean'),
            total_customers=('Churned', 'count'),
            churned_customers=('Churned', 'sum')
        ).sort_values(by='churn_rate', ascending=False)

        grouped['churn_rate'] *= 100

        # Create detailed text for each bar
        grouped['text'] = (
            grouped['churn_rate'].round(2).astype(str) + '%<br>' 
            # 'Churned: ' + grouped['churned_customers'].astype(str) + '<br>' 
            # 'Total: ' + grouped['total_customers'].astype(str)
        )

        # Plot
        fig = px.bar(
            x=grouped.index,
            y=grouped['churn_rate'],
            template="plotly_dark",
            labels={"x": "Income Category", "y": "Churn Rate (%)"},
            title="Churn Rate by Income Category",
            text=grouped['text']
        )

        fig.update_traces(
            textposition='outside',
            hovertemplate=
            "<b>%{x}</b><br>" +
            "Churn Rate: %{y:.2f}%<br>" +
            "%{text}<extra></extra>"
        )

        fig.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#0e1117',
            font=dict(color='white'),
            height=570
        )

        st.plotly_chart(fig, theme=None)

        # Insight
        st.warning(f'⚠️ Highest churn: **{grouped.index[0]}** ({grouped["churn_rate"].iloc[0]:.2f}%)')
        # ---------- Q3: Inactivity impact ----------
    with st.expander('📉 Q3 — Do inactive customers churn more?'):
            inac = df.groupby('Months_Inactive_12_mon')['Churned'].mean() * 100

            fig = px.line(
                x=inac.index,
                y=inac.values,
                template="plotly_dark",
                labels={
                    "x": "Months Inactive (Last 12 Months)",
                    "y": "Churn Rate (%)"
                },
                title="Impact of Inactivity on Churn"
            )

            fig.update_layout(
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#0e1117',
                font=dict(color='white')
            )

            st.plotly_chart(fig, theme=None)

            st.info("📌 Insight: More inactivity → higher churn risk")
    # ---------- Q4: Credit limit ----------
    with st.expander('💳 Q4 — Does credit limit affect churn?'):
        fig = px.box(
            df,
            x='Attrition_Flag',
            y='Credit_Limit',
            template="plotly_dark",
            labels={
                "Attrition_Flag": "Customer Status",
                "Credit_Limit": "Credit Limit ($)"
            },
            title="Credit Limit Distribution by Customer Status"
        )

        fig.update_layout(
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#0e1117',
            font=dict(color='white')
        )

        st.plotly_chart(fig, theme=None)

        st.info("📌 Lower credit limits are associated with higher churn")
    # ---------- Q5: Card category ----------
    with st.expander('🏦 Q5 — Which card category has highest churn?'):
        c = df.groupby('Card_Category')['Churned'].mean() * 100

        fig = px.bar(
        x=c.index,
        y=c.values,
        template="plotly_dark",
        labels={"x": "Card Category", "y": "Churn Rate (%)"},
        title="Churn Rate by Card Category",
        text=[f"{v:.2f}%" for v in c.values]  # 👈 النسبة على الأعمدة
    )

        fig.update_traces(
        textposition='outside'  # 👈 تظهر فوق العمود
    )

        fig.update_layout(
        height=550,  # اختياري لتحسين الشكل
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )

        st.plotly_chart(fig, use_container_width=True)

        st.warning(f'⚠️ Highest churn card: **{c.idxmax()}** ({c.max():.2f}%)')
    # ---------- Q6: Products vs churn ----------
    with st.expander('📦 Q6 — Do customers with more products churn less?'):
        r = df.groupby('Total_Relationship_Count')['Churned'].mean() * 100

        fig = px.bar(
        x=r.index,
        y=r.values,
        template="plotly_dark",
        labels={"x": "Number of Products", "y": "Churn Rate (%)"},
        title="Churn Rate by Number of Products",
        text=[f"{v:.2f}%" for v in r.values]  # 👈 النسبة
    )

        fig.update_traces(
        textposition='outside'
    )

        fig.update_layout(
        height=550,
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#0e1117',
        font=dict(color='white')
    )

        st.plotly_chart(fig, use_container_width=True)

        st.success('✅ More products = lower churn (loyal customers)')    
    # ---------- Q7: Transactions ----------
    with st.expander('📈 Q7 — What transaction behavior predicts churn?'):
        tx = df.groupby('Attrition_Flag')[['Total_Trans_Amt', 'Total_Trans_Ct']].mean().reset_index()

        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.bar(
                tx,
                x='Attrition_Flag',
                y='Total_Trans_Amt',
                template="plotly_dark",
                labels={
                    "Attrition_Flag": "Customer Status",
                    "Total_Trans_Amt": "Avg Transaction Amount ($)"
                },
                title="Average Transaction Amount",
                text=[f"{v:,.0f}" for v in tx['Total_Trans_Amt']]  # 👈 قيمة
            )

            fig1.update_traces(textposition='outside')

            fig1.update_layout(
                height=550,
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#0e1117',
                font=dict(color='white')
            )

            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                tx,
                x='Attrition_Flag',
                y='Total_Trans_Ct',
                template="plotly_dark",
                labels={
                    "Attrition_Flag": "Customer Status",
                    "Total_Trans_Ct": "Avg Transaction Count"
                },
                title="Average Transaction Count",
                text=[f"{v:.0f}" for v in tx['Total_Trans_Ct']]  # 👈 قيمة
            )

            fig2.update_traces(textposition='outside')

            fig2.update_layout(
                height=550,
                plot_bgcolor='#1e1e1e',
                paper_bgcolor='#0e1117',
                font=dict(color='white')
                
            )

            st.plotly_chart(fig2, use_container_width=True)

        st.info("📌 Lower transactions → higher churn probability")        
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.45); font-size: 12px; margin-top: 30px;">
    <p>🏦 Bank Churn Prediction System | Powered by Streamlit & XGBoost</p>
    <p>© 2024 | Machine Learning for Financial Services</p>
</div>
""", unsafe_allow_html=True)