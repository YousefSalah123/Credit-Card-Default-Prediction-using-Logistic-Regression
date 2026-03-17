import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="CrediRisk AI | Default Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS setup
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern minimalist headers */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 700;
    }
    
    /* Premium card containers */
    div.stMetric, div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid var(--border-color);
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA & MODEL CACHING
# ==========================================
@st.cache_data
def load_data():
    """Load the dataset (Cached to prevent reloading)"""
    return pd.read_csv("uci_credit_card.csv")

@st.cache_resource
def train_and_prepare_model(df):
    """
    Train model ONCE and cache it along with the scaler, top features, and evaluation metrics.
    """
    data = df.copy()

    # Clean categorical anomalies
    data['EDUCATION'] = data['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    data['MARRIAGE'] = data['MARRIAGE'].replace({0: 3})

    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    payamt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Feature engineering matching notebook logic
    data['total_bill'] = data[bill_cols].sum(axis=1)
    data['total_pay'] = data[payamt_cols].sum(axis=1)
    data['avg_pay_ratio'] = data[payamt_cols].sum(axis=1) / (data[bill_cols].sum(axis=1) + 1e-6)
    data['delayed_months'] = (data[pay_cols] > 0).sum(axis=1)
    data['max_delay'] = data[pay_cols].max(axis=1)
    data['avg_bill_amt'] = data[bill_cols].mean(axis=1)
    data['avg_pay_amt'] = data[payamt_cols].mean(axis=1)
    data['avg_bill_limit_ratio'] = 1 - (data['avg_bill_amt'] / data['LIMIT_BAL'])
    data['avg_pay_limit_ratio'] = data['avg_pay_amt'] / data['LIMIT_BAL']

    for c in ['avg_bill_limit_ratio', 'avg_pay_limit_ratio', 'avg_pay_ratio']:
        data[c] = data[c].replace([np.inf, -np.inf], 0).fillna(0)

    base_feats = (['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE'] +
                  bill_cols + payamt_cols + pay_cols +
                  ['total_bill', 'total_pay', 'avg_pay_ratio', 'delayed_months', 'max_delay',
                   'avg_bill_amt', 'avg_pay_amt', 'avg_bill_limit_ratio', 'avg_pay_limit_ratio'])

    X = data[base_feats].copy()
    y = data['default.payment.next.month'].copy()

    # Apply one-hot encoding exactly as expected
    X = pd.get_dummies(X, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)
    
    # Store feature columns for later predictions
    feature_cols = X.columns.tolist()

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val)

    for part in [X_train, X_val, X_test]:
        part.replace([np.inf, -np.inf], np.nan, inplace=True)
        part.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=42)
    model.fit(X_train_s, y_train)

    def eval_split(Xs, yt):
        yp = model.predict(Xs)
        return dict(
            accuracy=accuracy_score(yt, yp),
            precision=precision_score(yt, yp),
            recall=recall_score(yt, yp),
            f1=f1_score(yt, yp),
            y_true=yt,
            y_pred=yp
        )

    metrics = {
        "train": eval_split(X_train_s, y_train),
        "val": eval_split(X_val_s, y_val),
        "test": eval_split(X_test_s, y_test)
    }

    coef_series = pd.Series(model.coef_[0], index=X.columns)
    top_coef = coef_series.reindex(coef_series.abs().sort_values(ascending=False).head(12).index)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "top_coef": top_coef,
        "feature_cols": feature_cols,
        "raw_feats": base_feats
    }

# ==========================================
# PREDICTION HELPER FUNCTION
# ==========================================
def preprocess_and_predict(input_dict, pipeline):
    """Takes uncooked inputs, applies FE, scaling, and generates prediction"""
    test_df = pd.DataFrame([input_dict])
    
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    payamt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    
    test_df['total_bill'] = test_df[bill_cols].sum(axis=1)
    test_df['total_pay'] = test_df[payamt_cols].sum(axis=1)
    
    bill_sum = test_df[bill_cols].sum(axis=1)
    test_df['avg_pay_ratio'] = test_df[payamt_cols].sum(axis=1) / (bill_sum + 1e-6)
    
    test_df['delayed_months'] = (test_df[pay_cols] > 0).sum(axis=1)
    test_df['max_delay'] = test_df[pay_cols].max(axis=1)
    test_df['avg_bill_amt'] = test_df[bill_cols].mean(axis=1)
    test_df['avg_pay_amt'] = test_df[payamt_cols].mean(axis=1)
    
    test_df['avg_bill_limit_ratio'] = 1 - (test_df['avg_bill_amt'] / test_df['LIMIT_BAL'])
    test_df['avg_pay_limit_ratio'] = test_df['avg_pay_amt'] / test_df['LIMIT_BAL']
    
    for c in ['avg_bill_limit_ratio', 'avg_pay_limit_ratio', 'avg_pay_ratio']:
        test_df[c] = test_df[c].replace([np.inf, -np.inf], 0).fillna(0)
        
    final_df = pd.DataFrame(0, index=[0], columns=pipeline["feature_cols"])
    
    num_feats = [c for c in pipeline["feature_cols"] if not any(c.startswith(prefix) for prefix in ['SEX_', 'EDUCATION_', 'MARRIAGE_'])]
    for col in num_feats:
        if col in test_df.columns:
            final_df[col] = test_df[col]
            
    sex = test_df['SEX'].values[0]
    education = test_df['EDUCATION'].values[0]
    marriage = test_df['MARRIAGE'].values[0]
    
    if f'SEX_{sex}' in final_df.columns: final_df[f'SEX_{sex}'] = 1
    if f'EDUCATION_{education}' in final_df.columns: final_df[f'EDUCATION_{education}'] = 1
    if f'MARRIAGE_{marriage}' in final_df.columns: final_df[f'MARRIAGE_{marriage}'] = 1

    final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_df.fillna(0, inplace=True)
    
    X_scaled = pipeline['scaler'].transform(final_df)
    prob_default = pipeline['model'].predict_proba(X_scaled)[0][1]
    
    return prob_default

# ==========================================
# APP INITIALIZATION
# ==========================================
try:
    df = load_data()
    pipeline = train_and_prepare_model(df)
except Exception as e:
    st.error(f"Failed to load dataset or train model: {e}")
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h1 style="color: #3B82F6; margin-bottom: 0;">🛡️ CrediRisk AI</h1>
            <p style="color: gray; font-size: 0.9em; margin-top: 5px;">Enterprise Default Prediction</p>
        </div>
        <hr style="border-color: gray; margin-top: 0;">
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        ["Dashboard", "Risk Analysis (EDA)", "Model Architecture", "Single Assessment", "About"],
        label_visibility="collapsed"
    )
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 **Tip:** Use the Assessment tabs to score new clients in real-time.")

# ==========================================
# VIEW: OVERVIEW DASHBOARD
# ==========================================
if menu == "Dashboard":
    st.markdown("<h2>Credit Portfolio Dashboard</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    total_clients = len(df)
    default_rate = df['default.payment.next.month'].mean()
    avg_limit = df['LIMIT_BAL'].mean()
    high_risk = len(df[df['default.payment.next.month'] == 1])
    
    col1.metric("Total Clients", f"{total_clients:,}")
    col2.metric("Portfolio Default Rate", f"{default_rate:.1%}")
    col3.metric("Avg Credit Limit", f"NT${avg_limit:,.0f}")
    col4.metric("High-Risk Profiles", f"{high_risk:,}")
    
    st.divider()
    
    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Default Distribution")
        target_counts = df['default.payment.next.month'].value_counts()
        fig = px.pie(
            names=['Performing (0)', 'Default (1)'], 
            values=target_counts.values,
            hole=0.6,
            color_discrete_sequence=['#10B981', '#EF4444']
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        
    with colB:
        st.subheader("Credit Limit by Education")
        fig = px.box(
            df, 
            x='EDUCATION', 
            y='LIMIT_BAL',
            color='default.payment.next.month',
            color_discrete_sequence=['#10B981', '#EF4444'],
            labels={'EDUCATION': 'Education (1=Grad, 2=Uni, 3=HighSchool)', 'LIMIT_BAL': 'Limit Balance'}
        )
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VIEW: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
elif menu == "Risk Analysis (EDA)":
    st.header("📊 Interactive Risk Analysis")
    st.markdown("Explore macroeconomic and behavioral patterns within the portfolio.")
    
    tab1, tab2, tab3 = st.tabs(["Demographics", "Payment Behavior", "Financials"])
    
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            age_dist = px.histogram(df, x='AGE', nbins=30, color='default.payment.next.month',
                                    barmode='overlay', color_discrete_sequence=['#3B82F6', '#EF4444'],
                                    title="Age Distribution vs Risk")
            st.plotly_chart(age_dist, use_container_width=True)
        with c2:
            gender_risk = df.groupby('SEX')['default.payment.next.month'].mean().reset_index()
            gender_risk['SEX'] = gender_risk['SEX'].map({1: 'Male', 2: 'Female'})
            fig = px.bar(gender_risk, x='SEX', y='default.payment.next.month',
                         title="Default Rate by Gender", text_auto='.1%', color='SEX',
                         color_discrete_sequence=['#636EFA', '#EF553B'])
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        st.subheader("Impact of Delayed Payments (PAY_0)")
        pay0_risk = df.groupby('PAY_0')['default.payment.next.month'].mean().reset_index()
        fig = px.bar(pay0_risk, x='PAY_0', y='default.payment.next.month',
                     labels={'PAY_0': 'Repayment Status (Last Month)', 'default.payment.next.month': 'Default Probability'},
                     color='default.payment.next.month', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Notice the steep exponential increase in default probability as delays extend past 1-2 months.")

    with tab3:
        st.subheader("Feature Correlations")
        feats = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'BILL_AMT1', 'PAY_AMT1', 'default.payment.next.month']
        corr = df[feats].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VIEW: MODEL ARCHITECTURE
# ==========================================
elif menu == "Model Architecture":
    st.header("⚙️ Evaluation & Diagnostics")
    m = pipeline["metrics"]
    
    st.subheader("Performance Verification")
    metrics_df = pd.DataFrame([
        ["Training", m["train"]["accuracy"], m["train"]["precision"], m["train"]["recall"], m["train"]["f1"]],
        ["Validation", m["val"]["accuracy"], m["val"]["precision"], m["val"]["recall"], m["val"]["f1"]],
        ["Holdout Test", m["test"]["accuracy"], m["test"]["precision"], m["test"]["recall"], m["test"]["f1"]],
    ], columns=["Set", "Accuracy", "Precision", "Recall", "F1 Score"])
    
    st.dataframe(metrics_df.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1 Score": "{:.3f}"
    }), use_container_width=True)
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Test Confusion Matrix")
        y_true = m["test"]["y_true"]
        y_pred = m["test"]["y_pred"]
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        fig = px.imshow(cm, text_auto=".2f",
                        labels=dict(x="Predicted", y="Actual"),
                        x=['Performing', 'Default'],
                        y=['Performing', 'Default'],
                        color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
    with colB:
        st.subheader("Feature Significance")
        coef_series = pipeline["top_coef"]
        coef_df = coef_series.reset_index()
        coef_df.columns = ["Feature", "Coefficient Impact"]
        fig = px.bar(coef_df, x="Coefficient Impact", y="Feature", orientation='h',
                     color="Coefficient Impact", color_continuous_scale="RdBu_r")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VIEW: SINGLE ASSESSMENT
# ==========================================
elif menu == "Single Assessment":
    st.header("🎯 Live Risk Assessment")
    st.markdown("Enter client data below to predict the probability of credit default.")
    
    with st.form("single_predict_form", clear_on_submit=False):
        st.subheader("👤 Demographics")
        col1, col2, col3, col4 = st.columns(4)
        age = col1.number_input("Age", min_value=18, max_value=100, value=35)
        limit = col2.number_input("Credit Limit (NT$)", min_value=10000, value=100000, step=10000)
        sex = col3.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        education = col4.selectbox("Education", options=[1, 2, 3, 4], 
                                   format_func=lambda x: {1: 'Graduate', 2: 'University', 3: 'High School', 4: 'Other'}[x])
        marriage = st.selectbox("Marital Status", options=[1, 2, 3], 
                                format_func=lambda x: {1: 'Married', 2: 'Single', 3: 'Other'}[x])
        
        st.markdown("---")
        st.subheader("🕒 Payment Status (Lower is better, -1=Paid, 0=Revolving, 1+=Months Delayed)")
        pcols = st.columns(6)
        pay_vals = [pcols[i].number_input(f"PAY_{[0,2,3,4,5,6][i]}", min_value=-2, max_value=8, value=0) for i in range(6)]
        
        st.markdown("---")
        st.subheader("🧾 Billing History (NT$)")
        bcols = st.columns(6)
        bill_vals = [bcols[i].number_input(f"BILL_AMT{i+1}", value=50000, step=5000) for i in range(6)]
        
        st.markdown("---")
        st.subheader("💰 Payment History (NT$)")
        amcols = st.columns(6)
        pay_amt_vals = [amcols[i].number_input(f"PAY_AMT{i+1}", min_value=0, value=5000, step=1000) for i in range(6)]
        
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Predict Default Risk", use_container_width=True)
        
    if submitted:
        input_data = {
            'LIMIT_BAL': limit, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
            'PAY_0': pay_vals[0], 'PAY_2': pay_vals[1], 'PAY_3': pay_vals[2], 
            'PAY_4': pay_vals[3], 'PAY_5': pay_vals[4], 'PAY_6': pay_vals[5],
            'BILL_AMT1': bill_vals[0], 'BILL_AMT2': bill_vals[1], 'BILL_AMT3': bill_vals[2],
            'BILL_AMT4': bill_vals[3], 'BILL_AMT5': bill_vals[4], 'BILL_AMT6': bill_vals[5],
            'PAY_AMT1': pay_amt_vals[0], 'PAY_AMT2': pay_amt_vals[1], 'PAY_AMT3': pay_amt_vals[2],
            'PAY_AMT4': pay_amt_vals[3], 'PAY_AMT5': pay_amt_vals[4], 'PAY_AMT6': pay_amt_vals[5]
        }
        
        with st.spinner("Analyzing risk profile..."):
            prob = preprocess_and_predict(input_data, pipeline)
            
        st.markdown("---")
        st.subheader("Assessment Result")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prob > 0.5:
                st.error("🚨 HIGH RISK of Default")
                st.markdown(f"<h2 style='color:#EF4444'>{prob*100:.1f}%</h2>", unsafe_allow_html=True)
                st.markdown("Recommendation: Require collateral or deny tier acceleration.")
            elif prob > 0.3:
                st.warning("⚠️ MODERATE RISK")
                st.markdown(f"<h2 style='color:#FBBF24'>{prob*100:.1f}%</h2>", unsafe_allow_html=True)
                st.markdown("Recommendation: Monitor closely, restrict limit increases.")
            else:
                st.success("✅ LOW RISK")
                st.markdown(f"<h2 style='color:#10B981'>{prob*100:.1f}%</h2>", unsafe_allow_html=True)
                st.markdown("Recommendation: Standard operational procedures.")
                
        with res_col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Probability"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#1E293B"},
                    'steps': [
                        {'range': [0, 30], 'color': "#10B981"},
                        {'range': [30, 50], 'color': "#FBBF24"},
                        {'range': [50, 100], 'color': "#EF4444"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                }
            ))
            fig.update_layout(height=250, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# VIEW: ABOUT
# ==========================================
elif menu == "About":
    st.header("ℹ️ About")
    st.markdown("""
        **CrediRisk AI** is an enterprise-grade Credit Card Default Prediction dashboard built by **Yousef Salah**.

        ### Project Overview
        This dashboard utilizes a robust Machine Learning pipeline (Logistic Regression with balanced class weights) to accurately assess the risk of credit card defaults among clients using the UCI Credit Card default dataset.
        
        ### Key Insights
        - **Data Analysis**: Extensive Exploratory Data Analysis (EDA) reveals that severe payment delays and certain demographic combinations significantly elevate default risks.
        - **Model Priorities**: The model specifically optimizes for Recall (identifying defaults correctly) due to the severe cost of false negatives in the financial sector.
        - **Major Predictors**: The strongest predictors of default are historical payment status anomalies and high credit limit utilization.
        
        ### Under the Hood
        - **Frontend**: Streamlit with custom layout and Plotly graphics for dynamic interactivity.
        - **Backend**: Python Machine Learning pipeline trained efficiently and cached for high-performance inference on-the-fly.
        
        *Thank you for exploring the dynamic insights and trying out the live assessment tool!*
    """)
