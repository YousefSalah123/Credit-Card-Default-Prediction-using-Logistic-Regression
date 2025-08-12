import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")

st.title("💳 Credit Card Default Prediction")
st.write("An interactive app to explore data, view model performance, and predict defaults.")

# Sidebar navigation
menu = st.sidebar.radio("Navigate", [
                        "Data Overview", "Exploratory Data Analysis", "Model & Evaluation", "Make Prediction"])


@st.cache_data
def load_data():
    return pd.read_csv("uci_credit_card.csv")


df = load_data()


@st.cache_resource
def train_baseline_logreg(df: pd.DataFrame):
    data = df.copy()

    # Clean categorical anomalies
    data['EDUCATION'] = data['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    data['MARRIAGE'] = data['MARRIAGE'].replace({0: 3})

    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    payamt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                   'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    # Feature engineering
    data['total_bill'] = data[bill_cols].sum(axis=1)
    data['total_pay'] = data[payamt_cols].sum(axis=1)
    data['avg_pay_ratio'] = data[payamt_cols].sum(
        axis=1) / (data[bill_cols].sum(axis=1) + 1e-6)
    data['delayed_months'] = (data[pay_cols] > 0).sum(axis=1)
    data['max_delay'] = data[pay_cols].max(axis=1)
    data['avg_bill_amt'] = data[bill_cols].mean(axis=1)
    data['avg_pay_amt'] = data[payamt_cols].mean(axis=1)
    data['avg_bill_limit_ratio'] = 1 - \
        (data['avg_bill_amt'] / data['LIMIT_BAL'])
    data['avg_pay_limit_ratio'] = data['avg_pay_amt'] / data['LIMIT_BAL']

    for c in ['avg_bill_limit_ratio', 'avg_pay_limit_ratio', 'avg_pay_ratio']:
        data[c] = data[c].replace([np.inf, -np.inf], 0).fillna(0)

    base_feats = (['LIMIT_BAL', 'AGE', 'SEX', 'EDUCATION', 'MARRIAGE'] +
                  bill_cols + payamt_cols + pay_cols +
                  ['total_bill', 'total_pay', 'avg_pay_ratio', 'delayed_months', 'max_delay',
                   'avg_bill_amt', 'avg_pay_amt', 'avg_bill_limit_ratio', 'avg_pay_limit_ratio'])

    X = data[base_feats].copy()
    y = data['default.payment.next.month'].copy()

    X = pd.get_dummies(
        X, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val)

    for part in [X_train, X_val, X_test]:
        part.replace([np.inf, -np.inf], np.nan, inplace=True)
        part.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, solver='liblinear',
                               class_weight='balanced', random_state=42)
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
    top_coef = coef_series.reindex(
        coef_series.abs().sort_values(ascending=False).head(12).index)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "top_coef": top_coef
    }


# ---------- EDA OBSERVATION TEXTS (ADDED) ----------
OBS_TEXT = {
    "target": """**Our Observations:**  
- The target variable is **imbalanced**, with far more non-defaults (0) than defaults (1).  
- Around **78% did not default**, while only **22% defaulted**.  
- This imbalance may cause the model to **underpredict defaults**, which are critical to identify.  
- **Class balancing techniques** (e.g., class weights or resampling) will be necessary for reliable predictions.""",

    "demographics": """**Our Observations:**  
- The dataset contains more **females (label 2)** than males (~18K vs ~12K).  
- **University education (2)** most common, followed by graduate school (1); categories **0, 5, 6** are invalid/unknown → consolidate.  
- Most clients are **single (2)** or **married (1)**; **0 in MARRIAGE** appears invalid.  
- **Cleaning** EDUCATION & MARRIAGE avoids noise; categorical encoding after consolidation improves modeling.""",

    "age": """**Our Observations:**  
- Age distribution is **right‑skewed**, most between **25–40** with a peak near **29**.  
- Long tail above 50; few clients >60 → limited influence.  
- Some high-age outliers exist; scaling suffices (no strong need for transform).""",

    "limit_bal": """**Our Observations:**  
- **Log transformation** reduces right skew and reveals mild multimodality (standard credit tiers).  
- Peaks suggest standardized limit offerings (e.g., product tiers).  
- More symmetric shape aids linear model stability; no extreme post-transform outliers.""",

    "pay_status_all": """**Our Observations:**  
- Across `PAY_0`–`PAY_6`, most values are **0 (on time)** or **-1 (paid duly / early)**.  
- Delays of **1–2 months** appear regularly; extreme delays (≥3) are rare but high risk.  
- Values up to **7/8** indicate severe delinquency.  
- These are **ordinal categorical** features—retain ordering (do not one-hot blindly).  
- Consistent shape across months ⇒ **stable behavioral pattern** useful for prediction.""",

    "bill_amts": """**Our Observations:**  
- `BILL_AMT1`–`BILL_AMT6` are **highly right‑skewed** with many large positive outliers (> NT$1M).  
- Majority of balances < NT$100K.  
- Some **negative values** (refunds / adjustments) should be reviewed.  
- Month-to-month distribution stability suggests aggregated features (averages, totals, ratios) will add value.""",

    "pay_amts": """**Our Observations:**  
- `PAY_AMT1`–`PAY_AMT6` also **right‑skewed** with extreme spikes (large lump payments).  
- Many small / near-zero payments → partial or minimum payments are common.  
- Large payments may indicate lower risk (capacity to settle).  
- Skew + outliers justify log transforms or robust scaling for linear models.""",

    "pay_ratio": """**Our Observations:**  
- Recent payment ratio is extremely **right‑skewed**; majority near zero (small or no payment).  
- Extreme outliers (huge ratios) likely full settlement or anomalies → cap / winsorize.  
- Useful engineered feature after clipping to reduce distortion.""",

    "pay0_vs_default": """**Our Observations:**  
- **Monotonic increase** in default rate as `PAY_0` value rises.  
- Even a **1‑month delay** materially elevates risk; severe delays accelerate sharply.  
- Single most powerful raw indicator among recency variables.""",

    "delayed_months": """**Our Observations:**  
- Default probability **escalates steeply after 2+ delayed months** in last 6.  
- Frequency compounds risk beyond latest status alone—include as engineered count.  
- Nonlinear jump suggests binning or interaction terms could help.""",

    "limit_default_rate": """**Our Observations:**  
- Higher credit limit quintiles show **lower default rates**.  
- Suggests underwriting effectiveness: approved higher limits correlate with stronger repayment capacity.  
- Supports including credit limit + ratios (bill/limit) in engineered features.""",

    "corr": """**Our Observations:**  
- `BILL_AMT*` strongly inter‑correlated (0.80–0.95) → redundancy; consider dimensionality reduction or selective use.  
- `PAY_*` mutually correlated (behavioral consistency) and moderately linked to default (≈0.20–0.32).  
- `LIMIT_BAL` negatively correlated with delays & default (capacity / risk proxy).  
- `AGE` weak overall contribution.  
- Engineering cumulative & ratio features builds on these relationships."""
}

# ---------- EDA HELPER FUNCTIONS (NEW) ----------


def _fig(size=(6, 4)):
    return plt.subplots(figsize=size)


def eda_target(df):
    fig, ax = _fig()
    counts = df["default.payment.next.month"].value_counts()
    sns.countplot(x="default.payment.next.month", data=df,
                  ax=ax, palette=["#4C9BE8", "#E86A5B"])
    ax.set_title("Target Distribution")
    ax.set_xlabel("Default (0/1)")
    total = len(df)
    for i, v in enumerate(counts):
        ax.text(i, v*1.01, f"{v:,}\n{v/total*100:.1f}%",
                ha="center", fontsize=9)
    return fig, "target"


def eda_demographics(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.countplot(x="SEX", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("Sex (1=Male,2=Female)")
    sns.countplot(x="EDUCATION", data=df, ax=axes[1], palette="Set3")
    axes[1].set_title("Education")
    sns.countplot(x="MARRIAGE", data=df, ax=axes[2], palette="viridis")
    axes[2].set_title("Marriage")
    plt.tight_layout()
    return fig, "demographics"


def eda_age(df):
    fig, ax = _fig()
    sns.histplot(df["AGE"], bins=20, kde=True, ax=ax, color="#1B998B")
    ax.set_title("Age Distribution")
    return fig, "age"


def eda_limit_log(df):
    fig, ax = _fig()
    sns.histplot(np.log1p(df["LIMIT_BAL"]), bins=30,
                 kde=True, ax=ax, color="#7B5DD6")
    ax.set_title("Credit Limit (Log Scale)")
    return fig, "limit_bal"


def eda_pay_status_all(df):
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    for ax, col in zip(axes.flatten(), pay_cols):
        sns.countplot(x=col, data=df, ax=ax, palette="RdYlBu_r")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, "pay_status_all"


def eda_bills(df):
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3',
                 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df[bill_cols], ax=ax)
    ax.set_title("Bill Amounts (BILL_AMT1–6)")
    ax.tick_params(axis='x', rotation=45)
    return fig, "bill_amts"


def eda_pay_amts(df):
    payamt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
                   'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df[payamt_cols], ax=ax)
    ax.set_title("Payment Amounts (PAY_AMT1–6)")
    ax.tick_params(axis='x', rotation=45)
    return fig, "pay_amts"


def eda_pay_ratio(df):
    if "pay_ratio_1" not in df.columns:
        df["pay_ratio_1"] = df['PAY_AMT1'] / (df['BILL_AMT1'] + 1e-6)
    ratio = df["pay_ratio_1"].replace([np.inf, -np.inf], np.nan).fillna(0)
    # cap extreme for readability
    ratio = np.clip(ratio, 0, np.percentile(ratio, 99))
    fig, ax = _fig()
    sns.histplot(ratio, bins=50, kde=True, ax=ax, color="#f39c12")
    ax.set_title("Recent Payment Ratio (PAY_AMT1 / BILL_AMT1)")
    return fig, "pay_ratio"


def eda_default_by_pay0(df):
    fig, ax = _fig()
    rate = df.groupby("PAY_0")["default.payment.next.month"].mean()
    rate.plot(kind="bar", ax=ax, color="#D1495B", alpha=.85)
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Rate by PAY_0")
    ax.tick_params(axis="x", rotation=45)
    return fig, "pay0_vs_default"


def eda_default_by_delay_count(df):
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    delayed_cnt = (df[pay_cols] > 0).sum(axis=1)
    agg = pd.DataFrame({"delayed_months": delayed_cnt,
                        "target": df["default.payment.next.month"]}) \
        .groupby("delayed_months")["target"].mean()
    fig, ax = _fig()
    agg.plot(kind="bar", ax=ax, color="#8C4F9F")
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Rate vs # Delayed Months")
    return fig, "delayed_months"


def eda_default_by_limit(df):
    tmp = df.copy()
    tmp["limit_bin"] = pd.qcut(tmp["LIMIT_BAL"], 5, labels=[
                               "Very Low", "Low", "Medium", "High", "Very High"])
    fig, ax = _fig()
    tmp.groupby("limit_bin")["default.payment.next.month"].mean().plot(
        kind="bar", ax=ax, color="#2D87BB")
    ax.set_ylabel("Default Rate")
    ax.set_title("Default Rate by Credit Limit Quintile")
    return fig, "limit_default_rate"


def eda_corr(df):
    feats = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3',
             'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2',
             'default.payment.next.month']
    corr = df[feats].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, cmap="coolwarm", annot=True, fmt=".2f",
                square=True, cbar_kws={"shrink": .75}, ax=ax, center=0)
    ax.set_title("Correlation (Key Features)")
    return fig, "corr"


# ---------- UPDATED RENDER FUNCTION ----------
def render_eda(df):
    st.header("📊 Exploratory Data Analysis")
    st.caption(
        "Figures with original 'Our Observations' commentary from the notebook.")

    # 1. Target
    fig, key = eda_target(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 2. Demographics & Age
    fig, key = eda_demographics(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])
    fig, key = eda_age(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 3. Credit Limit
    fig, key = eda_limit_log(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 4. Payment Status (All Months)
    fig, key = eda_pay_status_all(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 5. Bill Amounts
    fig, key = eda_bills(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 6. Payment Amounts
    fig, key = eda_pay_amts(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 7. Payment Ratio
    fig, key = eda_pay_ratio(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 8. Default Rate by PAY_0
    fig, key = eda_default_by_pay0(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 9. Default Rate vs Delayed Months
    fig, key = eda_default_by_delay_count(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 10. Default Rate by Credit Limit
    fig, key = eda_default_by_limit(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])

    # 11. Correlation
    fig, key = eda_corr(df)
    st.pyplot(fig)
    st.markdown(OBS_TEXT[key])


# ====== ADD / FIX MAIN PAGE ROUTING (MISSING IF CAUSED SYNTAX ERROR) ======
feature_cols = [c for c in df.columns if c not in [
    "ID", "default.payment.next.month"]]

if menu == "Data Overview":
    st.subheader("Dataset Snapshot")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Default Rate",
                f"{df['default.payment.next.month'].mean():.1%}")
    st.markdown("**Preview**")
    st.dataframe(df.head())
    st.markdown("**Missing Values**")
    miss = df.isna().sum()
    st.write(miss[miss > 0] if miss.sum() > 0 else "No missing values.")
    st.markdown("**Class Balance**")
    counts = df["default.payment.next.month"].value_counts()
    st.write(counts.rename({0: "No Default", 1: "Default"}))

elif menu == "Exploratory Data Analysis":
    render_eda(df)

elif menu == "Model & Evaluation":
    st.header("🤖 Model & Evaluation")
    st.caption(
        "Baseline Logistic Regression (class_weight='balanced') at threshold 0.50")
    try:
        results = train_baseline_logreg(df)
        m = results["metrics"]

        # Metrics summary table
        metrics_df = pd.DataFrame([

            ["Train", m["train"]["accuracy"], m["train"]["precision"],
                m["train"]["recall"], m["train"]["f1"]],
            ["Validation", m["val"]["accuracy"], m["val"]
                ["precision"], m["val"]["recall"], m["val"]["f1"]],
            ["Test", m["test"]["accuracy"], m["test"]["precision"],
                m["test"]["recall"], m["test"]["f1"]],
        ], columns=["Split", "Accuracy", "Precision", "Recall", "F1"])
        metrics_df[["Accuracy", "Precision", "Recall", "F1"]] = metrics_df[[
            "Accuracy", "Precision", "Recall", "F1"]].applymap(lambda x: f"{x:.3f}")
        st.subheader("Performance Summary")
        st.dataframe(metrics_df, use_container_width=True)

        # Confusion Matrices (normalized)
        st.subheader("Confusion Matrices (Normalized by Actual Class)")
        cols = st.columns(3)
        for col, (name, data_dict) in zip(cols, m.items()):
            with col:
                y_true = data_dict["y_true"]
                y_pred = data_dict["y_pred"]
                cm_norm = confusion_matrix(y_true, y_pred, normalize='true')
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                            xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(f"{name.title()} Set")
                st.pyplot(fig)

        # Raw confusion (Test)
        st.markdown("**Test Set Raw Confusion Matrix**")
        cm_raw = confusion_matrix(m["test"]["y_true"], m["test"]["y_pred"])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Top Coefficients
        st.subheader("Top Feature Coefficients (Absolute Impact)")
        coef_series = results["top_coef"]
        coef_df = coef_series.reset_index()
        coef_df.columns = ["Feature", "Coefficient"]
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=coef_df, y="Feature", x="Coefficient",
                    palette=["#d1495b" if v > 0 else "#2d87bb" for v in coef_df["Coefficient"]])
        ax.set_title("Strongest Logistic Coefficients")
        ax.axvline(0, color="#444", linewidth=1)
        st.pyplot(fig)

        st.markdown("**Interpretation:** Positive coefficients increase default odds; negative reduce them. Delay frequency & magnitude dominate, while credit limit & age reduce risk.")

    except Exception as e:
        st.error(f"Evaluation error: {e}")
        import traceback
        import textwrap
        st.code(traceback.format_exc())

elif menu == "Make Prediction":
    st.subheader("Single Prediction")
    with st.form("prediction_form"):
        input_data = {}
        for col in feature_cols:
            # Numeric inputs (treat categorical as numeric codes as in training)
            default_val = float(df[col].median())
            input_data[col] = st.number_input(col, value=default_val)
        submit = st.form_submit_button("Predict")

    if submit:
        X = df[feature_cols]
        y = df["default.payment.next.month"]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(X_scaled, y)
        user_df = pd.DataFrame([input_data])[feature_cols]
        user_scaled = scaler.transform(user_df)
        prob = model.predict_proba(user_scaled)[0]
        st.success(
            f"Prediction: {'Default' if prob[1] > 0.5 else 'No Default'}")
        st.info(
            f"Probability of Default: {prob[1]*100:.2f}% | No Default: {prob[0]*100:.2f}%")

    st.subheader("Batch Prediction (Upload CSV)")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        missing_cols = set(feature_cols) - set(batch_df.columns)
        if missing_cols:
            st.error(f"Missing required columns: {sorted(missing_cols)}")
        else:
            X = df[feature_cols]
            y = df["default.payment.next.month"]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(max_iter=1000, class_weight="balanced")
            model.fit(X_scaled, y)
            batch_scaled = scaler.transform(batch_df[feature_cols])
            probs = model.predict_proba(batch_scaled)
            results = batch_df.copy()
            results["Prob_No_Default"] = probs[:, 0]
            results["Prob_Default"] = probs[:, 1]
            st.dataframe(results.head())
            st.download_button("Download Predictions",
                               results.to_csv(index=False),
                               "predictions.csv",
                               mime="text/csv")
