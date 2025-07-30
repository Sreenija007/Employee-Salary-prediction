import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide"
)

# Modern dark-mode tweaks for stat cards, headings, and container padding
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { color: #13c2c2 !important; font-weight: bold;}
    .stTabs [data-baseweb="tab-list"] {background: #222c36;}
    hr {border-top: 1px solid #434445;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- Welcome Panel ------------------------
with st.container():
    st.markdown(
        "<div style='background-color:#222c36;border-radius:14px;padding:28px 32px 12px 32px;"
        "margin-bottom:24px;border:2px solid #13c2c2;box-shadow:0 4px 16px #085c5c45;'>"
        "<h2 style='color:#13c2c2;margin-top:0;'>Employee Salary Predictor</h2>"
        "<p style='font-size:1.08rem;color:#e1f0ff;margin-bottom:.3em;'>"
        "Predict employee salaries in Indian Rupees (INR) per month based on Age, Gender, Education Level, Job Title, and Experience using Machine Learning.
.<br>"
        
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )
    st.info("The app uses the built-in default dataset **Salary-Data.csv**.", icon="ℹ️")

# ---------------------- Helper Classes & Functions ------------------------
class SalaryPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False

    @st.cache_data(show_spinner="Loading & Processing Data...")
    def load_and_preprocess_data(_self):
        df = pd.read_csv("Salary-Data.csv")
        df = df.dropna()
        if 'Salary_INR' not in df.columns:
            st.error("Dataset must contain a 'Salary_INR' column with monthly salary in INR.")
            return None
        df['Salary_INR'] = pd.to_numeric(df['Salary_INR'], errors='coerce')
        df = df.dropna(subset=['Salary_INR'])
        return df

    def feature_engineering(self, df):
        df_processed = df.copy()
        categorical_cols = ['Gender', 'Education Level', 'Job Title']
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
        self.feature_names = [
            'Age', 'Years of Experience',
            'Gender_encoded', 'Education Level_encoded', 'Job Title_encoded'
        ]
        X = df_processed[self.feature_names]
        y = df_processed['Salary_INR']
        return X, y, df_processed

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred)
        }
        self.is_trained = True
        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics

    def predict_salary(self, age, gender, education, job_title, experience):
        if not self.is_trained:
            return None
        input_df = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Education Level': [education],
            'Job Title': [job_title],
            'Years of Experience': [experience]
        })
        for col in ['Gender', 'Education Level', 'Job Title']:
            if col in self.label_encoders:
                try:
                    input_df[f'{col}_encoded'] = self.label_encoders[col].transform(input_df[col])
                except ValueError:
                    input_df[f'{col}_encoded'] = 0
            else:
                input_df[f'{col}_encoded'] = 0
        X_input = input_df[self.feature_names]
        pred = self.model.predict(X_input)[0]
        return pred

# --------------------------- Main App Layout --------------------------

if 'predictor' not in st.session_state:
    st.session_state.predictor = SalaryPredictor()
predictor = st.session_state.predictor

tabs = st.tabs(["🏠 Overview", "📊 Analysis", "🔢 Predict", "📈 Performance"])

# ----- OVERVIEW TAB -----
with tabs[0]:
    st.write("## How to Use")
    st.markdown(
        """
        1. **Explore Data**: See salary analytics for all employees.
        2. **Make a Prediction**: Enter details and get a monthly salary estimate.
        3. **View Model Metrics**: Check model accuracy and diagnostics.
        """
    )
    df = predictor.load_and_preprocess_data()
    st.session_state.df = df

# ----- ANALYSIS TAB -----
with tabs[1]:
    st.header("📊 Data Analysis (Monthly Salary)")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Unable to load default dataset.")
    else:
        st.caption("First 20 Employees")
        st.dataframe(
            df[['Age', 'Gender', 'Years of Experience', 'Education Level', 'Job Title', 'Salary_INR']].head(20),
            use_container_width=True
        )
        st.markdown(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
        st.divider()
        st.subheader("Monthly Salary Distribution")
        fig_hist = px.histogram(
            df, x='Salary_INR', nbins=30, 
            title='Monthly Salary Distribution (INR)', 
            height=350, template='plotly_dark',
            color_discrete_sequence=['#13c2c2']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("Average Salary by Education Level")
            edu_salary = df.groupby('Education Level')['Salary_INR'].mean().reset_index()
            fig_edu = px.bar(
                edu_salary, x='Education Level', y='Salary_INR',
                color='Education Level', color_discrete_sequence=px.colors.qualitative.Safe,
                title="Monthly Salary by Education", text_auto='.2s', height=325, template='plotly_dark'
            )
            st.plotly_chart(fig_edu, use_container_width=True)
        with col2:
            st.subheader("Average Salary by Gender")
            gender_salary = df.groupby('Gender')['Salary_INR'].mean().reset_index()
            fig_gender = px.bar(
                gender_salary, x='Gender', y='Salary_INR',
                color='Gender', color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Monthly Salary by Gender", text_auto='.2s', height=325, template='plotly_dark'
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        st.divider()
        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.subheader("Age vs Monthly Salary")
            fig_age = px.scatter(
                df, x='Age', y='Salary_INR', color='Gender',
                symbol='Education Level', template='plotly_dark',
                title="Age vs Monthly Salary", color_discrete_sequence=['#13c2c2','#e1bfff']
            )
            st.plotly_chart(fig_age, use_container_width=True)
        with col4:
            st.subheader("Experience vs Monthly Salary")
            fig_exp = px.scatter(
                df, x='Years of Experience', y='Salary_INR', color='Education Level', symbol='Gender',
                template='plotly_dark', title="Experience vs Monthly Salary"
            )
            st.plotly_chart(fig_exp, use_container_width=True)

# ----- PREDICTION TAB -----
with tabs[2]:
    st.header("🔢 Salary Prediction (Monthly)")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Dataset not loaded. Please check the Overview tab.")
    else:
        st.markdown(
            "<div style='background-color:#23232a;border-radius:10px;padding:22px 24px 3px 24px;"
            "margin-bottom:12px;'>", 
            unsafe_allow_html=True
        )
        with st.form(key="predict_form"):
            cols = st.columns(2)
            age = cols[0].number_input("Age", 18, 65, 28)
            max_exp = max(0, age - 15)
            experience = cols[1].number_input(
                "Years of Experience",
                min_value=0.0,
                max_value=float(max_exp),
                value=min(2.0, float(max_exp)),
                step=0.5,
                format="%.1f"
            )
            cols[0].caption("Experience can't exceed (Age - 15)")
            gender = cols[0].selectbox("Gender", sorted(df['Gender'].unique()))
            education = cols[1].selectbox("Education Level", sorted(df['Education Level'].unique()))
            job_title = st.selectbox("Job Title", sorted(df['Job Title'].unique()))
            submit = st.form_submit_button("Predict 💸")

        st.markdown("</div>", unsafe_allow_html=True)

        if submit:
            X, y, _ = predictor.feature_engineering(df)
            predictor.train_model(X, y)
            pred_monthly = predictor.predict_salary(age, gender, education, job_title, experience)
            if pred_monthly is not None:
                st.success(f"**Estimated Monthly Salary: ₹ {pred_monthly:,.0f} INR**", icon="💸")
                st.caption("Prediction is based on the current dataset, using Random Forest regression.")

# ----- PERFORMANCE TAB -----
with tabs[3]:
    st.header("📈 Model Performance Metrics (Monthly Salary)")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Dataset not loaded. Please check the Overview tab.")
    else:
        X, y, _ = predictor.feature_engineering(df)
        X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics = predictor.train_model(X, y)
        # Modern stat card metrics
        metric_cols = st.columns(3)
        metric_cols[0].metric("R² (Test)", f"{metrics['test_r2']:.2f}")
        metric_cols[1].metric("RMSE (₹/mo)", f"{metrics['test_rmse']:,.0f}")
        metric_cols[2].metric("MAE (₹/mo)", f"{metrics['test_mae']:,.0f}")
        st.divider()

        st.subheader("Actual vs Predicted (Test Set, Monthly)")
        perf_df = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred})
        fig_perf = px.scatter(
            perf_df, x="Actual", y="Predicted", trendline="ols",
            title="Actual vs Predicted Monthly Salary (Test Set)",
            labels={'Actual': "Actual (₹/month)", 'Predicted': "Predicted (₹/month)"},
            template='plotly_dark', color_discrete_sequence=['#13c2c2']
        )
        fig_perf.update_layout(height=420)
        st.plotly_chart(fig_perf, use_container_width=True)
