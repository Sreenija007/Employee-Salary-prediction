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

class SalaryPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False

    @st.cache_data
    def load_and_preprocess_data(_self):
        # Always use the default CSV, ignore any uploaded file
        df = pd.read_csv("Salary-Data.csv")
        df = df.dropna()

        # Ensure 'Salary_INR' column exists and convert to numeric
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
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
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

# --------------------------- Page Layout/Interface --------------------------

if 'predictor' not in st.session_state:
    st.session_state.predictor = SalaryPredictor()
predictor = st.session_state.predictor

st.markdown("""
<h2 style="color:#003366;font-family:sans-serif;border-bottom:3px solid #0099CA;padding-bottom:.3em;">
Employee Salary Predictor
</h2>
<span style="color:#555;font-size:1.08em">
Predict employee salaries in <b>Indian Rupees (INR) per month</b> based on Age, Gender, Education Level, Job Title, and Experience using Machine Learning.
</span>
""", unsafe_allow_html=True)
st.divider()

tabs = st.tabs(["🏠 Home", "📈 Data Analysis", "🔮 Salary Prediction", "📋 Model Performance"])

# HOME TAB
with tabs[0]:
    st.write("""
    ###  Welcome!
    This app uses a **Random Forest** machine learning model trained on a cleaned built-in dataset to estimate monthly salaries.
    
    **Instructions:**
    - Provides comprehensive data visualizations including salary distribution, correlation with age and experience, and group-wise average salaries..
    - Explore data and predictions in the respective tabs.
    """)
    st.info("The app uses the built-in default dataset `Salary-Data.csv`.", icon="ℹ️")
    df = predictor.load_and_preprocess_data()
    st.session_state.df = df

# DATA ANALYSIS TAB
with tabs[1]:
    st.header("📈 Data Analysis (Monthly Salary)")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Unable to load default dataset.")
    else:
        st.subheader("Data Preview (First 20 rows)")
        st.dataframe(df[['Age', 'Gender', 'Years of Experience', 'Education Level', 'Job Title', 'Salary_INR']].head(20), use_container_width=True)
        st.markdown(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
        st.divider()
        st.subheader("Monthly Salary Distribution")
        fig_hist = px.histogram(df, x='Salary_INR', nbins=30, title='Monthly Salary Distribution (INR)')
        st.plotly_chart(fig_hist, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Monthly Salary by Education Level")
            edu_salary = df.groupby('Education Level')['Salary_INR'].mean().reset_index()
            fig_edu = px.bar(edu_salary, x='Education Level', y='Salary_INR', color='Education Level',
                             title="Monthly Salary by Education Level", text_auto='.2s')
            st.plotly_chart(fig_edu, use_container_width=True)
        with col2:
            st.subheader("Average Monthly Salary by Gender")
            gender_salary = df.groupby('Gender')['Salary_INR'].mean().reset_index()
            fig_gender = px.bar(gender_salary, x='Gender', y='Salary_INR', color='Gender',
                                title="Monthly Salary by Gender", text_auto='.2s')
            st.plotly_chart(fig_gender, use_container_width=True)
        st.divider()
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Age vs Monthly Salary")
            fig_age = px.scatter(df, x='Age', y='Salary_INR', color='Gender', symbol='Education Level',
                                 title="Age vs Monthly Salary")
            st.plotly_chart(fig_age, use_container_width=True)
        with col4:
            st.subheader("Experience vs Monthly Salary")
            fig_exp = px.scatter(df, x='Years of Experience', y='Salary_INR', color='Education Level', symbol='Gender',
                                 title="Experience vs Monthly Salary")
            st.plotly_chart(fig_exp, use_container_width=True)

# SALARY PREDICTION TAB
with tabs[2]:
    st.header(" Salary Prediction (Monthly)")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Dataset not loaded. Please check the Home tab.")
    else:
        with st.form(key="predict_form"):
            cols = st.columns(2)
            age = cols[0].number_input("Age", 18, 65, 28)
            max_exp = max(0, age - 15)  # Assuming people can't start careers before 15yo
            experience = cols[1].number_input("Years of Experience", min_value=0.0, max_value=float(max_exp), 
            value=min(2.0, float(max_exp)), 
            step=0.5, 
            format="%.1f"
            )

            gender = cols[0].selectbox("Gender", sorted(df['Gender'].unique()))
            education = cols[1].selectbox("Education Level", sorted(df['Education Level'].unique()))
            job_title = st.selectbox("Job Title", sorted(df['Job Title'].unique()))
            submit = st.form_submit_button("Predict Monthly Salary 💸")

        if submit:
            X, y, _ = predictor.feature_engineering(df)
            predictor.train_model(X, y)
            pred_monthly = predictor.predict_salary(age, gender, education, job_title, experience)
            if pred_monthly is not None:
                st.success(f"**Estimated Monthly Salary: ₹ {pred_monthly:,.0f} INR**", icon="💸")
                st.caption("Prediction is based on current monthly income patterns in the dataset, using Random Forest regression.")

# MODEL PERFORMANCE TAB
with tabs[3]:
    st.header("📋 Model Performance Metrics (Monthly Salary)")
    df = st.session_state.get('df', None)
    if df is None:
        st.warning("Dataset not loaded. Please check the Home tab.")
    else:
        X, y, _ = predictor.feature_engineering(df)
        X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics = predictor.train_model(X, y)
        st.markdown(f"""
        <div style="background:#F2F2F2;padding:1em 2em;border-radius:10px;border:1px solid #90caf9;">
            <h4 style="color:#195b89;">R² Score</h4>
            <ul>
                <li><b>Training:</b> {metrics['train_r2']:.3f}</li>
                <li><b>Testing:</b> {metrics['test_r2']:.3f}</li>
            </ul>
            <h4 style="color:#195b89;">Root Mean Squared Error (RMSE)</h4>
            <ul>
                <li><b>Training:</b> ₹{metrics['train_rmse']:,.0f} / month</li>
                <li><b>Testing:</b> ₹{metrics['test_rmse']:,.0f} / month</li>
            </ul>
            <h4 style="color:#195b89;">Mean Absolute Error (MAE)</h4>
            <ul>
                <li><b>Training:</b> ₹{metrics['train_mae']:,.0f} / month</li>
                <li><b>Testing:</b> ₹{metrics['test_mae']:,.0f} / month</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.subheader("Prediction vs Actual (Test Data, Monthly)")
        perf_df = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred})
        fig_perf = px.scatter(perf_df, x="Actual", y="Predicted", trendline="ols",
                              title="Actual vs Predicted Monthly Salary (Test Set)",
                              labels={'Actual': "Actual (₹/month)", 'Predicted': "Predicted (₹/month)"})
        fig_perf.update_layout(height=420)
        st.plotly_chart(fig_perf, use_container_width=True)
