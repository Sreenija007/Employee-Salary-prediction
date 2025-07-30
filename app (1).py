
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitSalaryPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.usd_to_inr = 83.12
        self.is_trained = False

    @st.cache_data
    def load_and_preprocess_data(_self, uploaded_file):
        """Load and preprocess the salary data"""
        df = pd.read_csv(uploaded_file)
        df = df.dropna()
        df['Salary_INR'] = df['Salary'] * _self.usd_to_inr
        return df

    def feature_engineering(self, df):
        """Create features for the model"""
        df_processed = df.copy()

        categorical_columns = ['Gender', 'Education Level', 'Job Title']

        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le

        feature_columns = ['Age', 'Years of Experience', 'Gender_encoded', 
                          'Education Level_encoded', 'Job Title_encoded']

        X = df_processed[feature_columns]
        y = df_processed['Salary_INR']

        self.feature_names = feature_columns

        return X, y, df_processed

    def train_model(self, X, y):
        """Train the linear regression model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)

        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

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
        """Predict salary for new input"""
        if not self.is_trained:
            return None

        try:
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Education Level': [education],
                'Job Title': [job_title],
                'Years of Experience': [experience]
            })

            for col in ['Gender', 'Education Level', 'Job Title']:
                if col in self.label_encoders:
                    try:
                        input_data[f'{col}_encoded'] = self.label_encoders[col].transform(input_data[col])
                    except ValueError:
                        input_data[f'{col}_encoded'] = 0
                else:
                    input_data[f'{col}_encoded'] = 0

            X_input = input_data[self.feature_names]
            X_input_scaled = self.scaler.transform(X_input)

            prediction = self.model.predict(X_input_scaled)[0]
            return prediction

        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None

def create_plotly_visualizations(df, metrics=None):
    """Create interactive visualizations using Plotly"""

    # 1. Salary Distribution
    fig1 = px.histogram(df, x='Salary_INR', nbins=30, 
                       title='Distribution of Salaries (INR)',
                       labels={'Salary_INR': 'Salary (INR)', 'count': 'Frequency'})
    fig1.update_layout(showlegend=False)

    # 2. Age vs Salary
    fig2 = px.scatter(df, x='Age', y='Salary_INR', 
                     title='Age vs Salary',
                     labels={'Age': 'Age', 'Salary_INR': 'Salary (INR)'})

    # 3. Experience vs Salary
    fig3 = px.scatter(df, x='Years of Experience', y='Salary_INR',
                     title='Experience vs Salary',
                     labels={'Years of Experience': 'Years of Experience', 'Salary_INR': 'Salary (INR)'})

    # 4. Education Level vs Average Salary
    edu_salary = df.groupby('Education Level')['Salary_INR'].mean().reset_index()
    fig4 = px.bar(edu_salary, x='Education Level', y='Salary_INR',
                 title='Average Salary by Education Level',
                 labels={'Education Level': 'Education Level', 'Salary_INR': 'Average Salary (INR)'})

    # 5. Gender vs Average Salary
    gender_salary = df.groupby('Gender')['Salary_INR'].mean().reset_index()
    fig5 = px.bar(gender_salary, x='Gender', y='Salary_INR',
                 title='Average Salary by Gender',
                 labels={'Gender': 'Gender', 'Salary_INR': 'Average Salary (INR)'})

    return fig1, fig2, fig3, fig4, fig5

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Employee Salary Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Predict employee salaries using Machine Learning (Linear Regression)</p>', unsafe_allow_html=True)

    # Initialize the predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = StreamlitSalaryPredictor()

    # Sidebar
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📈 Data Analysis", "🔮 Salary Prediction", "📋 Model Performance"])

    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### Welcome to the Employee Salary Predictor! 🎯

            This application uses **Linear Regression** to predict employee salaries based on:
            - 👤 Age
            - 🎓 Education Level  
            - 💼 Job Title
            - ⏰ Years of Experience
            - 👨‍👩‍👧‍👦 Gender

            **How to use:**
            1. Upload your salary dataset (CSV format)
            2. Explore the data analysis
            3. Make salary predictions
            4. View model performance metrics

            **Features:**
            - 💹 Predictions in Indian Rupees (INR)
            - 📊 Interactive visualizations
            - 🎯 Model performance metrics
            - 🚀 Easy-to-use interface
            """)

        # File upload
        st.markdown('<h2 class="sub-header">📁 Upload Dataset</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = st.session_state.predictor.load_and_preprocess_data(uploaded_file)
                st.session_state.df = df

                st.success(f"✅ Dataset loaded successfully! ({len(df)} records)")

                # Train the model
                with st.spinner("🔄 Training the model..."):
                    X, y, df_processed = st.session_state.predictor.feature_engineering(df)
                    X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics = st.session_state.predictor.train_model(X, y)

                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.y_train_pred = y_train_pred
                    st.session_state.y_test_pred = y_test_pred
                    st.session_state.metrics = metrics

                st.success("🎯 Model trained successfully!")

                # Display basic info
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📊 Total Records", len(df))
                with col2:
                    st.metric("💰 Min Salary", f"₹{df['Salary_INR'].min():,.0f}")
                with col3:
                    st.metric("💎 Max Salary", f"₹{df['Salary_INR'].max():,.0f}")
                with col4:
                    st.metric("📈 Avg Salary", f"₹{df['Salary_INR'].mean():,.0f}")

            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

    elif page == "📈 Data Analysis":
        if 'df' not in st.session_state:
            st.warning("⚠️ Please upload a dataset first from the Home page.")
            return

        df = st.session_state.df

        st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)

        # Create visualizations
        fig1, fig2, fig3, fig4, fig5 = create_plotly_visualizations(df)

        # Display visualizations in a nice layout
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig3, use_container_width=True)
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)
            st.plotly_chart(fig4, use_container_width=True)

        # Data insights
        st.markdown('<h3 class="sub-header">🔍 Key Insights</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **📈 Salary Statistics:**
            - Average salary: ₹{:,.0f}
            - Median salary: ₹{:,.0f}
            - Standard deviation: ₹{:,.0f}
            """.format(df['Salary_INR'].mean(), df['Salary_INR'].median(), df['Salary_INR'].std()))

        with col2:
            st.markdown("""
            **👥 Demographics:**
            - Total employees: {:,}
            - Unique job titles: {:,}
            - Age range: {:.0f} - {:.0f} years
            """.format(len(df), df['Job Title'].nunique(), df['Age'].min(), df['Age'].max()))

    elif page == "🔮 Salary Prediction":
        if not hasattr(st.session_state, 'predictor') or not st.session_state.predictor.is_trained:
            st.warning("⚠️ Please upload and train the model first from the Home page.")
            return

        st.markdown('<h2 class="sub-header">🔮 Make Salary Predictions</h2>', unsafe_allow_html=True)

        df = st.session_state.df

        # Input form
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("👤 Age", min_value=18, max_value=65, value=30, step=1)
            gender = st.selectbox("👨‍👩‍👧‍👦 Gender", df['Gender'].unique())
            education = st.selectbox("🎓 Education Level", df['Education Level'].unique())

        with col2:
            experience = st.slider("⏰ Years of Experience", min_value=0, max_value=30, value=5, step=1)
            job_title = st.selectbox("💼 Job Title", sorted(df['Job Title'].unique()))

        # Prediction button
        if st.button("🎯 Predict Salary", type="primary"):
            prediction = st.session_state.predictor.predict_salary(age, gender, education, job_title, experience)

            if prediction:
                st.markdown(f"""
                <div class="prediction-result">
                    💰 Predicted Annual Salary: ₹{prediction:,.0f}
                </div>
                """, unsafe_allow_html=True)

                # Additional info
                monthly_salary = prediction / 12
                st.info(f"📅 Monthly Salary: ₹{monthly_salary:,.0f}")

                # Comparison with average
                avg_salary = df['Salary_INR'].mean()
                difference = prediction - avg_salary
                percentage_diff = (difference / avg_salary) * 100

                if difference > 0:
                    st.success(f"📈 This salary is ₹{difference:,.0f} ({percentage_diff:+.1f}%) above average")
                else:
                    st.warning(f"📉 This salary is ₹{abs(difference):,.0f} ({percentage_diff:+.1f}%) below average")

    elif page == "📋 Model Performance":
        if 'metrics' not in st.session_state:
            st.warning("⚠️ Please upload and train the model first from the Home page.")
            return

        st.markdown('<h2 class="sub-header">📊 Model Performance Metrics</h2>', unsafe_allow_html=True)

        metrics = st.session_state.metrics

        # Display metrics in cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🎯 R² Score (Accuracy)</h4>
                <p><strong>Training:</strong> {metrics['train_r2']:.4f}</p>
                <p><strong>Testing:</strong> {metrics['test_r2']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📏 RMSE (Root Mean Square Error)</h4>
                <p><strong>Training:</strong> ₹{metrics['train_rmse']:,.0f}</p>
                <p><strong>Testing:</strong> ₹{metrics['test_rmse']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📐 MAE (Mean Absolute Error)</h4>
                <p><strong>Training:</strong> ₹{metrics['train_mae']:,.0f}</p>
                <p><strong>Testing:</strong> ₹{metrics['test_mae']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        # Actual vs Predicted scatter plots
        col1, col2 = st.columns(2)

        with col1:
            fig_train = px.scatter(
                x=st.session_state.y_train, 
                y=st.session_state.y_train_pred,
                title=f'Training Set: Actual vs Predicted (R² = {metrics["train_r2"]:.3f})',
                labels={'x': 'Actual Salary (INR)', 'y': 'Predicted Salary (INR)'}
            )
            # Add perfect prediction line
            min_val = min(st.session_state.y_train.min(), st.session_state.y_train_pred.min())
            max_val = max(st.session_state.y_train.max(), st.session_state.y_train_pred.max())
            fig_train.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, 
                               line=dict(color="red", dash="dash"))
            st.plotly_chart(fig_train, use_container_width=True)

        with col2:
            fig_test = px.scatter(
                x=st.session_state.y_test, 
                y=st.session_state.y_test_pred,
                title=f'Test Set: Actual vs Predicted (R² = {metrics["test_r2"]:.3f})',
                labels={'x': 'Actual Salary (INR)', 'y': 'Predicted Salary (INR)'}
            )
            # Add perfect prediction line
            min_val = min(st.session_state.y_test.min(), st.session_state.y_test_pred.min())
            max_val = max(st.session_state.y_test.max(), st.session_state.y_test_pred.max())
            fig_test.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, 
                              line=dict(color="red", dash="dash"))
            st.plotly_chart(fig_test, use_container_width=True)

        # Model interpretation
        st.markdown('<h3 class="sub-header">🧠 Model Interpretation</h3>', unsafe_allow_html=True)

        r2_score = metrics['test_r2']
        if r2_score > 0.8:
            interpretation = "🎯 Excellent model performance! The model explains more than 80% of the variance in salaries."
        elif r2_score > 0.6:
            interpretation = "👍 Good model performance. The model explains a significant portion of salary variance."
        elif r2_score > 0.4:
            interpretation = "⚠️ Moderate model performance. There's room for improvement."
        else:
            interpretation = "❌ Poor model performance. Consider feature engineering or different algorithms."

        st.info(interpretation)

if __name__ == "__main__":
    main()
