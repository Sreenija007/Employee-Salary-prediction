
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.usd_to_inr = 83.12  # Current conversion rate (approximate)

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the salary data"""
        # Load data
        df = pd.read_csv(file_path)

        # Remove rows with missing values
        df = df.dropna()

        # Convert USD salaries to INR
        df['Salary_INR'] = df['Salary'] * self.usd_to_inr

        print(f"Dataset loaded successfully with {len(df)} records")
        print(f"Salary range in INR: ₹{df['Salary_INR'].min():,.0f} - ₹{df['Salary_INR'].max():,.0f}")

        return df

    def feature_engineering(self, df):
        """Create features for the model"""
        # Create a copy for processing
        df_processed = df.copy()

        # Encode categorical variables
        categorical_columns = ['Gender', 'Education Level', 'Job Title']

        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le

        # Select features for training
        feature_columns = ['Age', 'Years of Experience', 'Gender_encoded', 
                          'Education Level_encoded', 'Job Title_encoded']

        X = df_processed[feature_columns]
        y = df_processed['Salary_INR']

        self.feature_names = feature_columns

        return X, y, df_processed

    def train_model(self, X, y):
        """Train the linear regression model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train the model
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        metrics = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae
        }

        return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics

    def create_visualizations(self, df, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics):
        """Create comprehensive visualizations"""

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Salary Distribution
        plt.subplot(3, 3, 1)
        plt.hist(df['Salary_INR'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Salaries (INR)', fontsize=14, fontweight='bold')
        plt.xlabel('Salary (INR)')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)

        # 2. Age vs Salary
        plt.subplot(3, 3, 2)
        plt.scatter(df['Age'], df['Salary_INR'], alpha=0.6, color='coral')
        plt.title('Age vs Salary', fontsize=14, fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Salary (INR)')
        plt.grid(alpha=0.3)

        # 3. Experience vs Salary
        plt.subplot(3, 3, 3)
        plt.scatter(df['Years of Experience'], df['Salary_INR'], alpha=0.6, color='lightgreen')
        plt.title('Experience vs Salary', fontsize=14, fontweight='bold')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary (INR)')
        plt.grid(alpha=0.3)

        # 4. Education Level vs Average Salary
        plt.subplot(3, 3, 4)
        edu_salary = df.groupby('Education Level')['Salary_INR'].mean().sort_values()
        bars = plt.bar(edu_salary.index, edu_salary.values, color=['#FF9999', '#66B2FF', '#99FF99'])
        plt.title('Average Salary by Education Level', fontsize=14, fontweight='bold')
        plt.xlabel('Education Level')
        plt.ylabel('Average Salary (INR)')
        plt.xticks(rotation=45)
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}', ha='center', va='bottom')
        plt.grid(alpha=0.3)

        # 5. Gender vs Average Salary
        plt.subplot(3, 3, 5)
        gender_salary = df.groupby('Gender')['Salary_INR'].mean()
        bars = plt.bar(gender_salary.index, gender_salary.values, color=['#FFB6C1', '#87CEEB'])
        plt.title('Average Salary by Gender', fontsize=14, fontweight='bold')
        plt.xlabel('Gender')
        plt.ylabel('Average Salary (INR)')
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'₹{height:,.0f}', ha='center', va='bottom')
        plt.grid(alpha=0.3)

        # 6. Actual vs Predicted (Training Set)
        plt.subplot(3, 3, 6)
        plt.scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='Training')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        plt.title(f'Training Set: Actual vs Predicted\nR² = {metrics["train_r2"]:.3f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Actual Salary (INR)')
        plt.ylabel('Predicted Salary (INR)')
        plt.grid(alpha=0.3)

        # 7. Actual vs Predicted (Test Set)
        plt.subplot(3, 3, 7)
        plt.scatter(y_test, y_test_pred, alpha=0.6, color='red', label='Testing')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.title(f'Test Set: Actual vs Predicted\nR² = {metrics["test_r2"]:.3f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Actual Salary (INR)')
        plt.ylabel('Predicted Salary (INR)')
        plt.grid(alpha=0.3)

        # 8. Feature Importance (Coefficients)
        plt.subplot(3, 3, 8)
        feature_importance = abs(self.model.coef_)
        feature_names_clean = [name.replace('_encoded', '').replace('_', ' ') for name in self.feature_names]
        bars = plt.bar(range(len(feature_importance)), feature_importance, color='purple', alpha=0.7)
        plt.title('Feature Importance (|Coefficients|)', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Absolute Coefficient Value')
        plt.xticks(range(len(feature_names_clean)), feature_names_clean, rotation=45)
        plt.grid(alpha=0.3)

        # 9. Model Performance Metrics
        plt.subplot(3, 3, 9)
        metrics_names = ['R² Score', 'RMSE (lakhs)', 'MAE (lakhs)']
        train_values = [metrics['train_r2'], metrics['train_rmse']/100000, metrics['train_mae']/100000]
        test_values = [metrics['test_r2'], metrics['test_rmse']/100000, metrics['test_mae']/100000]

        x = np.arange(len(metrics_names))
        width = 0.35

        plt.bar(x - width/2, train_values, width, label='Training', color='lightblue', alpha=0.8)
        plt.bar(x + width/2, test_values, width, label='Testing', color='lightcoral', alpha=0.8)

        plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.xticks(x, metrics_names)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('salary_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print detailed metrics
        print("\n" + "="*50)
        print("MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"Training R² Score: {metrics['train_r2']:.4f}")
        print(f"Testing R² Score: {metrics['test_r2']:.4f}")
        print(f"Training RMSE: ₹{metrics['train_rmse']:,.0f}")
        print(f"Testing RMSE: ₹{metrics['test_rmse']:,.0f}")
        print(f"Training MAE: ₹{metrics['train_mae']:,.0f}")
        print(f"Testing MAE: ₹{metrics['test_mae']:,.0f}")

    def predict_salary(self, age, gender, education, job_title, experience):
        """Predict salary for new input"""
        try:
            # Create input dataframe
            input_data = pd.DataFrame({
                'Age': [age],
                'Gender': [gender],
                'Education Level': [education],
                'Job Title': [job_title],
                'Years of Experience': [experience]
            })

            # Encode categorical variables
            for col in ['Gender', 'Education Level', 'Job Title']:
                if col in self.label_encoders:
                    try:
                        input_data[f'{col}_encoded'] = self.label_encoders[col].transform(input_data[col])
                    except ValueError:
                        # Handle unseen categories
                        print(f"Warning: Unknown {col} '{input_data[col].iloc[0]}'. Using most common value.")
                        input_data[f'{col}_encoded'] = 0
                else:
                    input_data[f'{col}_encoded'] = 0

            # Prepare features
            X_input = input_data[self.feature_names]
            X_input_scaled = self.scaler.transform(X_input)

            # Make prediction
            prediction = self.model.predict(X_input_scaled)[0]

            return prediction

        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

def main():
    """Main function to run the salary prediction pipeline"""
    print("="*60)
    print("EMPLOYEE SALARY PREDICTION USING LINEAR REGRESSION")
    print("="*60)

    # Initialize the predictor
    predictor = SalaryPredictor()

    # Load and preprocess data
    df = predictor.load_and_preprocess_data('Salary-Data.csv')

    # Feature engineering
    X, y, df_processed = predictor.feature_engineering(df)

    # Train model
    X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics = predictor.train_model(X, y)

    # Create visualizations
    predictor.create_visualizations(df, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, metrics)

    # Example prediction
    print("\n" + "="*50)
    print("EXAMPLE SALARY PREDICTION")
    print("="*50)

    example_prediction = predictor.predict_salary(
        age=30,
        gender='Male',
        education="Bachelor's",
        job_title='Software Engineer',
        experience=5
    )

    if example_prediction:
        print(f"Predicted salary for 30-year-old Male Software Engineer with Bachelor's degree and 5 years experience:")
        print(f"₹{example_prediction:,.0f} per year")

    return predictor

if __name__ == "__main__":
    predictor = main()
