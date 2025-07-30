**Employee Salary Predictor**

This project is a Machine Learning application designed to predict employee monthly salaries in Indian Rupees (INR) based on features like age, gender, years of experience, education level, and job title.

**Features**

- Uses a Random Forest Regressor model for accurate salary prediction.

- Processes a clean, realistic dataset of 1000+ employee records with monthly salary data.

- Includes feature engineering to encode categorical variables.

- Provides comprehensive data visualizations including salary distribution, correlation with age and experience, and group-wise average      salaries.

- Interactive web app built with Streamlit for easy use.

- Fully deployed on Streamlit Cloud for worldwide access.

- No dataset upload needed; app uses a built-in clean dataset by default.

**Access the deployed app at:**

https://

**Project Structure**

app.py - Streamlit app code for frontend, data visualization, and user input handling.

salary_predictor.py - Core Machine Learning pipeline with model training and prediction.

Salary-Data.csv - Clean, synthetic dataset with realistic employee salary data.

requirements.txt - Python dependencies for easy environment setup.

Results
High prediction accuracy with R² around 0.9.

Error metrics: RMSE and MAE are reported in INR per month.

Visual insights into salary patterns by demographics and experience.
