# 💰 Employee Salary Predictor

A machine learning web application that predicts employee salaries using Linear Regression. Built with Python, Streamlit, and scikit-learn.

## 🎯 Project Overview

This project develops a machine learning model that predicts employee salaries based on factors like:
- 👤 Age
- 🎓 Education Level  
- 💼 Job Title
- ⏰ Years of Experience
- 👨‍👩‍👧‍👦 Gender

The model uses **Linear Regression** and provides predictions in **Indian Rupees (INR)**.

## 🚀 Features

- **Interactive Web Interface**: Built with Streamlit
- **Real-time Predictions**: Instant salary predictions
- **Data Visualization**: Comprehensive charts and graphs
- **Model Performance Metrics**: R², RMSE, MAE
- **Indian Currency**: All predictions in INR
- **Data Upload**: Upload your own CSV datasets
- **Responsive Design**: Works on desktop and mobile

## 📁 Project Structure

```
employee-salary-predictor/
│
├── app.py                  # Streamlit web application
├── salary_predictor.py     # Core ML pipeline
├── Salary-Data.csv         # Sample dataset
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore file
```

## 🛠️ Installation & Setup

### Method 1: Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/employee-salary-predictor.git
   cd employee-salary-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv

   # On Windows
   venv\Scripts\activate

   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and go to `http://localhost:8501`

### Method 2: Direct Run (if you have Python installed)

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn plotly
streamlit run app.py
```

## 🌐 Deployment on Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create a GitHub repository**
   - Go to [GitHub](https://github.com)
   - Click "New Repository"
   - Name it `employee-salary-predictor`
   - Make it public
   - Initialize with README

2. **Upload your files to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Employee Salary Predictor"
   git branch -M main
   git remote add origin https://github.com/yourusername/employee-salary-predictor.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to [Streamlit Cloud](https://streamlit.io/cloud)**

2. **Sign in with GitHub**

3. **Create a new app**
   - Click "New app"
   - Choose your repository: `yourusername/employee-salary-predictor`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: Choose a custom URL (optional)

4. **Deploy**
   - Click "Deploy!"
   - Wait for deployment (usually 2-5 minutes)
   - Your app will be live at: `https://your-app-name.streamlit.app`

### Step 3: Configure Secrets (if needed)

If you need to store sensitive information:
1. Go to your app settings
2. Click on "Secrets"
3. Add your secrets in TOML format

## 📊 Using the Application

### 1. Home Page
- Upload your CSV dataset
- View basic statistics
- Model automatically trains on your data

### 2. Data Analysis
- Explore salary distributions
- View correlations between variables
- Interactive charts and visualizations

### 3. Salary Prediction
- Enter employee details:
  - Age (18-65)
  - Gender
  - Education Level
  - Job Title
  - Years of Experience (0-30)
- Get instant salary prediction in INR
- See comparison with average salary

### 4. Model Performance
- View R² score, RMSE, and MAE
- Actual vs Predicted scatter plots
- Model interpretation insights

## 🗃️ Dataset Format

Your CSV file should have these columns:
```csv
Age,Gender,Education Level,Job Title,Years of Experience,Salary
32,Male,Bachelor's,Software Engineer,5,90000
28,Female,Master's,Data Analyst,3,65000
...
```

**Required Columns:**
- `Age`: Numeric (18-65)
- `Gender`: Text (Male/Female)
- `Education Level`: Text (Bachelor's/Master's/PhD)
- `Job Title`: Text (any job title)
- `Years of Experience`: Numeric (0-30)
- `Salary`: Numeric (in USD, will be converted to INR)

## 🧠 Machine Learning Model

### Algorithm: Linear Regression
- **Simple and interpretable**
- **Fast training and prediction**
- **Good baseline for salary prediction**

### Features Used:
1. **Age**: Employee age
2. **Years of Experience**: Work experience
3. **Gender**: Encoded (Male=1, Female=0)
4. **Education Level**: Encoded (Bachelor's=0, Master's=1, PhD=2)
5. **Job Title**: Label encoded

### Model Pipeline:
1. **Data Preprocessing**: Handle missing values, convert USD to INR
2. **Feature Engineering**: Encode categorical variables
3. **Feature Scaling**: Standardize numerical features
4. **Model Training**: Train Linear Regression
5. **Evaluation**: Calculate R², RMSE, MAE
6. **Prediction**: Make salary predictions

## 📈 Model Performance Metrics

- **R² Score**: Measures model accuracy (0-1, higher is better)
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)

## 🎨 Visualizations

The app includes:
- **Salary Distribution**: Histogram of salary ranges
- **Age vs Salary**: Scatter plot showing age correlation
- **Experience vs Salary**: Experience impact on salary
- **Education Level Analysis**: Average salary by education
- **Gender Analysis**: Salary comparison by gender
- **Actual vs Predicted**: Model performance visualization
- **Feature Importance**: Which factors matter most

## 🔧 Customization

### Adding New Features:
1. Update the dataset with new columns
2. Modify the `feature_engineering()` function
3. Update the Streamlit interface
4. Retrain the model

### Changing Currency:
- Update `usd_to_inr` rate in the code
- Modify display formatting

### Different ML Models:
Replace `LinearRegression()` with:
- `RandomForestRegressor()`
- `GradientBoostingRegressor()`
- `XGBRegressor()`

## 🐛 Troubleshooting

### Common Issues:

1. **Import Error**
   ```bash
   pip install --upgrade streamlit pandas numpy scikit-learn
   ```

2. **Dataset Upload Issues**
   - Ensure CSV has required columns
   - Check for missing values
   - Verify data types

3. **Prediction Errors**
   - Make sure model is trained first
   - Check input data format
   - Verify all required fields

4. **Deployment Issues**
   - Check requirements.txt
   - Ensure all files are in repository
   - Verify Python version compatibility

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset source: Sample employee salary data
- Built with [Streamlit](https://streamlit.io/)
- Machine Learning with [scikit-learn](https://scikit-learn.org/)
- Visualizations with [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)

---

⭐ **Star this repository if you found it helpful!**

🐛 **Found a bug?** [Open an issue](https://github.com/yourusername/employee-salary-predictor/issues)

💡 **Have suggestions?** [Create a feature request](https://github.com/yourusername/employee-salary-predictor/issues)
