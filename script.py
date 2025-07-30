import pandas as pd
import numpy as np

# Simulating an improved dataset based on typical salary predictor features

np.random.seed(42)

size = 1000  # sample size

# Age between 20 and 60
ages = np.random.randint(20, 60, size)

# Gender categories
genders = np.random.choice(['Male', 'Female', 'Other'], size=size, p=[0.6, 0.35, 0.05])

# Years of Experience roughly correlates with age but not exactly
experience = np.clip(ages - np.random.randint(20, 25, size), 0, None) + np.random.normal(0, 2, size).astype(int)
experience = np.clip(experience, 0, 40)  # clamp experience

# Education Level categories
education_levels = np.random.choice(
    ["Bachelor's", "Master's", "PhD", "Diploma"],
    size=size, p=[0.5, 0.3, 0.1, 0.1]
)

# Job Title categories grouped
job_titles = np.random.choice(
    ['Web Developer', 'Data Scientist', 'Manager', 'Analyst', 'HR', 'Sales Executive', 'Software Engineer'],
    size=size,
    p=[0.25, 0.15, 0.15, 0.1, 0.1, 0.1, 0.15]
)

# Base salaries depending on job titles (in INR * 10^3 for realism)
base_salaries = {
    'Web Developer': 450,  # 450,000 INR
    'Software Engineer': 600,  # 600,000 INR
    'Data Scientist': 900,  # 900,000 INR
    'Manager': 1200,  # 1,200,000 INR
    'Analyst': 500,    # 500,000 INR
    'HR': 400,         # 400,000 INR
    'Sales Executive': 450  # 450,000 INR
}

# Salary influenced by experience, education, gender
sal_inr = []

for i in range(size):
    base = base_salaries[job_titles[i]]
    # Experience weight: 3% increase per year of experience
    exp_factor = 1 + 0.03 * experience[i]
    # Education weight
    edu_factor = 1.0
    if education_levels[i] == "Master's":
        edu_factor = 1.2
    elif education_levels[i] == "PhD":
        edu_factor = 1.4
    elif education_levels[i] == "Diploma":
        edu_factor = 0.85

    # Gender weight (assuming slight bias)
    gender_factor = 1.0
    if genders[i] == 'Male':
        gender_factor = 1.05
    elif genders[i] == 'Other':
        gender_factor = 0.95

    # Random noise factor (±10%)
    noise = np.random.uniform(0.9, 1.1)

    salary = base * exp_factor * edu_factor * gender_factor * noise
    sal_inr.append(salary * 1000)  # final INR

# Assemble into a dataframe
new_df = pd.DataFrame({
    'Age': ages,
    'Gender': genders,
    'Years of Experience': experience,
    'Education Level': education_levels,
    'Job Title': job_titles,
    'Salary_INR': sal_inr
})

# Clip extreme values at 99th percentile to reduce outliers
cap_value = new_df['Salary_INR'].quantile(0.99)
new_df['Salary_INR'] = new_df['Salary_INR'].clip(upper=cap_value)

new_df.head(10)  # preview
