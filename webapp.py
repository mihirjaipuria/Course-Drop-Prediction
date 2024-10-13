import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

# Set seed for reproducibility
np.random.seed(42)

# Function to generate synthetic data
def generate_data(n_students=10000):
    student_id = np.arange(1, n_students + 1)
    age = np.random.randint(18, 25, size=n_students)
    gender = np.random.choice(['Male', 'Female'], size=n_students)
    majors = ['Computer Science', 'Mechanical Engineering', 'Environmental Science', 
              'Civil Engineering', 'Electrical Engineering']
    major = np.random.choice(majors, size=n_students)
    year = np.random.randint(1, 5, size=n_students)
    regions = ['West Bengal', 'Delhi', 'Karnataka', 'Maharashtra', 'Tamil Nadu']
    region = np.random.choice(regions, size=n_students)
    
    logins_per_week = np.random.randint(1, 10, size=n_students)
    videos_watched = np.random.randint(1, 20, size=n_students)
    time_spent_on_platform = np.random.randint(1, 15, size=n_students)
    avg_quiz_score = np.random.randint(0, 100, size=n_students)
    
    courses_completed = np.random.randint(0, 6, size=n_students)
    courses_started = np.random.randint(2, 7, size=n_students)
    avg_score_across_courses = np.random.randint(0, 100, size=n_students)
    
    data = pd.DataFrame({
        'student_id': student_id,
        'age': age,
        'gender': gender,
        'major': major,
        'year': year,
        'region': region,
        'logins_per_week': logins_per_week,
        'videos_watched': videos_watched,
        'time_spent_on_platform': time_spent_on_platform,
        'avg_quiz_score': avg_quiz_score,
        'courses_completed': courses_completed,
        'courses_started': courses_started,
        'avg_score_across_courses': avg_score_across_courses
    })

    data['completion_status'] = np.random.choice([0, 1], size=n_students, p=[0.3, 0.7])

    return data

# Function to preprocess data
def preprocess_data(data, is_full_dataset=True):
    # Feature Engineering
    data['engagement_score'] = (data['logins_per_week'] * 0.5 + 
                                data['videos_watched'] * 0.3 + 
                                data['time_spent_on_platform'] * 0.2)
    data['completion_ratio'] = data['courses_completed'] / data['courses_started']
    data['completion_ratio'].replace([np.inf, -np.inf], 0, inplace=True)
    data['login_quiz_interaction'] = data['logins_per_week'] * data['avg_quiz_score']
    data['video_completion_ratio'] = data['videos_watched'] / (data['courses_completed'] + 1)
    data['high_engagement'] = (data['engagement_score'] > 5).astype(int)

    if is_full_dataset:
        X = data.drop(columns=['student_id', 'completion_status'])
        y = data['completion_status']
    else:
        X = data
        y = None

    X = pd.get_dummies(X, drop_first=True)

    return X, y

# Function to train models
def train_models(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)

    rf_model.fit(X_train_resampled, y_train_resampled)
    gb_model.fit(X_train_resampled, y_train_resampled)

    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', gb_model)
        ], voting='soft'
    )
    voting_clf.fit(X_train_resampled, y_train_resampled)

    return rf_model, gb_model, voting_clf

# Streamlit App
st.set_page_config(page_title="Course Completion Predictor", layout="wide")
st.title("Course Completion Predictor")

# Generate data
if 'data' not in st.session_state:
    st.session_state.data = generate_data()

# Preprocess data
X, y = preprocess_data(st.session_state.data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train models
if 'models' not in st.session_state:
    st.session_state.models = train_models(X_train, y_train)

rf_model, gb_model, voting_clf = st.session_state.models

# Sidebar for user input
st.sidebar.header("Student Information")
age = st.sidebar.slider("Age", 18, 25, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
major = st.sidebar.selectbox("Major", ["Computer Science", "Mechanical Engineering", "Environmental Science", "Civil Engineering", "Electrical Engineering"])
year = st.sidebar.slider("Year", 1, 5, 2)
region = st.sidebar.selectbox("Region", ["West Bengal", "Delhi", "Karnataka", "Maharashtra", "Tamil Nadu"])
logins_per_week = st.sidebar.slider("Logins per Week", 1, 10, 5)
videos_watched = st.sidebar.slider("Videos Watched", 1, 20, 10)
time_spent_on_platform = st.sidebar.slider("Time Spent on Platform (hours)", 1, 15, 7)
avg_quiz_score = st.sidebar.slider("Average Quiz Score", 0, 100, 75)
courses_completed = st.sidebar.slider("Courses Completed", 0, 5, 2)
courses_started = st.sidebar.slider("Courses Started", 2, 7, 4)
avg_score_across_courses = st.sidebar.slider("Average Score Across Courses", 0, 100, 80)

# Create a DataFrame for the user input
user_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'major': [major],
    'year': [year],
    'region': [region],
    'logins_per_week': [logins_per_week],
    'videos_watched': [videos_watched],
    'time_spent_on_platform': [time_spent_on_platform],
    'avg_quiz_score': [avg_quiz_score],
    'courses_completed': [courses_completed],
    'courses_started': [courses_started],
    'avg_score_across_courses': [avg_score_across_courses]
})

# Preprocess user data
user_X, _ = preprocess_data(user_data, is_full_dataset=False)
user_X = user_X.reindex(columns=X.columns, fill_value=0)

# Make predictions
predictions = {
    "Random Forest": rf_model.predict_proba(user_X)[0][1],
    "Gradient Boosting": gb_model.predict_proba(user_X)[0][1],
    "Voting Classifier": voting_clf.predict_proba(user_X)[0][1]
}

# Display predictions
col1, col2 = st.columns(2)

with col1:
    st.header("Course Completion Probability")
    for model, prob in predictions.items():
        st.write(f"{model}: {prob:.2%}")


# Recommendations
st.header("Recommendations")
engagement_score = (logins_per_week * 0.5 + videos_watched * 0.3 + time_spent_on_platform * 0.2)
completion_ratio = courses_completed / courses_started if courses_started > 0 else 0

if engagement_score < 5:
    st.write("- Increase your engagement by logging in more frequently and watching more videos.")
if avg_quiz_score < 70:
    st.write("- Focus on improving your quiz scores. Consider reviewing the material more thoroughly.")
if completion_ratio < 0.5:
    st.write("- Try to complete more of the courses you start. Set achievable goals for course completion.")
if time_spent_on_platform < 7:
    st.write("- Spend more time on the learning platform to improve your chances of course completion.")

st.write("\nRemember, these recommendations are based on general trends in the data and may not apply to everyone equally.")

# Data Overview
st.header("Data Overview")
st.write(st.session_state.data.describe())
