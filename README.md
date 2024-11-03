## Usage

1. **Data Generation and Preprocessing**: 
   - The code generates synthetic data for 10,000 students with demographic, engagement, and historical performance features.
   - It includes steps for feature engineering, such as creating an engagement score and calculating a completion ratio, as well as handling categorical variables through one-hot encoding.

2. **Model Training and Evaluation**: 
   - The app trains Random Forest and Gradient Boosting models on resampled data (using SMOTE) to address class imbalance.
   - Models are evaluated on test data, providing metrics like accuracy and classification reports.

3. **Feature Importance**: 
   - The feature importances of the Random Forest model help highlight the factors most influential for predicting course completion.

4. **Streamlit Interface**: 
   - The Streamlit app allows users to input student data and get course completion predictions along with customized recommendations for improvement.

---

## Models and Algorithms

- **Random Forest Classifier**: Provides a baseline ensemble method for classification.
- **Gradient Boosting Classifier**: Adds an adaptive boosting approach to improve predictive accuracy.
- **Voting Classifier**: Combines predictions from Random Forest and Gradient Boosting to improve robustness.

Each model is trained on resampled data to handle imbalances and ensure better predictive performance.

---

## Web Application

The Streamlit app provides an interactive interface:

1. **Student Profile Input**: 
   - Allows users to enter details like age, gender, major, and engagement metrics.

2. **Course Completion Probability**: 
   - Predicts the likelihood of course completion based on the input features using the Voting Classifier model.

3. **Recommendations**: 
   - Offers personalized suggestions to improve engagement and course completion probability based on the model’s analysis.

---

## Project Structure

```plaintext
|-- data/                    # Directory for synthetic data generation
|-- app.py                   # Main Streamlit app file
|-- model_training.py        # Script for model training and evaluation
|-- requirements.txt         # List of required libraries
|-- README.md                # Project documentation
