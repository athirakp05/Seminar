import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\athir\Desktop\milkquality.csv')

# Remove leading spaces from column names
df.columns = df.columns.str.strip()

# Preprocessing
label_encoder = preprocessing.LabelEncoder()
df['grade'] = label_encoder.fit_transform(df['grade'])

# Define features and target variable
milk_features = ['pH', 'temperature', 'taste', 'odor', 'fat', 'turbidity', 'colour']
milk_feature_description = {
    'pH': 'PH values of the milk which ranges from 3 to 9.5 max ',
    'temperature': 'Temperature of the milk which ranges from 34(celsius) to 45.20 (celsius) max',
    'taste': 'Taste of the milk which is categorical data 0 (Bad) or 1 (Good)',
    'odor': 'Odor of the milk which is categorical data 0 (Bad) or 1 (Good)',
    'fat': 'Fat of the milk which is categorical data 0 (Low) or 1 (High)',
    'turbidity': 'Turbidity of the milk which is categorical data 0 (Low) or 1 (High)',
    'colour': 'Color of the milk which ranges from 240 to 255(RGB values)(pale to dark'
}
X = df[milk_features]
y = df['grade']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train the Decision Tree Classifier model with adjusted hyperparameters
dt_clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
dt_clf.fit(X_train, y_train)

# Function to calculate accuracy score and display confusion matrix
def calculate_accuracy():
    # Predict the labels for testing data
    y_pred = dt_clf.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Generate classification report
    cr = classification_report(y_test, y_pred, target_names=['low_quality', 'medium_quality', 'high_quality'], output_dict=True)

    # Extract precision, recall, and F1-score for each quality grade
    precision = [cr[label]['precision'] for label in ['low_quality', 'medium_quality', 'high_quality']]
    recall = [cr[label]['recall'] for label in ['low_quality', 'medium_quality', 'high_quality']]
    f1_score = [cr[label]['f1-score'] for label in ['low_quality', 'medium_quality', 'high_quality']]

    # Create a DataFrame to store metrics
    metrics_df = pd.DataFrame({
        'Quality Grade': ['low_quality', 'medium_quality', 'high_quality'],
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score
    })

    # Display accuracy score, metrics, and classification report in a table
    st.info(f"The accuracy score of the model is: {accuracy}")
    st.table(metrics_df)
    st.info("Classification Report:")
    st.table(pd.DataFrame(cr).transpose())

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['low_quality', 'medium_quality', 'high_quality'], yticklabels=['low_quality', 'medium_quality', 'high_quality'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

# Function to predict grade and display quality variations graph
def predict_grade(user_input):
    try:
        # Convert user input to DataFrame
        user_df = pd.DataFrame(user_input)

        # Fill missing values with 0 for categorical features
        user_df.fillna(0, inplace=True)

        # Predict the probabilities of each grade based on user input
        probabilities = dt_clf.predict_proba(user_df)

        # Display the neural network outputs
        grades = ['low_quality', 'medium_quality', 'high_quality']
        results = []
        for i, prob in enumerate(probabilities):
            result = {
                'pH': user_input[i].get('pH', 0),  # Use get method to handle missing key
                'taste': user_input[i].get('taste', 0),
                'odor': user_input[i].get('odor', 0),
                'fat': user_input[i].get('fat', 0),
                'turbidity': user_input[i].get('turbidity', 0),
                'colour': user_input[i].get('colour', 0),
                'temperature': user_input[i].get('temperature', 0)  # Default value if temperature is missing
            }
            for grade, p in zip(grades, prob):
                result[grade.capitalize()] = f"{p * 100:.2f}%"
            results.append(result)

        # Create a DataFrame with the results
        result_df = pd.DataFrame(results)

        # Create a bar plot of the predicted probabilities
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=grades, y=probabilities[0], palette='Blues')
        plt.xlabel('Quality Level')
        plt.ylabel('Probability')
        plt.title('Predicted Quality Probabilities')
        st.pyplot(fig)

        return result_df

    except ValueError:
        st.error("Please enter valid numerical values.")


# Function to display parameter descriptions and normal values
def display_parameter_description():
    st.subheader("Parameter Descriptions and Normal Values")
    for feature in milk_features:
        st.write(f"**{feature.capitalize()}**: {milk_feature_description[feature]}")

# Main Streamlit app
def main():
    if 'accuracy_calculated' not in st.session_state:
        st.session_state.accuracy_calculated = False

    st.title("Milk Quality Prediction")

    # Allow users to upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        uploaded_df = pd.read_csv(uploaded_file)

        # Remove leading spaces from column names
        uploaded_df.columns = uploaded_df.columns.str.strip()

        # Preprocess the uploaded data
        for col in uploaded_df.columns:
            if col not in milk_features:
                st.error(f"Column '{col}' is not supported. Please ensure your CSV file contains columns: {', '.join(milk_features)}")
                return

        # Concatenate training data with uploaded data
        combined_df = pd.concat([df, uploaded_df], ignore_index=True)

        # Convert categorical features to numerical using label encoding
        for col in ['taste', 'odor', 'fat', 'turbidity']:
            if col in combined_df.columns:
                label_encoder.fit(combined_df[col])  # Fit label encoder on combined data
                uploaded_df[col] = label_encoder.transform(uploaded_df[col])

        # Predict grades for the uploaded data
        predictions = dt_clf.predict(uploaded_df[milk_features])  # Use only relevant features for prediction

        # Map numerical predictions to their corresponding labels
        prediction_labels = {0: 'low_quality', 1: 'medium_quality', 2: 'high_quality'}
        predicted_labels = [prediction_labels[prediction] for prediction in predictions]

        # Add predicted labels to the DataFrame
        uploaded_df['predicted_grade'] = predicted_labels

        # Display predictions in a table
        st.write("Predictions:")
        st.write(uploaded_df)

        # Allow users to download predictions as CSV file
        st.download_button(
            label="Download Predictions as CSV",
            data=uploaded_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )

        # Create a pie chart showing the distribution of milk samples by quality grade
        grade_counts = uploaded_df['predicted_grade'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(grade_counts, labels=grade_counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Distribution of Milk Samples by Quality Grade')
        st.pyplot(fig)

    # Input fields for individual prediction
    st.subheader("Predict Individual Milk Quality")
    user_input = []
    for i, feature in enumerate(milk_features):
        if feature in ['taste', 'odor', 'fat', 'turbidity']:
            user_input.append({feature: st.selectbox(feature, [0, 1], key=f"{feature}_input_{i}")})
        else:
            user_input.append({feature: float(st.text_input(feature, 0.0, key=f"{feature}_input_{i}"))})

    # Buttons
    if st.button("Predict Grade", key="predict_grade"):
        predicted_df = predict_grade(user_input)
        st.write("Predicted Individual Milk Quality:")
        st.write(predicted_df)

    if st.button("Calculate Accuracy", key="calculate_accuracy") and not st.session_state.accuracy_calculated:
        calculate_accuracy()
        st.session_state.accuracy_calculated = True

    if st.button("Parameter Descriptions", key="parameter_descriptions"):
        display_parameter_description()
    # Custom CSS
    st.markdown(
        """
        <style>
            /* Add custom CSS styles here */
            .stButton>button {
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
                transition: all 0.3s ease 0s;
            }

            .stButton>button:hover {
                background-color: #45a049;
            }

            .stSelectbox>div {
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
                background-color: #f2f2f2;
                border: none;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()
