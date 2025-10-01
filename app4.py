import streamlit as st # Streamlit is used to build the interactive web application.
import pandas as pd # Pandas is used for data manipulation and creating DataFrames.
import numpy as np # NumPy is used for numerical operations and array creation.

# Scikit-learn models for classification
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Scikit-learn utility for splitting data
from sklearn.model_selection import train_test_split

# Scikit-learn metrics for model evaluation (imported but not all are used in this example)
from sklearn.metrics import precision_score, recall_score, f1_score

# Scikit-learn preprocessor for transforming data
from sklearn.preprocessing import LabelEncoder

# The following imports are for the new display classes
# that replaced the deprecated plot_ functions in scikit-learn 1.0+.
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

# Matplotlib for generating and displaying plots
import matplotlib.pyplot as plt

def main():
    """Main function to run the Streamlit app."""
    # Set the title of the Streamlit application
    st.title("Interactive ML Model Dashboard")

    # Sidebar for user input and model selection
    st.sidebar.title("Select Model and Parameters")
    
    # Model selection dropdown in the sidebar
    classifier_name = st.sidebar.selectbox(
        "Select classifier",
        ("Logistic Regression", "Random Forest", "SVM")
    )
    
    # Use Streamlit's data caching to avoid reloading the data on every interaction.
    # This is a key feature for performance in Streamlit apps.
    @st.cache_data
    def load_data():
        """Loads and preprocesses the mushrooms.csv dataset."""
        try:
            df = pd.read_csv('mushrooms.csv')
        except FileNotFoundError:
            st.error("Error: 'mushrooms.csv' not found. Please place the file in the same directory.")
            # FIX: Return an empty list for column_names to match the expected return signature.
            return np.array([]), np.array([]), []
        
        # Create a copy to avoid mutating the cached object
        df_copy = df.copy()
        
        # Separate features and target variable
        # The 'class' column is the target variable (e = edible, p = poisonous)
        X = df_copy.drop('class', axis=1)
        y = df_copy['class']
        
        # All features are categorical, so we need to encode them.
        le = LabelEncoder()
        
        # Store column names before converting to NumPy array
        column_names = X.columns.tolist()
        
        for column in column_names:
            X[column] = le.fit_transform(X[column])
            
        y = le.fit_transform(y)
        
        # Convert DataFrame X to a NumPy array, but y is already a NumPy array.
        return X.to_numpy(), y, column_names

    # Load the data using the cached function
    X, y, column_names = load_data()

    # Check if data was loaded successfully before proceeding
    if X.size == 0 or y.size == 0:
        return

    st.write("### Dataset Preview")
    # Display the processed data in a Pandas DataFrame, using the stored column names
    st.write(pd.DataFrame(X, columns=column_names))
    st.write(f"Shape of the dataset: {X.shape}")
    st.write(f"Number of classes: {np.unique(y).size}")

    # Split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Function to get the classifier and its parameters based on selection
    def get_classifier(name):
        params = {}
        # SVM model and its C parameter slider
        if name == "SVM":
            st.sidebar.subheader("SVM Parameters")
            params['C'] = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
            # Add probability=True to enable plotting of ROC and Precision-Recall curves.
            return SVC(probability=True, **params)
        # Logistic Regression model and its C parameter slider
        elif name == "Logistic Regression":
            st.sidebar.subheader("Logistic Regression Parameters")
            params['C'] = st.sidebar.slider("C (Regularization)", 0.01, 10.0)
            return LogisticRegression(**params)
        # Random Forest model and its parameter sliders
        elif name == "Random Forest":
            st.sidebar.subheader("Random Forest Parameters")
            params['n_estimators'] = st.sidebar.slider("Number of estimators", 1, 100, 25)
            params['max_depth'] = st.sidebar.slider("Max depth", 1, 15, 5)
            return RandomForestClassifier(**params)
    
    # Instantiate and train the selected model
    model = get_classifier(classifier_name)
    model.fit(X_train, y_train)
    
    # Evaluate the model's accuracy on the test data
    score = model.score(X_test, y_test)
    st.write(f"### Model: {classifier_name}")
    st.write(f"Accuracy: **{score:.2f}**")
    
    # --- Displaying Plots ---
    st.write("### Model Performance Visualizations")
    
    # Checkbox to display the Confusion Matrix
    if st.checkbox("Show Confusion Matrix"):
        st.subheader("Confusion Matrix")
        # Create a figure and axes before plotting
        fig, ax = plt.subplots()
        # Pass the axes to the from_estimator function
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig) # Use st.pyplot to display the matplotlib figure
        plt.close(fig) # Close the figure to free up memory
        
    # Checkbox to display the ROC Curve
    if st.checkbox("Show ROC Curve"):
        st.subheader("ROC Curve")
        # Create a figure and axes before plotting
        fig, ax = plt.subplots()
        # Pass the axes to the from_estimator function
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        
    # Checkbox to display the Precision-Recall Curve
    if st.checkbox("Show Precision-Recall Curve"):
        st.subheader("Precision-Recall Curve")
        # Create a figure and axes before plotting
        fig, ax = plt.subplots()
        # Pass the axes to the from_estimator function
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

if __name__ == '__main__':
    # This is the standard entry point for Python scripts.
    # It ensures the main() function runs only when the script is executed directly.
    main()