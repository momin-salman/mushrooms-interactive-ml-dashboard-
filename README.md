This is a dynamic machine learning dashboard built with Streamlit and scikit-learn for exploring and comparing classification models on the Mushroom dataset.

Live Application
Try the interactive dashboard here:

👉 https://mushrooms-int.streamlit.app/

Project Goal
The primary goal of this application is to demonstrate how different classification algorithms perform when predicting whether a mushroom is edible (e) or poisonous (p) based on various categorical features (cap shape, odor, color, etc.).

Features
The dashboard allows users to:

Select a Classifier: Choose between Support Vector Machine (SVM), Logistic Regression, and Random Forest.

Tune Hyperparameters: Adjust key parameters (like regularization strength C or tree depth) in real-time using interactive sliders.

Evaluate Performance: Instantly view the model's accuracy score on the test set.

Visualize Results: Display crucial evaluation plots:

Confusion Matrix: To see true positives/negatives.

ROC Curve: To evaluate classifier performance across all threshold settings.

Precision-Recall Curve: Ideal for binary classification where one class (poisonous) may be more critical.

Technologies Used
Streamlit: For building the interactive web application interface.

scikit-learn: For implementing and training the classification models.

Pandas & NumPy: For data loading and preprocessing (Label Encoding is used for all categorical features).

Matplotlib: For rendering the performance visualization plots
