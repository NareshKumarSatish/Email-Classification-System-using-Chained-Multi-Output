# Email-Classification-System-using-Chained-Multi-Output# Email-Classification-using-Chained-Multi-Outputs
This project implements an email classification system using a chained multi-output architecture and the RandomForest classifier. The system is designed to categorize emails into multiple categories simultaneously, leveraging the strengths of ensemble learning for robust performance.

## Key Features
Chained Multi-Output Architecture: This approach allows for the sequential prediction of multiple target variables, where each prediction can be conditioned on the previous ones. This is particularly useful for multi-label classification tasks, where the presence of one label may influence the likelihood of another.

RandomForest Classifier: A powerful ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of predictions. The use of RandomForest ensures that the model generalizes well to unseen data and handles the variability within the dataset.

# Workflow
## Data Preparation:
Load and preprocess the email data from different sources.
Clean and de-duplicate the content using various text processing techniques.
Remove noise and irrelevant information to enhance the quality of the input data.

## Feature Extraction:
Use TF-IDF (Term Frequency-Inverse Document Frequency) to transform the textual content into numerical features.
Combine different feature sets to create a comprehensive representation of the data.

## Model Training and Evaluation:
Implement a chained multi-output architecture where each classifier in the chain predicts a specific label.
Train the RandomForest classifiers on the training data.
Evaluate the model performance using k-fold cross-validation to ensure robustness and reliability.

## Prediction:
Generate predictions for new email data using the trained model.
Compute accuracy and other performance metrics to assess the model's effectiveness.
Usage

Data Loading:
Ensure your email data is formatted correctly and saved in the data directory.
Use the get_input_data function to load and preprocess the data.

Training the Model:
Instantiate the Data class with your feature matrix and DataFrame.
Train the model using the RandomForest class, specifying the name and class columns.

Making Predictions:
Use the model_predict function to generate predictions for new data.
The function will return the predictions, accuracy for each type, group accuracies, and the overall average accuracy.

Evaluating Performance:
Perform k-fold cross-validation to obtain reliable performance estimates.
Review fold-specific results and average performance metrics to understand the model's behavior.

## Dependencies
Python 3.8+
NumPy
Pandas
Scikit-learn
Stanza
# Email-Classification-using-Chained-Multi-Output
