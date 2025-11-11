
## Emotion Detection Project

# Overview
This project implements an emotion detection system using machine learning to classify text into different emotional categories.

# Dataset
The project uses the dair-ai/emotion dataset from Hugging Face, which contains labeled text samples across different emotion categories.

# Data Files
. emotion_train.csv - Training dataset (~16K samples)
. emotion_val.csv - Validation dataset (~2K samples)
. emotion_test.csv - Test dataset (~2K samples)

Each file contains two columns:

. text - The input text sample
. label - The emotion label (0-5)

# Emotion Labels
The labels represent the following emotions:

0 - Sadness
1 - Joy
2 - Love
3 - Anger
4 - Fear
5 - Surprise

## Methodology
# Data Preparation
The dataset is loaded using the Hugging Face datasets library and converted to pandas DataFrames for easier manipulation.

# Feature Extraction
Text features are extracted using TF-IDF Vectorization with a maximum of 5,000 features to capture the most important words.

# Model
A Logistic Regression classifier is trained on the vectorized training data to predict emotion labels.

# Evaluation
The model is evaluated on the validation set using:

. Classification metrics (precision, recall, F1-score)
. Confusion matrix analysis

## Usage
Running the Notebook
# Load dataset
dataset = load_dataset("dair-ai/emotion")

# Train model
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Make predictions
text = ["I am very excited to see you!"]
text_vec = vectorizer.transform(text)
pred = model.predict(text_vec)

## Requirements
. datasets - Hugging Face datasets library
. scikit-learn - Machine learning library
. pandas - Data manipulation
. numpy - Numerical computing
. matplotlib - Visualization

## Files
. emotion.ipynb - Main Jupyter notebook with full implementation
. emotion_train.csv - Training data
. emotion_val.csv - Validation data
. emotion_test.csv - Test data