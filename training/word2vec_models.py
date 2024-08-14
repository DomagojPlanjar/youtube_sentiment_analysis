import os
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import pickle

# Custom transformer to convert texts to Word2Vec embeddings
class MeanEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = word2vec.vector_size

    # Fit is not needed
    def fit(self, X, y=None):
        return self

    # Apply embeddings to each word in each text and take mean value for each text
    # If word is not embedded in model, return zeros vector
    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)

    # Load Google News word2vec model (already downloaded)
    model_path = os.path.join(parent_dir, 'GoogleNews-vectors-negative300.bin')
    word2vec = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Load and process dataset

    # Change between 'data.csv' and 'balanced_downsampled_data.csv' as wished
    file_path = os.path.join(parent_dir, 'data', 'balanced_downsampled_data.csv')

    models_save_dir = os.path.join(parent_dir, 'models')

    df = pd.read_csv(file_path, header=None, encoding='ISO-8859-1')
    df.columns = ['label'] + [f'col_{i}' for i in range(1, df.shape[1] - 1)] + ['text']  # Set column names

    # Keep only the label and text columns
    df = df[['label', 'text']]

    # Preprocess (tokenize) text and labels
    X = df['text'].apply(lambda x: x.split())
    y = df['label']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the MeanEmbeddingTransformer with word2vec
    # (Takes mean value of word embeddings of one text)
    embedding_transformer = MeanEmbeddingTransformer(word2vec)

    # Random Forest model pipeline
    rf_pipeline = Pipeline([
        ("word2vec_transformer", embedding_transformer),
        ("random_forest", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Logistic Regression model pipeline
    lr_pipeline = Pipeline([
        ("word2vec_transformer", embedding_transformer),
        ("logistic_regression", LogisticRegression(random_state=42, max_iter=1000))
    ])

    svm_pipeline = Pipeline([
        ("word2vec_transformer", embedding_transformer),
        ("svm", SVC(kernel='linear', random_state=42))
    ])

    # Train the Random Forest model
    logging.info("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)

    # Train the Logistic Regression model
    logging.info("Training Logistic Regression model...")
    lr_pipeline.fit(X_train, y_train)

    # Train the SVM model
    logging.info("Training SVM model...")
    svm_pipeline.fit(X_train, y_train)

    # Predict and evaluate Random Forest
    logging.info("Evaluating Random Forest model...")
    y_pred_rf = rf_pipeline.predict(X_test)
    logging.info(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
    logging.info(f"Random Forest Classification Report:\n{classification_report(y_test, y_pred_rf)}")

    # Predict and evaluate Logistic Regression
    logging.info("Evaluating Logistic Regression model...")
    y_pred_lr = lr_pipeline.predict(X_test)
    logging.info(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr)}")
    logging.info(f"Logistic Regression Classification Report:\n{classification_report(y_test, y_pred_lr)}")

    # Predict and evaluate SVM
    logging.info("Evaluating SVM model...")
    y_pred_svm = svm_pipeline.predict(X_test)
    logging.info(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm)}")
    logging.info(f"SVM Classification Report:\n{classification_report(y_test, y_pred_svm)}")


    # Save the models
    with open(os.path.join(models_save_dir, "random_forest_model.pkl"), "wb") as rf_file:
        pickle.dump(rf_pipeline, rf_file)
        logging.info("Random Forest model saved as 'random_forest_model.pkl'")

    with open(os.path.join(models_save_dir, "logistic_regression_model.pkl"), "wb") as lr_file:
        pickle.dump(lr_pipeline, lr_file)
        logging.info("Logistic Regression model saved as 'logistic_regression_model.pkl'")

    with open(os.path.join(models_save_dir, "svm_model.pkl"), "wb") as svm_file:
        pickle.dump(svm_pipeline, svm_file)
        logging.info("SVM model saved as 'svm_model.pkl'")

if __name__ == "__main__":
    main()
