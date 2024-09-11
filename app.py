# data manipulation and visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import string

# modeling
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from  sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report,confusion_matrix, ConfusionMatrixDisplay

import streamlit as st
import pandas as pd
import numpy as np


st.title('Restaurant Reviews App')

# load the dataset
def load_data():
    df = pd.read_csv("data/raw/Restaurant_Reviews.tsv", sep='\t')
    return df

def clean_data(df):
    """
    This function cleans the given DataFrame by renaming columns, removing null values, and removing duplicate rows.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing restaurant reviews. It should have at least two columns: 'ReviewText' and 'Sentiment'.

    Returns:
    pandas.DataFrame: The cleaned DataFrame with the specified modifications.
    """
    df.columns = ['text','label']  # Rename columns to 'text' and 'label'
    df = df.dropna()  # Remove rows with null values
    df = df.drop_duplicates(keep='first')  # Remove duplicate rows, keeping the first occurrence

    return df

# preprocess the data
def get_pos_tag(tag):
    pos_map = {
        'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
        'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
        'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
        'RB': 'r', 'RBR': 'r', 'RBS': 'r'
    }
    return pos_map.get(tag, 'n')  # Default to noun if tag is not found

def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers

    # Tokenize text into words and POS tags
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)

    # Lemmatize words based on POS tags
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word, pos=get_pos_tag(tag)) for word, tag in tagged_tokens]

    # Remove stopwords
    filtered_tokens = [word for word in lemmatized_tokens if word not in set(stopwords.words('english'))]

    # Join tokens back into a string
    text = ' '.join(filtered_tokens)

    return text



# Load and clean the dataset
def split_data():

    df = load_data()
    df = clean_data(df)

    # Apply Preprocessing
    df['text'] = df['text'].apply(preprocess_text)
    df.head()

    # Convert Text Data into Vectors
    vectorizer = CountVectorizer()
    st.write('X')
    X = vectorizer.fit_transform(df['text'])
    st.write('y')
    y = df['label']

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write('X_train')
    X_train

    st.write('y_train')
    y_train

st.write(split_data())

def evaluate_models(models, X_train, y_train, X_test, y_test, param_grid_dict, choose_score='accuracy'):
    """
    Evaluates a dictionary of models with hyperparameter tuning (optional) and outputs a table of results.
    Also pickles the model with the highest chosen score (accuracy, precision, or recall).

    Args:
        models (dict): A dictionary where keys are model names and values are scikit-learn models.
        X_train (array): Training data.
        y_train (array): Training labels.
        X_test (array): Testing data.
        y_test (array): Testing labels.
        param_grid_dict (dict, optional): A dictionary where keys are model names and
                                          values are dictionaries defining the hyperparameter grid for GridSearchCV.
                                          Defaults to None.
        choose_score (str, optional): The metric to use for selecting the best model for pickling.
                                      Options: 'accuracy', 'precision', 'recall'. Defaults to 'accuracy'.

    Returns:
        pandas.DataFrame: A DataFrame containing the evaluation results, including best hyperparameters if tuning was performed.
    """

    results = []
    path = 'models/'  # Path to save pickled models

    best_model = None
    best_score = 0  # Initialize with negative infinity

    for name, model in models.items():
        if param_grid_dict is not None and name in param_grid_dict:
            # Perform hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(model, param_grid_dict[name], scoring=choose_score, cv=5)
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model.fit(X_train, y_train)
            best_params = None

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        results.append({
            'model name': name,
            'best hyperparameters': best_params,
            'accuracy': f'{accuracy:.4f}',
            'precision': f'{precision:.4f}',
            'recall': f'{recall:.4f}'
        })     

        # Get the metric value based on chosen score
        metric_score = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }[choose_score]

        # Update best model and score if current score is higher
        if metric_score > best_score:
            best_model = model
            best_name = name
            best_score = metric_score
            best_params = best_params        

        df = pd.DataFrame(results)
        df = df.sort_values(choose_score, ascending=False)

    # Pickle the trained model
    with open(path + 'model' + '.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    if best_model is not None:
        # Print results only for the best model
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            cm = confusion_matrix(y_test, y_pred)

            print(f"\n--- Best Model Results ---\n")
            print(f"Best Model Name: {best_name}")
            print(f"{choose_score}: {best_score:.4f}")
            print(f"Best Hyperparameters: {best_params}")

            
            print(f"\nConfusion Matrix Results" ) 
            print("True Positives:", cm[1,1])
            print("True Negatives:", cm[0, 0])
            print("False Positives (Type I error):", cm[0, 1],'(where a true negative was incorrectly predicted as positive)')
            print("False Negatives (Type II error):", cm[1, 0], '(where a true positive was incorrectly predicted as negative)') 

    # Save the master results table (unchanged)
    df.to_csv(path + 'results_table.csv', index=False)
    
    print('------------------------')
    print('\nMODEL SUMMARY')
    return df




# # Fit models
# models = {
#     'Naive Bayes': MultinomialNB(),
#     'Decision Tree': DecisionTreeClassifier(),
#     'Random Forest': RandomForestClassifier(),
#     'Logistic Regression': LogisticRegression(),
#     'K Neighbors': KNeighborsClassifier()
# }

# # Define hyperparameter grids 
# param_grid_dict = {
#     'Decision Tree': {
#         'max_depth': [2, 4, 6],
#         'min_samples_split': [2, 5, 10]
#     },
#     'Random Forest': {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [2, 4, 6]
#     },
#     'KNeighbors': {
#         'n_neighbors': [3, 5, 7],
#         'weights': ['uniform', 'distance'],
#         'metric': ['euclidean', 'minkowski', 'manhattan']
#     }
# }

# evaluate_models(models, X_train, y_train, X_test, y_test, param_grid_dict,choose_score='precision')