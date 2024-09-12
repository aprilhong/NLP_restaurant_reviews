# :plate_with_cutlery: Sentiment Analysis of Restaurant Reviews :+1: :-1:


### About the Data
This dataset from [Kaggle](https://www.kaggle.com/datasets/ziadmostafa1/restaurant-reviews) contains real restaurant reviews with labels indicating whether the customer liked or disliked the restaurant. 


### Project Objective
- Conduct a **Sentiment Analysis** to predict customer satisfaction
- Train a machine learning model to automatically determine if a review is positive or negative

### File Descriptions 
<details><summary> expand/collapse </summary>

- [data/raw](data/raw): stores the raw, unprocessed data used for analysis
  - [Restaurant_Reviews.tsv](data\raw\Restaurant_Reviews.tsv): raw dataset from [Kaggle](https://www.kaggle.com/datasets/ziadmostafa1/restaurant-reviews)
- [models](models): folder containing all trained machine learning models and the results of model evaluation (accuracy)
  - model.pkl: trained champion model file
  - vectorizer.pkl: fitted vectorizer file
  - results_table.csv:  summary table of scoring metrics from all models
- [notebooks](notebooks): Jupyter notebook documenting the data analysis and model building process
- [app.py](app.py): central file that loads data, interacts with models, and creates a user-friendly Streamlit interface.
- [requirements.txt](requirements.txt): lists all the Python libraries and packages required to run project
- [setup.py](setup.py): python script to handle downloading NLTK data
</details>

### Methodologies 

complete work in [jupyter notebook](notebooks/01-ah-restaurant-review-prediction.ipynb)


<details><summary>expand/collapse</summary>

1. Import Libraries
2. Load and Clean Data 
3. Visualize Text Data
4. Text Preprocessing
    - Using Tokenization, Lemmatization, stop word removal, and part of speech tagging
5. Feature Extraction
6. Model Training and Evaluation
    - Trained 5 different models 
      - **Naive Bayes** - Champion model with highest accuracy score
      - Decision Tree
      - Random Forest
      - Logistic Regression
      - K Neighbors
7. Make predication on new unseen data
8. Compile machine learning pipeline in [app.py](app.py) to deploy on [streamlit](https://restaurantreviewspredict.streamlit.app/)
</details>

divier

### Web Application
<a href="https://restaurantreviewspredict.streamlit.app/">
  <img src="image.png" width="400" >
</a>
<br>
<br>
