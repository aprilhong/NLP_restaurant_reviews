# :plate_with_cutlery: Sentiment Analysis of Restaurant Reviews: A Predictive Model :+1: :-1:


### Project Summary

Sentiment analysis of restaurant reviews provides valuable insights into customer satisfaction, identifies trends, improves service, informs marketing strategies, and helps analyze competition. For this project, I used [Kaggle's dataset](https://www.kaggle.com/datasets/ziadmostafa1/restaurant-reviews) containing real restaurant reviews with labels indicating whether the customer liked or disliked the restaurant. To prepare the text data for machine learning models, I applied text preprocessing techniques such as tokenization, lemmatization, stop word removal, and part of speech tagging. Next, I trained and evaluated the data using 5 different algorithms to classify customer reviews as positive or negative. Ultimately, the Navies Bayes emerged as the champion model. Lastly, I integrated model into a [Streamlit Web Application](https://restaurantreviewspredict.streamlit.app/) to predict customer sentiment based on new reviews.

**Tools/Skills**: *Natural Language Processing, Text Preprocessing, Lemmatization, Tokenization, Stop Word Removal, Naive Bayes Algorithm, Model Pipelines, Model Deployment, Python, Streamlit*

<details><summary>File Descriptions </summary>
  
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

### Project Objective
- Conduct a **Sentiment Analysis** to predict customer satisfaction
- Train a machine learning model to automatically determine if a review is positive or negative

### Methodologies 
Complete work in [jupyter notebook](notebooks/01-ah-restaurant-review-prediction.ipynb)

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

### Web Application
<a href="https://restaurantreviewspredict.streamlit.app/">
  <img src="image.png" width="400" >
</a>
<br>
<br>
