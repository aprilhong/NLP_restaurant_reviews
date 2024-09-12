# Sentiment Analysis of Restaurant Reviews


Data

Methodologies

Result


# Web Application
<a href="https://restaurantreviewspredict.streamlit.app/">
  <img src="image.jpg" width="400" >
</a>

# File Descriptions

- (data/raw)[data/raw]: stores the raw, unprocessed data used for analysis
- models: folder containing all trained machine learning models and the results of model evaluation (accuracy)
  - model.pkl: pickle trained champion model
  - vectorizer.pkl: pickled file of the fitted vectorizer
  - results_table.csv:  summary table of scoring metrics from all models
- notebooks: Jupyter notebook documenting the data analysis and model building process
- app.py: central file that loads data, interacts with models, and creates a user-friendly Streamlit interface.
- requirements.txt: lists all the Python libraries and packages required to run project
- setup.py: python script to handle downloading NLTK data
