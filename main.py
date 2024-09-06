import nltk
nltk.download('punkt')
# Unduh stopwords
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.probability import FreqDist

from IPython.display import clear_output

import pandas as pd
import pickle
import string
import random
import joblib



classifier = None
review = ""
category = ""

def load_data():
  df = pd.read_csv("Restaurant_Reviews.tsv", sep= '\t')
  return df



# Give the user a moment to read some output before continuing with the next steps.
def enter():
  input("Press enter to continue ....")

def get_tag(tag):
  if tag.startswith('J'):
    return 'a'
  elif tag.startswith('V'):
    return 'v'
  elif tag.startswith('N'):
    return 'n'
  elif tag.startswith('R'):
    return 'r'
  else:
    return 'n'
  
def preprocess_word(word_list):

  # Remove stopwords
  word_list = [word for word in word_list if word.lower() not in stopwords.words('english')]

  # Remove punctuation
  word_list = [word for word in word_list if word.lower() not in string.punctuation]

  # Remove number
  word_list = [word for word in word_list if word.isalpha()]

  # Pos tag
  word_tag = pos_tag(word_list)

  # Lemmatizing
  wnl = WordNetLemmatizer()
  word_list = [wnl.lemmatize(word, pos=get_tag(tag)) for word, tag in word_tag]

  # Stemming
  stemmer = PorterStemmer()
  word_list = [stemmer.stem(word) for word in word_list]

  return word_list

def train_model():
    # Load the dataset
    df = load_data()

    # Convert reviews and sentiments into strings
    reviews = [str(review) for review in df['Review'].to_list()]
    sentiments = [str(sentiment) for sentiment in df['Liked'].to_list()]

    # Empty list to store all words in the reviews
    word_list = []

    # Tokenize each review into words and add them to the word list
    for sentence in reviews:
        words = word_tokenize(sentence)

        for word in words:
            word_list.append(word)

    # Preprocess the words
    word_list = preprocess_word(word_list)

    # Combine reviews and their corresponding sentiments into labeled data
    labeled_data = list(zip(reviews, sentiments))

    # Initialize a list to store the feature sets
    feature_sets = []

    # For each review and sentiment, create a feature dictionary
    for review, sentiment in labeled_data:
        feature = {}

        # Tokenize and preprocess the current review
        check_words = word_tokenize(review)
        check_words = preprocess_word(check_words)

        # For each word in the word list, check if it is in the current review
        for word in word_list:
            feature[word] = word in check_words

        # Add the feature set and sentiment to the feature sets list
        feature_sets.append((feature, sentiment))

    # Shuffle the feature sets to randomize the data
    random.shuffle(feature_sets)

    # Split the data into training (80%) and testing (20%) datasets
    train_count = int(len(feature_sets) * 0.8)
    train_dataset = feature_sets[:train_count]
    test_dataset = feature_sets[train_count:]

    # Train a Naive Bayes classifier 
    classifier = NaiveBayesClassifier.train(train_dataset)

    # Print the accuracy of the classifier on the test dataset
    print(f"Accuracy: {accuracy(classifier, test_dataset) * 100 : ,.2f}%")

    # Save the trained model to a file using pickle
    file = open('models/model_restaurant.pickle', 'wb')
    pickle.dump(classifier, file)
    file.close()

    # Return the trained classifier
    return classifier


# Make menu

def print_menu():
  global review
  displayReview = "No review"

  global category
  displayCategory = "None"

  if review != "":
    displayReview = review

  if category != "":
    displayCategory = category

  print("Restaurant Review Sentiment Analysis")
  print(f"Your review: {displayReview}")
  print(f"Category: {displayCategory}")
  print("1. Enter your review: ")
  print("2. Exit")
  choice = input(">> ")
  return choice


# Write review

def write():
  clear_output()
  global review
  global classifier
  global category

  print("Enter your review (must be more than 5 words): ")
  inputReview = input(">> ")

  if len(inputReview.split(' ')) < 5:
    print("Review must be more than 5 words")
    enter()
    return

  review = inputReview

  words = word_tokenize(review)
  words = preprocess_word(words)
  feature = FreqDist(words)

  category = classifier.classify(feature)
  print(f"your review is classfied as {category}")
  enter()


# Main menu

def main():

    global classifier
    try:
        file = open('models/model_restaurant.pickle', 'rb')
        classifier = pickle.load(file)
        file.close()
    except FileNotFoundError:
        classifier = train_model()

    while True:
        clear_output()
        choice = print_menu()
        if choice == '1':
            write()
        elif choice == '2':
            break

    print("Thank You!")


main()