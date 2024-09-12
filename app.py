import streamlit as st

# text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# modeling
import pickle
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')


#App layout
st.title('Restaurant Review Analysis')
review = st.text_input('Enter your review')
submit = st.button('Analyze')

#define vectorizer
vectorizer = CountVectorizer()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers

    # Tokenize text into words and POS tags
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)

    # Lemmatize words based on POS tags
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word.lower(), pos=get_pos_tag(tag)) for word, tag in tagged_tokens]

    # Remove stopwords
    filtered_tokens = [word for word in lemmatized_tokens if word not in set(stopwords.words('english'))]

    # Join tokens back into a string
    text = ' '.join(filtered_tokens)

    return text


# load the model
model = pickle.load(open('models/model.pkl', 'rb'))

if submit:
    review = preprocess_text(review)
    review = vectorizer.transform([review]).toarray()
    prediction = model.predict([review])
    st.write(prediction[0])
    if prediction == 1:
        st.write("We're glad you enjoyed your visit!")
    else: 
        st.write("Thank you for your feedback!")
    