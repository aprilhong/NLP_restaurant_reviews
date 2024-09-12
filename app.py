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
nltk.download('averaged_perceptron_tagger_eng')


#App layout
st.title('Restaurant Review Analysis')
st.image('https://images.unsplash.com/photo-1414235077428-338989a2e8c0?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')

def get_pos_tag(tag):
    pos_map = {
        'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
        'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
        'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
        'RB': 'r', 'RBR': 'r', 'RBS': 'r'
    }
    return pos_map.get(tag, 'n')  # Default to noun if tag is not found

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


review = st.text_input('Enter your review')
review = preprocess_text(review)
cv = pickle.load(open('models/vectorizer.pkl', 'rb'))
review = cv.transform([review]).toarray()
submit = st.button('Analyze')

# load the model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

if submit:
    prediction = model.predict(review)
    if prediction == 1:
        st.write("We're glad you enjoyed your visit. Thank you for the positive review!")
    else: 
        st.write("We apologize for your negative experience. We'll review your feedback to improve our service.")
    