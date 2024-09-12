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
review = st.text_input('Enter your review')
submit = st.button('Analyze')

# load the model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)



# Helper function to map NLTK POS tags to WordNet POS tags
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

if submit:
    review = preprocess_text(review)
    vectorizer = CountVectorizer()
    review = vectorizer.fit_transform(review).toarray()
    prediction = model.predict(review)
    st.write(prediction)
    if prediction == 1:
        st.write("We're glad you enjoyed your visit!")
    else: 
        st.write("Thank you for your feedback!")
    