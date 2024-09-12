import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# from nltk.tokenize import word_tokenize
# from nltk.tag import pos_tag

# Load pre-trained model and vectorizer (outside button click)
model = pickle.load(open('models/model.pkl', 'rb'))
cv = pickle.load(open('models/vectorizer.pkl', 'rb'))

# Function to preprocess the text
def get_pos_tag(tag):
    """
    This function maps a given POS tag to a simplified POS tag used in lemmatization.

    Parameters:
    tag (str): The original POS tag.

    Returns:
    str: The simplified POS tag. If the original tag is not found in the map, it defaults to 'n' (noun).
    """
    pos_map = {
        'NN': 'n', 'NNS': 'n', 'NNP': 'n', 'NNPS': 'n',
        'VB': 'v', 'VBD': 'v', 'VBG': 'v', 'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
        'JJ': 'a', 'JJR': 'a', 'JJS': 'a',
        'RB': 'r', 'RBR': 'r', 'RBS': 'r'
    }
    return pos_map.get(tag, 'n')  # Default to noun if tag is not found

# Function to preprocess text
def preprocess_text(text):
    try:
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)     # Remove numbers

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
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return text  # Return original text in case of error
    
# def preprocess_text(text):
#     text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
#     text = re.sub(r'\d+', '', text)  # Remove numbers

#     # Tokenize text into words and POS tags
#     tokens = nltk.word_tokenize(text)
#     tagged_tokens = nltk.pos_tag(tokens)

#     # Lemmatize words based on POS tags
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = [lemmatizer.lemmatize(word.lower(), pos=get_pos_tag(tag)) for word, tag in tagged_tokens]

#     # Remove stopwords
#     filtered_tokens = [word for word in lemmatized_tokens if word not in set(stopwords.words('english'))]

#     # Join tokens back into a string
#     text = ' '.join(filtered_tokens)

#     return text


# Streamlit App Layout
def main():
    st.title('Restaurant Review Analysis')
    st.image('https://images.unsplash.com/photo-1414235077428-338989a2e8c0?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D')

    review = st.text_input('Enter your review')
    review = preprocess_text(review)
    review = cv.transform([review]).toarray()
    submit = st.button('Analyze')

    if submit:
        prediction = model.predict(review)
        st.write('Sentiment Score:', str(prediction))
        if prediction == 1:
            st.write(":smiley: We're glad you enjoyed your visit. Thank you for the positive review!")
        else: 
            st.write(":slightly_frowning_face: We apologize for your negative experience. We'll review your feedback and improve our service.")


    st.divider()
    st.subheader('Image credit:')
    st.write('Creator: User Jay Wennington (@jaywennington) from Unsplash')
    st.write('Free to use under the Unsplash License: https://unsplash.com/license')
    st.write('https://unsplash.com/photos/dish-on-white-ceramic-plate-N_Y88TWmGwA')


if __name__ == '__main__':
    main()