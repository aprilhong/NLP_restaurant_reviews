import streamlit as st
import pickle

# load the model
model = pickle.load(open('models/model.pkl', 'rb'))

st.title('Restaurant Review Analysis')

tweet = st.text_input('Enter your review')

submit = st.button('Predict')

if submit:
    prediction = model.predict([tweet])
    print(prediction[0])
    st.write(prediction[0])