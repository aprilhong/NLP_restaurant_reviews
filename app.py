import streamlit as st
import pickle


st.title('Restaurant Review Analysis')

review = st.text_input('Enter your review')

submit = st.button('Analyze')

# load the model
model = pickle.load(open('models/model.pkl', 'rb'))

if submit:
    prediction = model.predict([review])
    print(prediction[0])
    st.write(prediction[0])