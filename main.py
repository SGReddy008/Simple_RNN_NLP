##Step 1: Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


##Load the wordindex
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()} 


#Load the pre trained model
model = load_model('simplernn.h5')


##Step 2: Helper Functions
##Function to decode the review
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

##Function to preprocess the review
def preprocess_review(review):
    # Tokenize the review
    tokens = review.lower().split()
    # Convert tokens to integers using the word index
    encoded_review = [word_index.get(token, 2) + 3 for token in tokens]  # 2 is for unknown words
    # Pad the sequence to a fixed length (e.g., 500)
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

##Step 3: Prediction Function
def predict_review(review):
    # Preprocess the review
    preprocessed_review = preprocess_review(review)
    # Make prediction
    prediction = model.predict(preprocessed_review)
    # Return the sentiment based on the prediction
    return "Positive" if prediction[0][0] > 0.5 else "Negative" , prediction[0][0]


##Streamlit APP
import streamlit as st
st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict its sentiment (Positive or Negative).")

#User input
user_input = st.text_area("Enter your review here:")

if st.button('Classify'):
    preprocessed_input = preprocess_review(user_input)

    ##Make prediction
    prediction = model.predict(preprocessed_input)

    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    ##Display the result
    st.write(f"Predicted Sentiment: {sentiment}")
    st.write(f"Confidence: {prediction[0][0]:.2f}")
else:
    st.write("Please enter a review and click 'Classify' to see the prediction.")
