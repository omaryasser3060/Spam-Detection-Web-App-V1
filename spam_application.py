import streamlit as st
import pandas as pd
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download('punkt')
nltk.download('stopwords')


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')) - {'not', 'no'}

    def preprocess(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        return ' '.join([self.stemmer.stem(word) for word in tokens if word not in self.stop_words])

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self.preprocess)


model = joblib.load("spam_detector_model.pkl")


st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("Spam Message Classifier")


st.write("Select the type of message you'd like to classify:")
msg_type = st.radio("Message Type", options=["SMS", "Email"], horizontal=True, key="msg_type")


with st.form("classification_form"):
    user_input = st.text_area(f"Enter your {msg_type} message below:", height=150, key="user_input")
    submit = st.form_submit_button("Classify Message")


if submit:
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        input_series = pd.Series([user_input])
        prediction = model.predict(input_series)[0]

        st.subheader("Prediction:")
        st.write(f"This {msg_type.lower()} is classified as: **{prediction.upper()}**")
