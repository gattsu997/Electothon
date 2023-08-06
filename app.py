
import argparse
import os
import textract
from PIL import Image
import pytesseract
import cv2
import string
from nltk.corpus import stopwords
import nltk
import re
from nltk.util import pr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import importlib
import joblib
import streamlit as st

#fake links
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load the model for fake links
model = joblib.load("best_model.pkl")


data = pd.read_csv("twitter.csv")
print(data.head())

data["labels"] = data["class"].map(
    {0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})
#print(data.head())

data = data[["tweet", "labels"]]
#print(data.head())

stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))
vectorizer = joblib.load("tfidf_vectorizer.pkl")

model = joblib.load("best_model.pkl")

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text = " ".join(text)
    return text


data["tweet"] = data["tweet"].apply(clean)
#print(data.head())

x = np.array(data["tweet"])
y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(x)  # Fit the Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


def extract_urls_and_strings(text):
    # Regular expression to find URLs
    url_pattern = r'https?://\S+|www\.\S+'

    # Extract URLs from the text
    urls = re.findall(url_pattern, text)

    # Remove the URLs from the text
    pure_string = re.sub(url_pattern, '', text)

    # Return both the URLs as an array and the pure string
    return urls, pure_string



def hate_speech_detection():
    st.title("Hate Speech Detection")
    uploaded_image = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        # Convert PIL image to OpenCV image
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # Perform text detection
        filename = "{}.jpg".format(os.getpid())
        cv2.imwrite(filename, gray)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        tex = pytesseract.image_to_string(Image.open(
            filename))
        print("length"+str(len(tex)))
        print(tex)

        os.remove(filename)

        # Display detected text
        # st.write("Detected Text:")
        # st.write(text)
        if(tex != ""):
            data = cv.transform([tex]).toarray()
            cap = clf.predict(data)
        
        else:
            st.title("cannot interpret it !!")

        # Display output images
        # st.image(gray, caption=cap, use_column_width=True)
        st.title(cap[0]+" in the meme !")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write("  ")
    else:
        sample = user
        urls, tex = extract_urls_and_strings(sample)
        if(tex!=""):
            
            data = cv.transform([tex]).toarray()
            a = clf.predict(data)
            st.title(a)
        else:
            st.title("No text present in the tweet")
        # Convert the text to a vector
        try:
            vector = vectorizer.transform([urls[0]])
            label = model.predict(vector)[0]
            if label == 0:
                st.title("The news article is fake.")
            else:
                st.title("The news article is real.")
        
        except Exception as e:
            st.title("No urls present in the tweet")

    # Predict the label
       

hate_speech_detection()




