# 📘 Project Documentation

### 1️⃣ Introduction

This project performs sentiment analysis using a combination of NLP preprocessing, vectorization, and a Neural Network model (MLP). The model predicts whether a given text (headline + review) is positive or negative. A Streamlit web app is built for user-friendly predictions.

### 📌 Why I Chose This Project

I chose this Sentiment Analysis using Deep Learning project because sentiment classification is one of the most practical and widely used Natural Language Processing (NLP) applications. Companies like Amazon, Flipkart, and social media platforms rely heavily on sentiment analysis to understand customer opinions, improve product quality, and automate review moderation.

* This project allowed me to work with:

* Real-world text data

* NLP preprocessing techniques

* Machine Learning + Deep Learning (TensorFlow & Keras)

* Vectorization using CountVectorizer

* Feature engineering (combining multiple vectors)

* Model deployment using Streamlit

* It gave me hands-on experience across the complete ML pipeline, from data cleaning to deployment.



### 2️⃣ Project Objective

* Build sentiment classifier using Deep Learning

* Preprocess text using NLP techniques

* Deploy the model using Streamlit

* Provide real‑time predictions to users



### 3️⃣ Features

* Text preprocessing (stopwords removal, stemming/lemmatization)

* Dual-vector input combining (CountVectorizer)

* MLP-based classifier

* Web UI using Streamlit

* Error-handling + clean UI



### 4️⃣ Dataset Description

* Dataset contains headline and review_body fields.

* Target variable: sentiment (0 = negative, 1 = positive)

* Used for training the neural network.



### 5️⃣ NLP Pipeline

* Lowercasing

* Tokenization (word_tokenize)

* Stopword removal

* Stemming / Lemmatization

* CountVectorizer applied separately on headline + review

* Combined vectors using: hstack([vec1, vec2])



### 6️⃣ Model Architecture

Sequential([
 Dense(256, activation='relu', input_shape=(16260,), kernel_regularizer=l2(0.001)),
 Dropout(0.5),
 Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
 Dropout(0.4),
 Dense(1, activation='sigmoid')
])

* Loss: Binary Crossentropy
* Optimizer: Adam
* Metric: Accuracy



### 7️⃣ Model Training

* 10 epochs

* Batch size = 32

* 20% validation split

* EarlyStopping applied



### 8️⃣ Streamlit App Workflow

* User enters headline & review

* Model preprocesses input

* Features are vectorized

* Combined vector passed to model

* Output displayed as:

* Sentiment

* Confidence score



### 9️⃣ Folder Structure

(Already included  in the README)



### 🔟 Deployment Guide

* Steps for GitHub + Streamlit cloud deploy (Already added in README — refer above)


## 📌 Challenges Faced During the Project


### 1. Text Preprocessing Complexity

* Handling messy text like:

* Stopwords

* Punctuation

* Mixed cases

* Non-English characters

* Emojis

* I solved this using tokenization, stopword removal, and lemmatization.

### 2. Combining Two Text Inputs (Headline + Review Body)

Both texts needed separate vectorization → then merging into a single input.
Initially, I got errors (csr_matrix transform issue).
Finally, I solved it using:

from scipy.sparse import hstack
input = hstack([vec1, vec2])

### 3. Model Overfitting

* My model was overfitting initially.I fixed it by:

* Adding Dropout

* Using L2 regularization

* Using EarlyStopping callback

### 4. Deployment Issues

* Streamlit Cloud required:

* GitHub repository

* requirements.txt

* Clean folder structure

* I fixed errors related to transformers not installed, CSR matrix issues, and missing files.

### 5. Debugging Streamlit Errors

* The biggest challenge was:

* “csr_matrix has no attribute transform”

* Import errors

* Wrong vectorizer loading

* After debugging, I properly saved & loaded vectorizers using joblib.

### 6. Creating a User Interface

* Building a clean Streamlit UI and handling user input required careful structuring and error handling.