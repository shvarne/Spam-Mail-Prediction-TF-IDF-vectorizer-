# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:22:11 2026

@author: shrey
"""

import streamlit as st
import pickle

model = pickle.load(open('D:/Generative AI (Ml, Dl, etc)/Natural Language Processing/Project-Spam mail Prediction/spam_model.pkl','rb'))
cv = pickle.load(open('D:/Generative AI (Ml, Dl, etc)/Natural Language Processing/Project-Spam mail Prediction/vectorizer.pkl','rb'))

# creating UI
st.title("Email Spam Detector")
st.write("Enter an email message below to see if its Spam or Ham")

# Getting user input
input_mail = st.text_area("Paste the email content here:", height = 200)

if st.button("Predict"):
    if input_mail:
        data = cv.transform([input_mail])
        prediction = model.predict(data)
        
        if prediction[0]==1:
            st.success("This is a **HAM** Mail !")
        else:
            st.error("This is a **SPAM** Mail !")
            
    else:
        st.warning("Please enter some text.")