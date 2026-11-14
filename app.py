import streamlit as st
import pickle

# Load the saved vectorizer and machine learning model
try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('spam_model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Make sure 'vectorizer.pkl' and 'spam_model.pkl' are in the same folder as app.py.")

st.title("📧 Spam Mail Detection App")
st.write("Enter an email or text message below to check whether it is **Spam** or **Not Spam (Ham)**.")

# User input text box
input_message = st.text_area("Enter message here:")

if st.button("Predict"):
    if input_message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        # 1. Transform the input text using the TF-IDF vectorizer
        transformed_input = vectorizer.transform([input_message])
        
        # 2. Predict using the trained machine learning model
        prediction = model.predict(transformed_input)[0]
        
        # 3. Display the result
        if prediction == 1 or prediction == 'spam':
            st.error("🚨 This is a **SPAM** message!")
        else:
            st.success("✅ This is **NOT SPAM (Ham)**.")
