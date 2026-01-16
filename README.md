# 📧 Spam Mail Detection Web App

This project is a **Machine Learning–based Spam Mail Detection system** that classifies emails/messages as **Spam** or **Not Spam**.  
The model is deployed using **Streamlit** to provide an interactive web interface.

---

## 🚀 Features
- Classifies messages into **Spam / Ham**
- Clean and simple **Streamlit web UI**
- Uses **Machine Learning & NLP techniques**
- Fast predictions with pre-trained model
- Beginner-friendly and practical project

---

## 🧠 Machine Learning Workflow
1. Data Collection & Cleaning  
2. Text Preprocessing  
   - Lowercasing  
   - Removing special characters  
   - Stopword removal  
3. Feature Extraction using **TF-IDF Vectorizer**
4. Model Training using **Scikit-learn**
5. Model Serialization using **Pickle**
6. Deployment using **Streamlit**

---

## 🛠️ Tech Stack
- **Python**
- **Scikit-learn**
- **Pandas, NumPy**
- **Natural Language Processing (NLP)**
- **Streamlit**
- **Pickle**

---

## 📂 Project Structure
├── mail_data.csv
├── spam_detection.ipynb
├── vectorizer.pkl
├── spam_model.pkl
├── app.py
└── README.md

## 📊 Model Used
- **Machine Learning Classification Model**
- **TF-IDF Vectorization**
- Trained on labeled spam/ham dataset

---

## 🌐 Web App Interface
- User enters a message/email
- Clicks **Predict**
- Model outputs whether the message is **Spam or Not Spam**

## ▶️ How to Run the Project
- Install required libraries
     pip install -r requirements.txt
- Run the Streamlit app
     streamlit run app.py
- Open the browser link shown in the terminal and use the app.

## 👩‍💻 Author
**Shreya Varne**  
B.Tech CSE (3rd Year)
