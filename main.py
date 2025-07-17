import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from joblib import load

# Download NLTK resources if not already downloaded
nltk.download('names')

# Function to extract features from a name
def extract_gender_features(name):
    name = name.lower()
    features = {
        "suffix": name[-1:],
        "suffix2": name[-2:] if len(name) > 1 else name[0],
        "suffix3": name[-3:] if len(name) > 2 else name[0],
        "suffix4": name[-4:] if len(name) > 3 else name[0],
        "suffix5": name[-5:] if len(name) > 4 else name[0],
        "suffix6": name[-6:] if len(name) > 5 else name[0],
        "prefix": name[:1],
        "prefix2": name[:2] if len(name) > 1 else name[0],
        "prefix3": name[:3] if len(name) > 2 else name[0],
        "prefix4": name[:4] if len(name) > 3 else name[0],
        "prefix5": name[:5] if len(name) > 4 else name[0]
    }
    return features

# Load the trained classifier
bayes = load('gender_prediction.joblib')

# Inject custom CSS for KAW KAW style
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(to right, #141e30, #243b55);
        color: #f5f5f5;
    }

    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .stTextInput > div > input {
        background-color: #1e1e2f;
        color: white;
        border: 1px solid #00ffd5;
        border-radius: 10px;
        padding: 10px;
    }

    .stButton > button {
        background-color: #00ffd5;
        color: black;
        border-radius: 10px;
        font-weight: bold;
        transition: 0.3s ease;
        padding: 0.75rem 1.5rem;
    }

    .stButton > button:hover {
        background-color: #0ff;
        box-shadow: 0 0 20px #00ffd5;
        transform: scale(1.05);
    }

    .stMarkdown h1 {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #00ffd5, #ff00c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 1.5s ease-in-out infinite alternate;
        text-align: center;
    }

    @keyframes glow {
        from {
            text-shadow: 0 0 10px #00ffd5;
        }
        to {
            text-shadow: 0 0 20px #ff00c8;
        }
    }

    .stSuccess {
        font-size: 1.25rem;
        background-color: rgba(0, 255, 213, 0.1);
        padding: 1rem;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit App
def main():
    load_custom_css()

    st.markdown("<h1>Gender Prediction App 🚻</h1>", unsafe_allow_html=True)
    st.write('✨ Type a name below and let the AI predict whether it sounds more masculine or feminine!')

    input_name = st.text_input('Enter a name:', max_chars=30)

    if st.button('🎯 Predict Gender'):
        if input_name.strip() != '':
            features = extract_gender_features(input_name)
            predicted_gender = bayes.classify(features)
            st.success(f'The predicted gender for **"{input_name}"** is: 🧠 **{predicted_gender.upper()}**')
        else:
            st.warning('⚠️ Please enter a name.')

if __name__ == '__main__':
    main()
