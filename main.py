import streamlit as st
import nltk
from joblib import load

# Download NLTK names corpus
nltk.download('names')

# Extract name features
def extract_gender_features(name):
    name = name.lower()
    return {
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

# Load model
bayes = load('gender_prediction.joblib')

# 🔥 KAW KAW CSS 🔥
def kaw_kaw_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Press+Start+2P&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(270deg, #0f0c29, #302b63, #24243e);
        background-size: 600% 600%;
        animation: gradientBG 10s ease infinite;
        color: #ffffff;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    .stTextInput > div > input {
        background-color: rgba(255, 255, 255, 0.1);
        border: 2px solid #00f0ff;
        border-radius: 10px;
        color: #fff;
        font-size: 1.2rem;
        padding: 12px;
    }

    .stButton > button {
        background: linear-gradient(45deg, #ff00cc, #3333ff);
        color: #fff;
        border: none;
        border-radius: 30px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        box-shadow: 0 0 15px #ff00cc;
        transition: 0.3s ease;
        font-family: 'Press Start 2P', cursive;
    }

    .stButton > button:hover {
        background: linear-gradient(45deg, #00ffff, #ff00cc);
        box-shadow: 0 0 25px #00ffff;
        transform: scale(1.08);
        cursor: pointer;
    }

    .stMarkdown h1 {
        font-size: 3rem;
        color: #00ffff;
        text-shadow: 0 0 15px #00ffff, 0 0 30px #ff00ff;
        text-align: center;
        margin-bottom: 2rem;
        animation: pulseGlow 2s infinite;
    }

    @keyframes pulseGlow {
        0% { text-shadow: 0 0 10px #00ffff; }
        50% { text-shadow: 0 0 20px #ff00ff; }
        100% { text-shadow: 0 0 10px #00ffff; }
    }

    .stSuccess {
        background-color: rgba(0, 255, 204, 0.1);
        border-left: 5px solid #00ffd5;
        font-size: 1.3rem;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 0 20px #00ffd5;
        text-align: center;
    }

    .stWarning {
        font-size: 1.2rem;
        color: #ffaaaa;
    }

    </style>
    """, unsafe_allow_html=True)

# 💥 THE APP
def main():
    kaw_kaw_css()

    st.markdown("<h1>⚡ KAW KAW Gender Predictor ⚡</h1>", unsafe_allow_html=True)
    st.write("🔥 Enter a name and witness the neon-powered AI magic.")

    input_name = st.text_input("🔤 Your Epic Name:", max_chars=30)

    if st.button("🚀 Predict Now"):
        if input_name.strip() != "":
            features = extract_gender_features(input_name)
            result = bayes.classify(features)
            emoji = "👦" if result == "male" else "👧"
            st.success(f'🔮 Predicted Gender for **"{input_name}"**: {emoji} **{result.upper()}**')
        else:
            st.warning("⚠️ Input cannot be empty. Type something legendary.")

if __name__ == '__main__':
    main()
