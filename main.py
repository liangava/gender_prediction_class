import streamlit as st
import nltk
from joblib import load

nltk.download('names')

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

bayes = load('gender_prediction.joblib')

def load_kaw_kaw_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    html, body {
        margin: 0;
        padding: 0;
        height: 100%;
        background: black;
        overflow: hidden;
        font-family: 'Orbitron', sans-serif;
    }

    canvas {
        position: fixed;
        top: 0;
        left: 0;
        z-index: -1;
    }

    .main-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        padding: 30px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 0 30px #00ffff66;
        max-width: 500px;
        text-align: center;
        animation: fadeIn 2s ease-in-out;
    }

    h1.glitch {
        font-size: 3rem;
        color: #00ffff;
        text-shadow: 0 0 5px #00ffff, 0 0 10px #ff00ff;
        animation: glitch 1s infinite;
    }

    @keyframes glitch {
        0% { transform: skewX(0deg); }
        20% { transform: skewX(-10deg); }
        40% { transform: skewX(10deg); }
        60% { transform: skewX(-5deg); }
        80% { transform: skewX(5deg); }
        100% { transform: skewX(0deg); }
    }

    .stTextInput > div > input {
        background-color: #111;
        color: #00ffff;
        border: 2px solid #00ffff;
        border-radius: 10px;
        padding: 10px;
        font-size: 1rem;
        margin-bottom: 20px;
    }

    .stButton > button {
        background: linear-gradient(45deg, #00ffff, #ff00ff);
        border: none;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 0 10px #00ffff;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: scale(1.1);
        box-shadow: 0 0 20px #ff00ff;
        cursor: pointer;
    }

    .stSuccess {
        background-color: rgba(0, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #00ffff;
        margin-top: 20px;
        font-size: 1.3rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>

    <canvas id="stars"></canvas>
    <script>
    const canvas = document.getElementById('stars');
    const ctx = canvas.getContext('2d');
    let stars = [];
    let w, h;

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }

    function initStars() {
        stars = [];
        for (let i = 0; i < 100; i++) {
            stars.push({
                x: Math.random() * w,
                y: Math.random() * h,
                radius: Math.random() * 1.5,
                alpha: Math.random()
            });
        }
    }

    function drawStars() {
        ctx.clearRect(0, 0, w, h);
        for (let star of stars) {
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(0,255,255," + star.alpha + ")";
            ctx.fill();
        }
    }

    function animate() {
        drawStars();
        for (let star of stars) {
            star.y += 0.5;
            if (star.y > h) {
                star.y = 0;
                star.x = Math.random() * w;
            }
        }
        requestAnimationFrame(animate);
    }

    window.addEventListener('resize', () => {
        resize();
        initStars();
    });

    resize();
    initStars();
    animate();
    </script>
    """, unsafe_allow_html=True)

def main():
    load_kaw_kaw_css()

    st.markdown('<div class="main-container"><div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="glitch">⚡ Gender Predictor ⚡</h1>', unsafe_allow_html=True)
    st.write("Type a name and unleash the prediction magic 💫")

    name_input = st.text_input("🔤 Enter your name:")

    if st.button("🚀 PREDICT NOW"):
        if name_input.strip() != "":
            features = extract_gender_features(name_input)
            gender = bayes.classify(features)
            emoji = "👨" if gender == "male" else "👩"
            st.success(f'🔮 Predicted Gender: {emoji} **{gender.upper()}**')
        else:
            st.warning("⚠️ Please enter a name.")

    st.markdown('</div></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
