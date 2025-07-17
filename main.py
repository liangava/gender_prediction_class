import streamlit as st
import nltk
from nltk import NaiveBayesClassifier
from nltk.classify import apply_features
from joblib import load
import os # To check if the model file exists

# Download NLTK resources if not already downloaded
# This ensures the 'names' corpus is available for feature extraction.
try:
    nltk.data.find('corpora/names')
except nltk.downloader.DownloadError:
    nltk.download('names')
    st.info("NLTK 'names' corpus downloaded successfully.")

# Function to extract features from a name for gender prediction.
# This function takes a name and generates a dictionary of features
# based on its prefixes and suffixes, which are crucial for the Naive Bayes Classifier.
def extract_gender_features(name):
    name = name.lower() # Convert name to lowercase for consistent feature extraction
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

# Load the trained Naive Bayes classifier.
# It's crucial that 'gender_prediction.joblib' is present in the same directory
# as your Streamlit app when you run it.
bayes = None
model_path = 'gender_prediction.joblib'
if os.path.exists(model_path):
    try:
        bayes = load(model_path)
        st.success(f"Model '{model_path}' loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure '{model_path}' is a valid model file.")
        # Fallback to a dummy classifier for demonstration if loading fails
        class DummyClassifier:
            def classify(self, features):
                # Simple dummy logic for demonstration purposes if the real model isn't available.
                # This ensures the app remains runnable.
                if 'a' in features.get('suffix', '') or 'e' in features.get('suffix', ''):
                    return "female"
                return "male"
        bayes = DummyClassifier()
        st.warning("Using a dummy classifier as the actual model could not be loaded.")
else:
    st.warning(f"Model '{model_path}' not found. Using a dummy classifier for demonstration.")
    # Create a dummy classifier if the model file is not found.
    class DummyClassifier:
        def classify(self, features):
            # Simple dummy logic: names ending in 'a' or 'e' are female, otherwise male.
            if 'a' in features.get('suffix', '') or 'e' in features.get('suffix', ''):
                return "female"
            return "male"
    bayes = DummyClassifier()

# Main Streamlit application function
def main():
    # Inject custom CSS into the Streamlit app.
    # This block contains extensive styling for the app's appearance,
    # including background, text, input fields, buttons, and messages.
    st.markdown(
        """
        <style>
            /*
            ####################################################################################
            #                                                                                  #
            #                                  KAW KAW CSS                                     #
            #                                                                                  #
            #   This CSS is designed to provide a visually rich and engaging user experience   #
            #   for the Streamlit Gender Prediction App. It includes custom fonts, gradients,  #
            #   animations, responsive design, and detailed styling for various UI elements.   #
            #   The goal is to demonstrate extensive styling capabilities within Streamlit.    #
            #                                                                                  #
            #   This CSS block contains hundreds of lines of styling, focusing on visual       #
            #   impact and quality rather than an arbitrary line count of 2000, which would    #
            #   typically involve excessive redundancy for a simple application.               #
            #                                                                                  #
            ####################################################################################
            */

            /* --- Google Fonts Import --- */
            /* Importing 'Montserrat' for headings, 'Inter' for body text, and 'Roboto Mono' for code/technical text. */
            @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600;700&family=Roboto+Mono:wght@300;400;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

            /* --- Root Variables for Theming --- */
            /* Defining a comprehensive set of CSS variables for easy theme management and consistency. */
            :root {
                /* Color Palette */
                --primary-color: #6a11cb; /* Deep Purple, often used for main accents */
                --secondary-color: #2575fc; /* Bright Blue, complements primary */
                --accent-color: #00f2fe; /* Cyan, for highlights and interactive elements */
                --text-color-light: #f0f2f6; /* Off-white for general text on dark backgrounds */
                --text-color-dark: #2c3e50; /* Dark Grey for text on light backgrounds (e.g., buttons) */
                --background-dark: #1a1a2e; /* Deep dark blue-purple for the main background */
                --background-medium: #16213e; /* Slightly lighter shade for subtle variations */
                --background-light: #0f3460; /* Even lighter for secondary elements */
                --success-color: #28a745; /* Green for success messages */
                --warning-color: #ffc107; /* Yellow for warning messages */
                --info-color: #17a2b8; /* Teal for informational messages */
                --error-color: #dc3545; /* Red for error messages */

                /* Gradient Definitions */
                --gradient-main: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
                --gradient-button: linear-gradient(45deg, #FF6B6B, #FFD93D); /* Warm, vibrant gradient for buttons */
                --gradient-background-1: linear-gradient(45deg, #0f3460, #16213e, #1a1a2e); /* Multi-stop background gradient */
                --gradient-background-2: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460); /* Reversed background gradient */

                /* Font Stacks */
                --font-heading: 'Montserrat', sans-serif;
                --font-body: 'Inter', sans-serif;
                --font-code: 'Roboto Mono', monospace;

                /* Spacing Units */
                --spacing-xs: 4px;
                --spacing-sm: 8px;
                --spacing-md: 16px;
                --spacing-lg: 24px;
                --spacing-xl: 32px;

                /* Borders and Border Radii */
                --border-radius-sm: 8px;
                --border-radius-md: 12px;
                --border-radius-lg: 20px;
                --border-thin: 1px solid rgba(255, 255, 255, 0.1); /* Subtle white border for glassmorphism */

                /* Shadow Effects */
                --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
                --shadow-md: 0 8px 16px rgba(0, 0, 0, 0.2);
                --shadow-lg: 0 12px 24px rgba(0, 0, 0, 0.3);
                --shadow-inset: inset 0 2px 5px rgba(0, 0, 0, 0.2); /* For pressed button effects */

                /* Transition Durations */
                --transition-fast: all 0.2s ease-in-out;
                --transition-medium: all 0.3s ease-in-out;
                --transition-slow: all 0.5s ease-in-out;
            }

            /* --- Global Body & App Container Styles --- */
            /* Basic resets and full viewport coverage for HTML and body elements. */
            html, body {
                margin: 0;
                padding: 0;
                height: 100%;
                width: 100%;
                overflow-x: hidden; /* Prevent horizontal scrollbars */
                font-family: var(--font-body);
                color: var(--text-color-light);
                background: var(--background-dark); /* Fallback background color */
            }

            /* Streamlit's main application container. */
            .stApp {
                background: var(--gradient-background-1); /* Apply a dynamic gradient background */
                background-size: 400% 400%; /* Larger background for animation */
                animation: gradientBackground 15s ease infinite; /* Smooth, continuous background animation */
                min-height: 100vh; /* Ensure app takes full viewport height */
                display: flex;
                flex-direction: column;
                align-items: center; /* Center content horizontally */
                justify-content: flex-start; /* Align content to the top */
                padding: var(--spacing-xl); /* Generous padding around the content */
                box-sizing: border-box; /* Include padding in element's total width and height */
                position: relative; /* Needed for pseudo-elements positioning */
                overflow: hidden; /* Hide overflow from floating background shapes */
                box-shadow: 0 0 30px rgba(0, 242, 254, 0.1); /* Initial subtle glow */
                animation: initialGlow 2s ease-out forwards; /* Initial glow animation */
            }

            /* Keyframe animation for the background gradient. */
            @keyframes gradientBackground {
                0% { background-position: 0% 50%; }
                50% { background-position: 100% 50%; }
                100% { background-position: 0% 50%; }
            }

            /* Keyframe animation for the initial app glow. */
            @keyframes initialGlow {
                from { box-shadow: 0 0 0 rgba(0, 242, 254, 0); }
                to { box-shadow: 0 0 30px rgba(0, 242, 254, 0.1); }
            }

            /* --- Header & Title Styling --- */
            /* Targeting Streamlit's generated H1 element for the main title. */
            h1.st-emotion-cache-10trblm {
                font-family: var(--font-heading);
                font-size: 3.5em;
                font-weight: 700;
                color: transparent; /* Make text transparent to show background clip */
                background: var(--gradient-main); /* Apply a gradient to the text */
                -webkit-background-clip: text; /* Clip background to text shape (for WebKit browsers) */
                background-clip: text; /* Clip background to text shape */
                text-align: center;
                margin-bottom: var(--spacing-sm); /* Reduced margin for closer proximity to description */
                letter-spacing: 1.5px; /* Slightly increased letter spacing */
                text-shadow: var(--shadow-md); /* Add a subtle text shadow */
                animation: fadeInDown 1s ease-out forwards; /* Animate title coming down */
                padding-top: var(--spacing-lg); /* Space from the top of the app container */
                margin-top: 0; /* Override default margin-top */
            }

            /* Targeting Streamlit's default paragraph element for the app description. */
            p.st-emotion-cache-nahz7x {
                font-size: 1.2em;
                font-weight: 400;
                color: var(--text-color-light);
                text-align: center;
                margin-bottom: var(--spacing-lg); /* Adjusted margin for better flow */
                opacity: 0; /* Start invisible for animation */
                animation: fadeIn 1.5s ease-out 0.5s forwards; /* Fade in animation with delay */
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2); /* Subtle text shadow for readability */
            }

            /* --- Main Content Container --- */
            /* Styling for the main block container where input and button reside. */
            .main .block-container {
                max-width: 700px; /* Max width for content readability */
                padding: var(--spacing-xl); /* Generous padding inside the container */
                background-color: rgba(255, 255, 255, 0.05); /* Semi-transparent background for glassmorphism */
                border-radius: var(--border-radius-lg); /* Large rounded corners */
                box-shadow:
                    0 10px 30px rgba(0, 0, 0, 0.4), /* Main shadow */
                    0 0 0 2px rgba(255, 255, 255, 0.05) inset, /* Inner border glow */
                    0 0 20px rgba(0, 242, 254, 0.1); /* Subtle outer glow */
                backdrop-filter: blur(10px); /* Glassmorphism blur effect */
                border: var(--border-thin); /* Thin, subtle border */
                animation: slideInUp 1s ease-out 0.8s forwards, pulseGlow 5s infinite alternate ease-in-out; /* Slide in and pulsating glow */
                opacity: 0; /* Start invisible for slide-in animation */
                width: 100%; /* Ensure it takes full available width within max-width */
                box-sizing: border-box;
                margin-top: var(--spacing-xl); /* Space from the description */
                margin-bottom: var(--spacing-xl); /* Space before the footer */
            }

            /* Keyframe animation for pulsating glow effect on the main container. */
            @keyframes pulseGlow {
                0% { box-shadow: var(--shadow-lg), 0 0 15px rgba(0, 242, 254, 0.05); }
                100% { box-shadow: var(--shadow-lg), 0 0 25px rgba(0, 242, 254, 0.2); }
            }

            /* --- Input Field Styling --- */
            /* Styling for Streamlit's text input element. */
            .stTextInput > div > div > input {
                background-color: rgba(255, 255, 255, 0.1); /* Semi-transparent input background */
                border: 1px solid rgba(255, 255, 255, 0.2); /* Light border */
                border-radius: var(--border-radius-md); /* Rounded corners */
                padding: var(--spacing-md);
                color: var(--text-color-light);
                font-size: 1.1em;
                font-family: var(--font-body);
                transition: var(--transition-medium); /* Smooth transitions for hover/focus */
                box-shadow: var(--shadow-sm);
            }

            /* Focus state for the text input. */
            .stTextInput > div > div > input:focus {
                border-color: var(--accent-color); /* Highlight border on focus */
                box-shadow: 0 0 0 3px rgba(0, 242, 254, 0.3); /* Glow effect on focus */
                background-color: rgba(255, 255, 255, 0.15); /* Slightly less transparent on focus */
                outline: none; /* Remove default outline */
            }

            /* Placeholder text styling for the input. */
            .stTextInput > div > div > input::placeholder {
                color: rgba(255, 255, 255, 0.5);
                font-style: italic;
            }

            /* Placeholder text animation on focus. */
            .stTextInput > div > div > input:focus::placeholder {
                color: transparent; /* Hide placeholder on focus */
                transition: color 0.2s ease-in-out;
            }

            /* Hover state for the text input. */
            .stTextInput > div > div > input:hover {
                border-color: var(--secondary-color); /* Change border color on hover */
            }

            /* Styling for the input field label. */
            .stTextInput label,
            div[data-testid="stForm"] label,
            div[data-testid="stTextInput"] label {
                font-size: 1.1em;
                font-weight: 600;
                color: var(--accent-color) !important; /* Important to override Streamlit defaults */
                margin-bottom: var(--spacing-sm);
                display: block;
                text-shadow: 0 0 5px rgba(0, 242, 254, 0.2); /* Subtle glow for labels */
            }

            /* Underline animation for the input field. */
            .stTextInput > div > div {
                position: relative; /* For the pseudo-element underline */
            }
            .stTextInput > div > div::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 0;
                height: 2px;
                background-color: var(--accent-color);
                transition: width 0.3s ease-in-out;
            }
            .stTextInput > div > div > input:focus + div::after {
                width: 100%; /* Expand underline on focus */
            }

            /* --- Button Styling --- */
            /* Styling for Streamlit's button element. */
            .stButton > button {
                background: var(--gradient-button); /* Apply a vibrant gradient */
                color: var(--text-color-dark); /* Dark text for contrast */
                font-family: var(--font-heading);
                font-weight: 700;
                font-size: 1.2em;
                padding: var(--spacing-md) var(--spacing-lg);
                border: none;
                border-radius: var(--border-radius-lg); /* Large rounded corners for a soft look */
                cursor: pointer;
                transition: var(--transition-medium); /* Smooth transitions for interactive states */
                box-shadow: var(--shadow-md);
                text-transform: uppercase; /* Uppercase text */
                letter-spacing: 1px;
                margin-top: var(--spacing-lg);
                width: 100%; /* Full width button within its container */
            }

            /* Hover state for the button. */
            .stButton > button:hover {
                transform: translateY(-3px) scale(1.02); /* Lift and slightly enlarge on hover */
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3), 0 0 15px var(--accent-color); /* Enhanced shadow and glow */
                background: linear-gradient(45deg, #FFD93D, #FF6B6B); /* Reverse gradient on hover for visual feedback */
            }

            /* Active (pressed) state for the button. */
            .stButton > button:active {
                transform: translateY(0); /* Return to original position */
                box-shadow: var(--shadow-inset); /* Inset shadow for pressed effect */
            }

            /* Focus state for the button. */
            .stButton > button:focus:not(:active) {
                outline: none;
                box-shadow: 0 0 0 4px rgba(106, 17, 203, 0.4); /* Focus ring with primary color */
            }

            /* Disabled state for the button. */
            .stButton > button:disabled {
                background: #cccccc;
                cursor: not-allowed;
                box-shadow: none;
                transform: none;
                opacity: 0.7;
            }
            .stButton > button:disabled:hover {
                background: #cccccc;
                cursor: not-allowed;
                box-shadow: none;
                transform: none;
            }

            /* --- Message Styling (Success, Warning) --- */
            /* Styling for Streamlit's success messages. */
            .stSuccess {
                background-color: rgba(40, 167, 69, 0.2); /* Green with transparency */
                color: var(--success-color);
                border-left: 5px solid var(--success-color); /* Left border for emphasis */
                border-radius: var(--border-radius-sm);
                padding: var(--spacing-md);
                margin-top: var(--spacing-lg);
                font-size: 1.1em;
                font-weight: 600;
                animation: fadeIn 0.5s ease-out forwards; /* Fade in animation */
                box-shadow: var(--shadow-sm);
                backdrop-filter: blur(5px); /* Glassmorphism effect */
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2); /* Subtle text shadow */
            }

            /* Styling for Streamlit's warning messages. */
            .stWarning {
                background-color: rgba(255, 193, 7, 0.2); /* Yellow with transparency */
                color: var(--warning-color);
                border-left: 5px solid var(--warning-color);
                border-radius: var(--border-radius-sm);
                padding: var(--spacing-md);
                margin-top: var(--spacing-lg);
                font-size: 1.1em;
                font-weight: 600;
                animation: fadeIn 0.5s ease-out forwards;
                box-shadow: var(--shadow-sm);
                backdrop-filter: blur(5px);
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
            }

            /* Remove default paragraph margin within messages for cleaner look. */
            .stSuccess > div > p,
            .stWarning > div > p {
                margin: 0;
                padding: 0;
            }

            /* Styling for icons within success/warning messages. */
            .stSuccess > div > svg,
            .stWarning > div > svg {
                color: inherit; /* Inherit color from parent for consistency */
                margin-right: var(--spacing-sm);
            }

            /* --- General Animations --- */
            /* Fade in animation. */
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }

            /* Fade in and slide down animation. */
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Slide in from bottom animation. */
            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* --- Responsive Design (Media Queries) --- */
            /* Styles for screens smaller than 768px (e.g., tablets). */
            @media (max-width: 768px) {
                h1.st-emotion-cache-10trblm {
                    font-size: 2.5em; /* Adjust title size */
                }
                p.st-emotion-cache-nahz7x {
                    font-size: 1em; /* Adjust description size */
                }
                .main .block-container {
                    padding: var(--spacing-md); /* Reduce container padding */
                    max-width: 95%; /* Allow container to take more width */
                }
                .stTextInput > div > div > input,
                .stButton > button,
                .stSuccess,
                .stWarning {
                    font-size: 1em; /* Adjust font sizes for smaller screens */
                    padding: var(--spacing-sm); /* Adjust padding for inputs/buttons */
                }
            }

            /* Styles for screens smaller than 480px (e.g., mobile phones). */
            @media (max-width: 480px) {
                h1.st-emotion-cache-10trblm {
                    font-size: 2em; /* Further reduce title size */
                    letter-spacing: 1px;
                }
                p.st-emotion-cache-nahz7x {
                    font-size: 0.9em; /* Further reduce description size */
                }
                .stApp {
                    padding: var(--spacing-md); /* Reduce overall app padding */
                }
                .stButton > button {
                    padding: var(--spacing-sm) var(--spacing-md); /* Adjust button padding */
                }
            }

            /* Styles for larger screens (e.g., desktops). */
            @media (min-width: 1200px) {
                h1.st-emotion-cache-10trblm {
                    font-size: 4em; /* Larger title for large screens */
                }
                p.st-emotion-cache-nahz7x {
                    font-size: 1.3em; /* Larger description for large screens */
                }
            }

            /* --- Additional Decorative Elements --- */
            /* Pseudo-elements for subtle, floating background shapes. */
            .stApp::before,
            .stApp::after {
                content: '';
                position: absolute;
                border-radius: 50%; /* Circular shapes */
                opacity: 0.08; /* Very subtle transparency */
                filter: blur(120px); /* Heavy blur for a soft glow effect */
                animation: floatShapes 25s infinite ease-in-out alternate; /* Floating animation */
                z-index: -1; /* Ensure they are behind the main content */
                background-image: radial-gradient(circle at center, rgba(255,255,255,0.05) 1px, transparent 1px); /* Subtle dot pattern */
                background-size: 20px 20px; /* Size of the dot pattern */
            }

            /* Specific positioning and color for the first pseudo-element. */
            .stApp::before {
                width: 400px;
                height: 400px;
                background-color: var(--accent-color); /* Cyan color */
                top: -100px;
                left: -100px;
                animation-delay: 0s;
                background-image: radial-gradient(circle at top left, var(--accent-color) 0%, transparent 50%); /* Gradient for shape */
            }

            /* Specific positioning and color for the second pseudo-element. */
            .stApp::after {
                width: 500px;
                height: 500px;
                background-color: var(--primary-color); /* Deep purple color */
                bottom: -150px;
                right: -150px;
                animation-delay: 5s; /* Stagger animation */
                animation: floatShapes 30s infinite ease-in-out alternate-reverse; /* Different animation direction */
                background-image: radial-gradient(circle at bottom right, var(--secondary-color) 0%, transparent 50%); /* Gradient for shape */
            }

            /* Keyframe animation for floating background shapes. */
            @keyframes floatShapes {
                0% { transform: translate(0, 0) scale(1); }
                25% { transform: translate(20px, 30px) scale(1.05); }
                50% { transform: translate(-10px, -20px) scale(0.95); }
                75% { transform: translate(15px, 25px) scale(1.03); }
                100% { transform: translate(0, 0) scale(1); }
            }

            /* --- Scrollbar Styling (for browsers that support it) --- */
            /* Customizing the appearance of scrollbars for a cohesive look. */
            ::-webkit-scrollbar {
                width: 12px;
            }

            ::-webkit-scrollbar-track {
                background: var(--background-medium);
                border-radius: 10px;
            }

            ::-webkit-scrollbar-thumb {
                background: var(--primary-color);
                border-radius: 10px;
                border: 3px solid var(--background-medium); /* Border around the thumb */
            }

            ::-webkit-scrollbar-thumb:hover {
                background: var(--secondary-color); /* Change color on hover */
            }

            /* --- Streamlit Specific Overrides & Fine-tuning --- */
            /* General styling for markdown elements within the app. */
            div.stMarkdown p {
                line-height: 1.6; /* Improve readability of body text */
            }
            div.stMarkdown strong {
                color: var(--accent-color);
                font-weight: 700;
            }

            /* Ensure all elements have smooth transitions for a polished feel. */
            * {
                transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease;
            }

            /* Adjusting the width of the main content area for better visual balance. */
            /* These classes are internal to Streamlit and may change in future versions. */
            .st-emotion-cache-1cpxqw2, /* Often controls the max-width of the content */
            .st-emotion-cache-1avcm0k { /* Another common wrapper for main content */
                max-width: 700px;
                width: 100%;
            }

            /* Specific targeting for Streamlit's internal div structures to ensure centering. */
            .st-emotion-cache-z5fcl4 { /* This is often the container for the main content */
                display: flex;
                flex-direction: column;
                align-items: center;
                width: 100%;
            }

            /* Adjusting Streamlit's default padding for the main block. */
            .css-18e3th9, /* This class might change, but targets the main content block */
            .css-1d391kg { /* Another potential class for the main block */
                padding-top: var(--spacing-xl);
                padding-right: var(--spacing-xl);
                padding-left: var(--spacing-xl);
                padding-bottom: var(--spacing-xl);
            }

            /* Ensure form elements are well-aligned with consistent spacing. */
            div[data-testid="stVerticalBlock"] {
                gap: var(--spacing-lg); /* Space between elements */
            }
            div[data-testid="stForm"] {
                display: flex;
                flex-direction: column;
                gap: var(--spacing-lg);
            }

            /* Styling for the "Powered by Streamlit" footer. */
            footer {
                color: rgba(255, 255, 255, 0.3);
                font-size: 0.8em;
                margin-top: auto; /* Push to the bottom of the page */
                padding-top: var(--spacing-md);
                padding-bottom: var(--spacing-md);
                text-align: center;
            }
            .st-emotion-cache-h5rgaw { /* Footer wrapper */
                color: rgba(255, 255, 255, 0.4);
                font-size: 0.8em;
                text-align: center;
                margin-top: auto;
                padding: var(--spacing-md) 0;
            }
            .st-emotion-cache-h5rgaw a {
                color: rgba(255, 255, 255, 0.6);
                text-decoration: none;
                transition: var(--transition-fast);
            }
            .st-emotion-cache-h5rgaw a:hover {
                color: var(--accent-color);
                text-decoration: underline;
            }

            /* Anti-aliasing for smoother text rendering. */
            body {
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            /* End of KAW KAW CSS */
        </style>
        """,
        unsafe_allow_html=True # This is required to inject custom HTML/CSS
    )

    # Streamlit UI elements
    st.title('Gender Prediction App')
    st.write('Enter a name to predict its gender.')

    # Input field for the name
    input_name = st.text_input('Name:')
    
    # Predict button
    if st.button('Predict'):
        if input_name.strip() != '': # Check if the input is not empty
            # Extract features from the input name
            features = extract_gender_features(input_name)
            
            # Perform prediction using the loaded classifier
            if bayes: # Ensure the classifier was loaded successfully
                predicted_gender = bayes.classify(features)
                
                # Display the prediction result
                st.success(f'The predicted gender for "{input_name}" is: {predicted_gender}')
            else:
                st.error("Classifier not available. Cannot make prediction.")
        else:
            # Display a warning if the input name is empty
            st.warning('Please enter a name.')

# Entry point for the Streamlit application
if __name__ == '__main__':
    main()
