import streamlit as st
import PIL.Image
import google.generativeai as genai
from google.generativeai import types
# Set page configuration
st.set_page_config(page_title="📸 PersonaVision AI", layout="wide")

# Load API key
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# System Prompt for Image Analysis
def analyze_human_attributes(image):
    prompt = """  
You are an AI trained to analyze human attributes from images with high accuracy.  
Carefully analyze the given image and return the following structured details:  

You must provide results based solely on the visible image data. Avoid any assumptions beyond what is observable. Do not apologize or return empty results.  

🧑‍🤝‍🧑 Demographic & Facial Analysis:  
- Gender Expression: Male / Female / Non-binary  
- Age Estimate: Approximate age in years  
- Ethnicity: Based on visible features  
- Facial Structure: Notable characteristics  
- Skin Tone: Light, Medium, Dark, specific undertones  
- Eye Shape: Almond, Round, Hooded, etc.  

😀 Emotional & Facial Expression Analysis:  
- Primary Mood: Happy, Sad, Neutral, Excited, etc.  
- Facial Expression: Smiling, Frowning, Neutral, Raised Eyebrows  
- Emotions Detected: Joyful, Focused, Angry, Surprised, Anxious  
- Confidence Level: Accuracy of prediction in percentage  

👕 Clothing & Fashion Details:  
- Top Type, Bottom Type, Color of Clothing  
- Style & Formality: Casual, Formal, Sportswear, Traditional  
- Brand Logos: Detect if any visible  
- Seasonal Clothing: Identify if suitable for warm, cold, or rainy weather  

🕶️ Accessories & Appearance Enhancements:  
- Glasses, Jewelry, Beard & Facial Hair, Makeup, Headwear  

💇 Hair & Facial Features:  
- Hair Length, Hair Type, Hair Color  
- Facial Hair: Yes/No (Specify type)  
- Eye Color: Blue, Green, Brown, Hazel, Gray  

📍 Environmental & Background Context:  
- Indoor or Outdoor Setting, Weather Condition, Lighting Condition  
- Objects in Background, People in Background  
"""  
    response = model.generate_content([prompt, image])
    return response.text.strip()

# 🎨 Custom Animated UI Styling
st.markdown("""
    <style>
        /* Title Animation */
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .title-container {
            text-align: center;
            font-size: 42px;
            font-weight: bold;
            background: linear-gradient(90deg, #4A90E2, #1E3A8A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding: 10px 0;
            animation: slideIn 1s ease-in-out;
        }

        .subtitle-container {
            text-align: center;
            font-size: 18px;
            color: #444;
            font-weight: 500;
            margin-bottom: 25px;
            animation: slideIn 1.2s ease-in-out;
        }

        /* Upload Box */
        @keyframes pulse {
            0% { transform: scale(1); box-shadow: 0px 0px 5px rgba(74, 144, 226, 0.2); }
            50% { transform: scale(1.02); box-shadow: 0px 0px 20px rgba(74, 144, 226, 0.3); }
            100% { transform: scale(1); box-shadow: 0px 0px 5px rgba(74, 144, 226, 0.2); }
        }

        .upload-box {
            background: #f1f5f9;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            border: 3px dashed #4A90E2;
            transition: all 0.3s ease-in-out;
            animation: pulse 1.5s infinite;
        }

        /* Animated Result Cards */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-container {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }

        .result-card {
            background: white;
            padding: 18px;
            border-radius: 12px;
            gap: 20px
            box-shadow: 4px 4px 15px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #4A90E2;
            display:flex;
            width: 45%;
            min-width: 300px;
            animation: fadeIn 1s ease-in-out;
        }

        .result-title {
            font-size: 18px;
            font-weight: bold;
            color: #1E3A8A;
            margin-bottom: 5px;
        }

        .result-text {
            margin: 12px;
            font-size: 16px;
            color: #333;
            font-weight: 500;
        }
         /* Adjusts the spacing between columns */
        .stColumn {
            padding-right: 40px !important; /* Adds space between col1 and col2 */
        
        }
    </style>
""", unsafe_allow_html=True)

# UI Header
st.markdown("<div class='title-container'>📸 PersonaVision AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-container'🔍 Unlock Insights from Images with AI-Powered Human Attribute Analysis</div>", unsafe_allow_html=True)

# Image Upload Section
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
uploaded_image = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# If an image is uploaded, analyze it
if uploaded_image:
    img = PIL.Image.open(uploaded_image)

    # Display image & results side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    with col2:
        st.markdown("<div class='result-container'>", unsafe_allow_html=True)
        st.subheader("📊 Analysis Results:")

        with st.spinner("🧐 Analyzing image attributes..."):
            analysis_result = analyze_human_attributes(img)

        # Split response into sections & display in cards
        sections = analysis_result.split("\n\n")  
        for section in sections:
            if ":" in section:
                title, content = section.split(":", 1)
                st.markdown(f"""
                    <div class='result-card'>
                        <div class='result-title'>{title.strip()}</div>
                        <div class='result-text'>{content.strip()}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔍 Powered by Gemini AI | AI-powered Human Image Attribute Analyzer</p>
        <p style='margin-top: 0.5rem;'>Developed with ❤️ by Fatima Nazeer</p>
    </div>
""", unsafe_allow_html=True)