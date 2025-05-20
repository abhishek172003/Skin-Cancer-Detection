import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import json
import os
import io
from pathlib import Path
import tensorflow as tf
import base64
from io import BytesIO
import pandas as pd
import time
import requests
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your environment variables.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model with the newer version
model = genai.GenerativeModel('gemini-1.5-flash')

# Page configuration
st.set_page_config(
    page_title="AI-Powered Skin Cancer Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced styling and animations
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        
        /* Global Styles */
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { transform: translateX(-100%); }
            to { transform: translateX(0); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* Progress Bar Animation */
        @keyframes progressAnimation {
            0% { width: 0%; }
            100% { width: 100%; }
        }
        
        .progress-container {
            width: 100%;
            height: 20px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-bar {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, #66B2B2, #2E4057, #66B2B2);
            background-size: 200% 100%;
            animation: progressAnimation 4s linear forwards,
                       gradientBG 2s ease infinite;
            border-radius: 10px;
        }
        
        /* Animated Elements */
        .main-title {
            animation: fadeIn 1s ease-out;
            background: linear-gradient(45deg, #2E4057, #66B2B2, #2E4057);
            background-size: 200% 200%;
            animation: gradientBG 5s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
            margin-bottom: 2rem;
            text-align: center;
            font-size: 3.5em !important;
            padding: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .upload-box {
            animation: slideIn 1s ease-out;
            border: 3px dashed #66B2B2;
            padding: 3rem;
            text-align: center;
            border-radius: 15px;
            background: rgba(102, 178, 178, 0.1);
            transition: all 0.3s ease;
            margin: 2rem 0;
            cursor: pointer;
        }
        
        .upload-box:hover {
            border-color: #2E4057;
            background: rgba(46, 64, 87, 0.1);
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }
        
        .upload-icon {
            font-size: 48px;
            color: #66B2B2;
            margin-bottom: 1rem;
        }
        
        .upload-text {
            font-size: 1.2em;
            color: #2E4057;
            margin-bottom: 0.5rem;
        }
        
        .upload-subtext {
            font-size: 0.9em;
            color: #666;
        }
        
        .results-box {
            animation: fadeIn 1s ease-out;
            background: linear-gradient(135deg, #2E4057 0%, #1F4068 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            margin: 1rem 0;
            border-left: 5px solid #66B2B2;
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
        }
        
        .history-section {
            animation: fadeIn 1s ease-out;
            margin-top: 3rem;
            padding: 2rem;
            background: linear-gradient(135deg, #1F4068 0%, #2E4057 100%);
            border-radius: 15px;
            color: white;
        }
        
        .history-title {
            color: white;
            font-size: 1.5em;
            margin-bottom: 1rem;
            border-bottom: 2px solid #66B2B2;
            padding-bottom: 0.5rem;
        }
        
        /* Button Styles */
        .stButton>button {
            width: 100%;
            margin-top: 10px;
            background: linear-gradient(135deg, #66B2B2 0%, #2E4057 100%);
            color: white;
            border: none;
            padding: 0.8rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(102, 178, 178, 0.4);
        }
        
        /* Dataframe Styling */
        .dataframe {
            animation: fadeIn 1s ease-out;
            border-radius: 10px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.05) !important;
            backdrop-filter: blur(4px);
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #1F4068;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #66B2B2;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #2E4057;
        }
        
        /* Startup Animation */
        .startup-animation {
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, #1F4068 0%, #2E4057 100%);
            border-radius: 15px;
            color: white;
            margin: 2rem auto;
            max-width: 800px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .startup-title {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #66B2B2, #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .startup-text {
            font-size: 1.5em;
            margin: 1rem 0;
            opacity: 0;
            animation: fadeIn 1s ease-out forwards;
        }
        
        /* Login Form Styling */
        .login-container {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background: white;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        }
        
        .login-title {
            text-align: center;
            color: #2E4057;
            margin-bottom: 1.5rem;
        }
        
        .login-input {
            width: 100%;
            padding: 0.8rem;
            margin-bottom: 1rem;
            border: 2px solid #66B2B2;
            border-radius: 8px;
            font-size: 1rem;
        }
        
        .login-button {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #66B2B2 0%, #2E4057 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .login-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(102, 178, 178, 0.4);
        }
        
        /* User Info Display */
        .user-info {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 1rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        
        .user-name {
            font-weight: 500;
            color: #2E4057;
        }
        
        .prediction-count {
            font-size: 0.9rem;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize session state variables
if 'user_logged_in' not in st.session_state:
    st.session_state.user_logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'prediction_count' not in st.session_state:
    st.session_state.prediction_count = 0
if 'user_history' not in st.session_state:
    st.session_state.user_history = {}
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'startup_animation' not in st.session_state:
    st.session_state.startup_animation = True
if 'doctors_list' not in st.session_state:
    st.session_state.doctors_list = []

# Login/Logout functions
def login():
    st.session_state.user_logged_in = True
    st.session_state.prediction_count = st.session_state.user_history.get(st.session_state.username, {}).get('predictions', 0)

def logout():
    st.session_state.user_logged_in = False
    st.session_state.username = None
    st.session_state.prediction_count = 0
    st.session_state.current_analysis = None
    st.session_state.last_uploaded_file = None

# Model loading function
@st.cache_resource
def load_model():
    img_width, img_height = 71, 71
    
    model = tf.keras.Sequential([
        tf.keras.applications.Xception(
            weights='imagenet',
            include_top=False,
            input_shape=(img_width, img_height, 3)
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    
    model.load_weights('model_xcpetion.h5')
    return model

def process_image(image):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (71, 71))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch


# Define urgency colors
urgency_colors = {
    'Critical': '#ff4444',
    'High': '#ffbb33',
    'Moderate': '#ffeb3b',
    'Low': '#00C851'
}

# Cancer information dictionary
cancer_info = {
    'akiec': {
        'name': 'Actinic Keratosis',
        'is_cancerous': 'Pre-cancerous',
        'info': 'Pre-cancerous growth caused by sun damage. While not cancer yet, it has potential to develop into skin cancer if left untreated.',
        'urgency': 'Moderate',
        'confidence_explanation': 'Confidence score indicates how certain the AI is about identifying this specific type of pre-cancerous growth.',
        'urgency_detail': 'Should be evaluated by a dermatologist within 1-2 months. Early treatment can prevent progression to cancer.',
        'treatment_options': 'Common treatments include cryotherapy (freezing), topical medications, photodynamic therapy, or surgical removal.',
        'prevention': 'Use broad-spectrum sunscreen, wear protective clothing, avoid peak sun hours (10am-4pm).',
        'risk_factors': 'Fair skin, sun exposure, age over 40, weakened immune system.',
        'resources': [
            ('Skin Cancer Foundation', 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis'),
            ('Mayo Clinic', 'https://www.mayoclinic.org/diseases-conditions/actinic-keratosis/symptoms-causes/syc-20354969')
        ]
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'is_cancerous': 'Cancerous',
        'info': 'Most common form of skin cancer. While serious, it rarely spreads beyond the original site.',
        'urgency': 'High',
        'confidence_explanation': 'Confidence score reflects the AI\'s certainty in identifying this common form of skin cancer.',
        'urgency_detail': 'Requires treatment within 1 month. While slow-growing, early treatment leads to better outcomes.',
        'treatment_options': 'Surgical excision, Mohs surgery, radiation therapy, or topical medications.',
        'prevention': 'Regular skin checks, sun protection, avoid tanning beds.',
        'risk_factors': 'Sun exposure, fair skin, age, radiation therapy history.',
        'resources': [
            ('American Cancer Society', 'https://www.cancer.org/cancer/basal-cell-skin-cancer.html'),
            ('National Cancer Institute', 'https://www.cancer.gov/types/skin')
        ]
    },
    'bkl': {
        'name': 'Benign Keratosis',
        'is_cancerous': 'Non-Cancerous',
        'info': 'A benign skin condition that is generally harmless.',
        'urgency': 'Low',
        'confidence_explanation': 'Confidence score reflects the AI\'s certainty in identifying this non-cancerous condition.',
        'urgency_detail': 'No immediate medical attention required, but monitoring for changes is advised.',
        'treatment_options': 'Generally requires no treatment but can be removed for cosmetic reasons.',
        'prevention': 'Regular skin checks and sun protection.',
        'risk_factors': 'Aging, genetics, prolonged sun exposure.',
        'resources': [
            ('DermNet NZ', 'https://dermnetnz.org/topics/seborrhoeic-keratosis')
        ]
    },
    'mel': {
        'name': 'Melanoma',
        'is_cancerous': 'Cancerous',
        'info': 'A serious form of skin cancer that requires immediate medical attention.',
        'urgency': 'Critical',
        'confidence_explanation': 'Confidence score reflects the AI\'s certainty in identifying this aggressive skin cancer.',
        'urgency_detail': 'Seek immediate medical evaluation for further tests and treatment.',
        'treatment_options': 'Surgical removal, chemotherapy, immunotherapy, targeted therapy.',
        'prevention': 'Avoid excessive sun exposure, wear sunscreen, monitor skin for new or changing moles.',
        'risk_factors': 'Fair skin, excessive UV exposure, family history of melanoma.',
        'resources': [
            ('Skin Cancer Foundation', 'https://www.skincancer.org/skin-cancer-information/melanoma')
        ]
    },
    'df': {
        'name': 'Dermatofibroma',
        'is_cancerous': 'Non-Cancerous',
        'info': 'A benign skin nodule that is usually harmless.',
        'urgency': 'Low',
        'confidence_explanation': 'Confidence score indicates the AI\'s certainty in identifying this condition.',
        'urgency_detail': 'No immediate action needed, but monitor for any changes.',
        'treatment_options': 'Generally requires no treatment but can be removed if bothersome.',
        'prevention': 'No specific prevention measures required.',
        'risk_factors': 'Unknown, but commonly occurs in adults.',
        'resources': [
            ('DermNet NZ', 'https://dermnetnz.org/topics/dermatofibroma')
        ]
    },
    'nv': {
        'name': 'Melanocytic Nevus',
        'is_cancerous': 'Non-Cancerous',
        'info': 'Commonly known as a mole, usually harmless.',
        'urgency': 'Low',
        'confidence_explanation': 'Confidence score reflects how certain the AI is in detecting this common mole.',
        'urgency_detail': 'Monitor for any changes in shape, size, or color.',
        'treatment_options': 'No treatment required unless changes occur.',
        'prevention': 'Sun protection and regular skin checks.',
        'risk_factors': 'Genetics, sun exposure.',
        'resources': [
            ('AAD', 'https://www.aad.org/public/diseases/a-z/moles')
        ]
    },
    'vasc': {
        'name': 'Vascular Lesion',
        'is_cancerous': 'Non-Cancerous',
        'info': 'A benign overgrowth of blood vessels in the skin.',
        'urgency': 'Low',
        'confidence_explanation': 'Confidence score reflects the AI\'s certainty in detecting this condition.',
        'urgency_detail': 'Generally harmless but can be removed for cosmetic reasons.',
        'treatment_options': 'Laser therapy, surgical removal if necessary.',
        'prevention': 'No known prevention methods.',
        'risk_factors': 'Genetics, aging.',
        'resources': [
            ('DermNet NZ', 'https://dermnetnz.org/topics/vascular-lesions')
        ]
    }
}

def generate_comprehensive_pdf_report(username, history, prediction_count):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Comprehensive Skin Cancer Detection Report", title_style))
    story.append(Spacer(1, 12))
    
    # User Information
    story.append(Paragraph(f"User: {username}", styles['Heading2']))
    story.append(Paragraph(f"Total Predictions Made: {prediction_count}", styles['Normal']))
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Predictions History
    story.append(Paragraph("Prediction History", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    for i, result in enumerate(history, 1):
        detected_type = cancer_info[result['type']]
        story.append(Paragraph(f"Prediction #{i}", styles['Heading3']))
        story.append(Paragraph(f"Date: {result['date']}", styles['Normal']))
        story.append(Paragraph(f"Type: {detected_type['name']}", styles['Normal']))
        story.append(Paragraph(f"Classification: {detected_type['is_cancerous']}", styles['Normal']))
        story.append(Paragraph(f"Confidence: {result['confidence']:.2f}%", styles['Normal']))
        story.append(Paragraph(f"Urgency Level: {detected_type['urgency']}", styles['Normal']))
        story.append(Paragraph("Medical Information:", styles['Normal']))
        story.append(Paragraph(f"• {detected_type['info']}", styles['Normal']))
        story.append(Paragraph("Recommended Action:", styles['Normal']))
        story.append(Paragraph(f"• {detected_type['urgency_detail']}", styles['Normal']))
        story.append(Spacer(1, 12))
    
    # General Recommendations
    story.append(Paragraph("General Recommendations", styles['Heading2']))
    recommendations = [
        "Regular skin self-examinations",
        "Annual professional skin cancer screenings",
        "Use broad-spectrum sunscreen daily",
        "Avoid peak sun hours (10am-4pm)",
        "Wear protective clothing and hats",
        "Monitor any changes in skin lesions"
    ]
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def validate_dermoscopic_image(image):
    """
    Validates if the uploaded image is a dermoscopic skin lesion image using Google Gemini.
    Returns a tuple of (is_valid, message)
    """
    try:
        # Convert PIL Image to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create a PIL Image object from bytes
        img = Image.open(BytesIO(img_byte_arr))
        
        # Prepare the prompt for Gemini
        prompt = """
        Analyze this image and determine if it is a dermoscopic skin lesion image. 
        A valid dermoscopic image should:
        1. Show a close-up view of a skin lesion
        2. Have good lighting and focus
        3. Not contain faces, text, or other non-lesion elements
        4. Be taken with a dermatoscope or similar medical imaging device
        
        Respond with only 'VALID' if it meets these criteria, or 'INVALID' followed by a brief reason if it doesn't.
        """
        
        # Get response from Gemini
        response = model.generate_content([prompt, img])
        response_text = response.text.strip().upper()
        
        if response_text.startswith('VALID'):
            return True, "Image is a valid dermoscopic skin lesion image."
        else:
            return False, f"Invalid image: {response_text.replace('INVALID', '').strip()}"
            
    except Exception as e:
        return False, f"Error during validation: {str(e)}"

def main():
    local_css()
    
    # Startup animation
    if st.session_state.startup_animation:
        st.markdown("""
            <div class="startup-animation">
                <h1 class="startup-title">AI-Powered Skin Cancer Detection System</h1>
                <div class="startup-text">Initializing Advanced Neural Networks...</div>
                <div class="progress-container">
                    <div class="progress-bar"></div>
                </div>
                <div class="startup-text" style="animation-delay: 1s">Loading Medical Analysis Models...</div>
                <div class="startup-text" style="animation-delay: 2s">Calibrating Detection Parameters...</div>
                <div class="startup-text" style="animation-delay: 3s">Preparing Diagnostic Tools...</div>
                <div class="startup-text" style="animation-delay: 4s">System Ready for Analysis!</div>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(5)
        st.session_state.startup_animation = False
        st.rerun()
    
    # Login/Registration Section with Consent Form
    if not st.session_state.user_logged_in:
        st.markdown('<h1 class="main-title">Welcome to AI-Powered Skin Cancer Detection System</h1>', unsafe_allow_html=True)
        
        # Add disclaimer box before login form
        st.markdown("""
            <div style='background-color: #f8d7da; border: 2px solid #dc3545; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #721c24; font-weight: bold;'>⚠️ Important Medical Disclaimer</h3>
                <ul style='color: #721c24; font-weight: bold;'>
                    <li>This is an AI-powered prediction system and should NOT be considered as a definitive medical diagnosis.</li>
                    <li>The predictions provided are based on machine learning models and may not be 100% accurate.</li>
                    <li>You MUST consult with a qualified healthcare professional for proper medical diagnosis and treatment.</li>
                    <li>By using this system, you acknowledge that the results are for preliminary screening purposes only.</li>
                    <li>The system providers cannot be held liable for any decisions made based on these predictions.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("Enter your name")
            
            # Add consent checkbox
            consent = st.checkbox("I understand and agree to the following terms:", value=False)
            st.markdown("""
                - This is a preliminary screening tool only
                - Results are predictions, not medical diagnoses
                - I will consult a healthcare professional for proper medical evaluation
                - I cannot hold the system providers liable for prediction outcomes
                - I am responsible for seeking appropriate medical care
            """)
            
            submit = st.form_submit_button("Start Session")
            
            if submit:
                if not consent:
                    st.error("⚠️ You must agree to the terms before proceeding.")
                elif not username:
                    st.error("Please enter your name.")
                else:
                    st.session_state.username = username
                    if username not in st.session_state.user_history:
                        st.session_state.user_history[username] = {
                            'predictions': 0,
                            'history': []
                        }
                    login()
                    st.rerun()
    
    else:
        # Main Application Interface
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f'<h1 class="main-title">AI-Powered Skin Cancer Detection System</h1>', unsafe_allow_html=True)
        with col2:
            st.markdown(f"Welcome, {st.session_state.username}!")
            if st.button("Logout"):
                logout()
                st.rerun()
        
        st.markdown(f"Total predictions made: {st.session_state.prediction_count}")
        
        uploaded_file = st.file_uploader("Choose a dermoscopic skin lesion image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
            st.session_state.last_uploaded_file = uploaded_file
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Validate the image first
            with st.spinner('Validating image...'):
                is_valid, validation_message = validate_dermoscopic_image(image)
                
                if not is_valid:
                    st.error(validation_message)
                    st.info("Please upload a valid dermoscopic skin lesion image. The image should be a close-up view of a skin lesion taken with a dermatoscope.")
                    return
            
            with st.spinner('Analyzing image with advanced AI...'):
                model = load_model()
                
                processed_img = process_image(image) 

                prediction = model.predict(processed_img)
                result_index = np.argmax(prediction)
                confidence = prediction[0][result_index] * 100
                
                result = {
                    'type': list(cancer_info.keys())[result_index],
                    'confidence': confidence,
                    'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.session_state.current_analysis = result
                st.session_state.prediction_count += 1
                st.session_state.user_history[st.session_state.username]['predictions'] = st.session_state.prediction_count
                st.session_state.user_history[st.session_state.username]['history'].append(result)
            
            with col2:
                detected_type = cancer_info[result['type']]
                
                st.markdown("### Detection Results")
                st.markdown("""
                    <div style='background-color: #f8d7da; border: 2px solid #dc3545; padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                        <p style='color: #721c24; font-weight: bold; margin: 0;'>
                            ⚠️ IMPORTANT: Please consult a healthcare professional for proper medical evaluation and diagnosis.
                            This is an AI prediction only.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div class='results-box'>
                        <h3 style="color: {'#ff4444' if detected_type['is_cancerous'] == 'Cancerous' else '#00C851'}">
                            Status: {detected_type['is_cancerous']}
                        </h3>
                        <h4>Type: {detected_type['name']}</h4>
                        <p>Confidence: {result['confidence']:.2f}%</p>
                        <p><i>{detected_type['confidence_explanation']}</i></p>
                        <p>Urgency: <span style="color: {urgency_colors[detected_type['urgency']]}">{detected_type['urgency']}</span></p>
                        <p><b>Recommended Action:</b> {detected_type['urgency_detail']}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # Now add the history section
                if st.session_state.user_history[st.session_state.username]['history']:
                    st.markdown("""
                        <div class="history-section">
                            <h2 class="history-title">Your Analysis History</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                history_df = pd.DataFrame(st.session_state.user_history[st.session_state.username]['history'])
                history_df['type'] = history_df['type'].apply(lambda x: cancer_info[x]['name'])
                st.dataframe(
                    history_df[['date', 'type', 'confidence']], 
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'date': 'Date & Time',
                        'type': 'Detected Type',
                        'confidence': st.column_config.NumberColumn(
                            'Confidence',
                            format="%.2f%%"
                        )
                    }
                )
                
                # Generate comprehensive PDF report
                comprehensive_report = generate_comprehensive_pdf_report(
                    st.session_state.username,
                    st.session_state.user_history[st.session_state.username]['history'],
                    st.session_state.prediction_count
                )
                
                st.download_button(
                    label="Download Comprehensive Report",
                    data=comprehensive_report,
                    file_name=f"skin_cancer_comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            
            # Resources section
            st.markdown("### Additional Resources")
            if st.session_state.current_analysis:
                detected_type = cancer_info[st.session_state.current_analysis['type']]
                st.markdown(f"""
                    #### Helpful Links for {detected_type['name']}
                """)
                for resource_name, resource_url in detected_type['resources']:
                    st.markdown(f"- [{resource_name}]({resource_url})")
            
            st.markdown("""
                #### General Skin Cancer Resources
                - [American Academy of Dermatology](https://www.aad.org/public/diseases/skin-cancer)
                - [World Health Organization - Skin Cancer](https://www.who.int/news-room/fact-sheets/detail/cancer)
                - [CDC - Skin Cancer](https://www.cdc.gov/cancer/skin/)
                - [Find a Dermatologist](https://www.aad.org/public/tools/find-a-derm)
            """)

if __name__ == "__main__":
    main()