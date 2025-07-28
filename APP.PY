import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import requests

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained model
model = tf.keras.models.load_model('skin_model.h5')

# Ollama API configuration
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3"  # Using llama2 as it's smaller and more commonly available

# Define the disease classes (based on the notebook)
disease_classes = [
    'BA- cellulitis',
    'BA-impetigo', 
    'FU-athlete-foot',
    'FU-nail-fungus',
    'FU-ringworm',
    'PA-cutaneous-larva-migrans',
    'VI-chickenpox',
    'VI-shingles'
]

# Disease descriptions for better user understanding
disease_descriptions = {
    'BA- cellulitis': 'A bacterial skin infection that causes red, swollen, and painful skin. It can spread quickly and requires medical attention.',
    'BA-impetigo': 'A highly contagious bacterial skin infection that causes red sores and blisters, commonly affecting children.',
    'FU-athlete-foot': 'A fungal infection that affects the skin between the toes, causing itching, burning, and scaling.',
    'FU-nail-fungus': 'A fungal infection that affects the toenails or fingernails, causing discoloration, thickening, and crumbling.',
    'FU-ringworm': 'A fungal skin infection that causes a circular rash with clear skin in the middle, despite its name it is not caused by worms.',
    'PA-cutaneous-larva-migrans': 'A parasitic skin infection caused by hookworm larvae, creating winding, snake-like tracks under the skin.',
    'VI-chickenpox': 'A viral infection that causes an itchy rash with fluid-filled blisters, highly contagious and common in children.',
    'VI-shingles': 'A viral infection that causes a painful rash, typically appearing as a stripe of blisters on one side of the body.'
}

def get_detailed_explanation(disease_name):
    """Get detailed disease explanation and cures from Ollama API or fallback"""
    try:
        # Create a comprehensive prompt for the disease
        prompt = f"""Explain {disease_name} in detail and provide treatment options. Please structure your response with:

1. What is {disease_name}? (Brief explanation)
2. Symptoms and signs
3. Causes and risk factors
4. Treatment options and cures (4-5 points)
5. Prevention tips

Please keep the response concise but informative, focusing on practical treatment advice."""
        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["response"]
        else:
            print(f"Ollama API error: {response.status_code}, {response.text}")
            return get_fallback_explanation(disease_name)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return get_fallback_explanation(disease_name)
    except Exception as e:
        print(f"Error getting detailed explanation: {e}")
        return get_fallback_explanation(disease_name)

def get_fallback_explanation(disease_name):
    """Provide fallback detailed explanations when Ollama is not available"""
    fallback_explanations = {
        'BA- cellulitis': """1. What is BA-cellulitis?
Cellulitis is a bacterial skin infection that affects the deeper layers of skin and underlying tissue.

2. Symptoms and signs:
- Red, swollen, and painful skin
- Warmth and tenderness in affected area
- Fever and chills
- Rapid spreading of redness
- Blisters or skin dimpling

3. Causes and risk factors:
- Bacterial infection (usually Streptococcus or Staphylococcus)
- Skin injuries, cuts, or surgical wounds
- Poor circulation or diabetes
- Weakened immune system
- Obesity

4. Treatment options and cures:
- Oral antibiotics (penicillin, cephalexin, or clindamycin)
- Intravenous antibiotics for severe cases
- Pain relievers and anti-inflammatory medications
- Elevation of affected limb to reduce swelling
- Warm compresses to improve blood flow

5. Prevention tips:
- Keep skin clean and moisturized
- Treat cuts and wounds promptly
- Maintain good hygiene practices
- Control underlying conditions like diabetes
- Avoid scratching or picking at skin lesions""",

        'BA-impetigo': """1. What is BA-impetigo?
Impetigo is a highly contagious bacterial skin infection that causes red sores and blisters.

2. Symptoms and signs:
- Red sores that quickly rupture and ooze
- Honey-colored crusts
- Itching and mild pain
- Sores around nose, mouth, and hands
- Swollen lymph nodes

3. Causes and risk factors:
- Staphylococcus aureus or Streptococcus pyogenes bacteria
- Skin-to-skin contact with infected person
- Poor hygiene
- Warm, humid weather
- Cuts, scrapes, or insect bites

4. Treatment options and cures:
- Topical antibiotics (mupirocin, retapamulin)
- Oral antibiotics for widespread infection
- Gentle cleaning with soap and water
- Keeping affected areas covered
- Avoiding scratching to prevent spread

5. Prevention tips:
- Maintain good personal hygiene
- Wash hands frequently
- Keep cuts and scrapes clean
- Avoid sharing personal items
- Treat underlying skin conditions promptly""",

        'FU-athlete-foot': """1. What is FU-athlete-foot?
Athlete's foot is a fungal infection that affects the skin between the toes and soles of feet.

2. Symptoms and signs:
- Itching, burning, and stinging between toes
- Scaly, peeling, or cracked skin
- Redness and inflammation
- Blisters that may ooze
- Unpleasant foot odor

3. Causes and risk factors:
- Fungal infection (Trichophyton species)
- Walking barefoot in public areas
- Wearing tight, non-breathable shoes
- Sweaty feet and poor ventilation
- Weakened immune system

4. Treatment options and cures:
- Over-the-counter antifungal creams (clotrimazole, miconazole)
- Prescription antifungal medications
- Keeping feet clean and dry
- Wearing breathable cotton socks
- Using antifungal powders in shoes

5. Prevention tips:
- Keep feet clean and dry
- Wear breathable shoes and socks
- Use antifungal powder regularly
- Avoid walking barefoot in public areas
- Change socks frequently""",

        'FU-nail-fungus': """1. What is FU-nail-fungus?
Nail fungus is a fungal infection that affects toenails or fingernails, causing discoloration and thickening.

2. Symptoms and signs:
- Thickened, brittle nails
- Yellow, brown, or white discoloration
- Nail separation from nail bed
- Foul odor
- Pain and discomfort

3. Causes and risk factors:
- Fungal infection (dermatophytes, yeasts, molds)
- Walking barefoot in damp areas
- Wearing tight, non-breathable shoes
- Poor circulation or diabetes
- Weakened immune system

4. Treatment options and cures:
- Oral antifungal medications (terbinafine, itraconazole)
- Topical antifungal treatments
- Laser therapy for stubborn cases
- Surgical nail removal in severe cases
- Proper nail care and hygiene

5. Prevention tips:
- Keep feet clean and dry
- Wear breathable shoes and socks
- Avoid walking barefoot in public areas
- Trim nails regularly and properly
- Use antifungal sprays in shoes""",

        'FU-ringworm': """1. What is FU-ringworm?
Ringworm is a fungal skin infection that causes a circular rash with clear skin in the middle.

2. Symptoms and signs:
- Circular, red, scaly patches
- Clear center with red border
- Itching and burning sensation
- Hair loss in affected areas
- Blistering in severe cases

3. Causes and risk factors:
- Fungal infection (dermatophytes)
- Direct contact with infected person or animal
- Sharing personal items
- Warm, humid environments
- Weakened immune system

4. Treatment options and cures:
- Over-the-counter antifungal creams
- Prescription antifungal medications
- Oral antifungal drugs for severe cases
- Keeping affected areas clean and dry
- Avoiding scratching to prevent spread

5. Prevention tips:
- Maintain good personal hygiene
- Avoid sharing personal items
- Keep skin clean and dry
- Wear loose, breathable clothing
- Treat pets for fungal infections""",

        'PA-cutaneous-larva-migrans': """1. What is PA-cutaneous-larva-migrans?
Cutaneous larva migrans is a parasitic skin infection caused by hookworm larvae.

2. Symptoms and signs:
- Winding, snake-like tracks under skin
- Intense itching and burning
- Red, raised lines that move
- Blistering and secondary infection
- Pain and discomfort

3. Causes and risk factors:
- Hookworm larvae from animal feces
- Walking barefoot on contaminated soil
- Contact with infected sand or soil
- Travel to tropical regions
- Poor sanitation conditions

4. Treatment options and cures:
- Oral anti-parasitic medications (albendazole, ivermectin)
- Topical treatments for itching
- Antibiotics for secondary infections
- Cryotherapy for localized lesions
- Proper wound care and hygiene

5. Prevention tips:
- Wear shoes in potentially contaminated areas
- Avoid walking barefoot on beaches
- Practice good hygiene after outdoor activities
- Keep pets dewormed regularly
- Avoid contact with animal feces""",

        'VI-chickenpox': """1. What is VI-chickenpox?
Chickenpox is a viral infection that causes an itchy rash with fluid-filled blisters.

2. Symptoms and signs:
- Itchy, fluid-filled blisters
- Fever and fatigue
- Headache and loss of appetite
- Rash that spreads across body
- Scabbing of blisters

3. Causes and risk factors:
- Varicella-zoster virus infection
- Direct contact with infected person
- Airborne transmission
- Weakened immune system
- Lack of vaccination

4. Treatment options and cures:
- Rest and plenty of fluids
- Calamine lotion for itching
- Acetaminophen for fever
- Antiviral medications for severe cases
- Oatmeal baths for comfort

5. Prevention tips:
- Vaccination against chickenpox
- Avoid contact with infected individuals
- Maintain good hygiene practices
- Boost immune system with healthy diet
- Get adequate rest and sleep""",

        'VI-shingles': """1. What is VI-shingles?
Shingles is a viral infection that causes a painful rash, typically appearing as a stripe of blisters.

2. Symptoms and signs:
- Painful, burning rash
- Fluid-filled blisters in a stripe pattern
- Itching and tingling sensations
- Fever and fatigue
- Sensitivity to touch

3. Causes and risk factors:
- Reactivation of varicella-zoster virus
- Weakened immune system
- Stress and emotional trauma
- Aging (more common in older adults)
- Certain medications or illnesses

4. Treatment options and cures:
- Antiviral medications (acyclovir, valacyclovir)
- Pain relievers and anti-inflammatory drugs
- Topical treatments for rash
- Rest and stress management
- Cool compresses for comfort

5. Prevention tips:
- Vaccination against shingles
- Maintain a healthy immune system
- Manage stress levels
- Get adequate rest and nutrition
- Avoid contact with individuals who haven't had chickenpox"""
    }
    
    return fallback_explanations.get(disease_name, f"Detailed information about {disease_name} is not available. Please consult with a healthcare professional for proper diagnosis and treatment.")

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        # Read and resize image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Resize to 224x224 (model input size)
        img = cv2.resize(img, (128, 128))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_disease(image_path):
    """Predict skin disease from image"""
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return None, "Failed to process image"
        
        # Make prediction
        predictions = model.predict(processed_img)
        
        # Get the predicted class
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = disease_classes[predicted_class_idx]
        
        # Get confidence score
        confidence = float(predictions[0][predicted_class_idx])
        
        return predicted_class, confidence
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/project')
def project():
    return render_template('project.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/dataset/<path:filename>')
def dataset_image(filename):
    """Serve images from the dataset folder"""
    return send_from_directory('skin-disease-datasaet/images', filename)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file:
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join('temp', filename)
            
            # Create temp directory if it doesn't exist
            os.makedirs('temp', exist_ok=True)
            
            file.save(temp_path)
            
            # Predict the disease
            predicted_class, confidence = predict_disease(temp_path)
            
            # Clean up temporary file
            os.remove(temp_path)
            
            if predicted_class is None:
                return jsonify({'error': f'Prediction failed: {confidence}'})
            
            # Format the prediction result
            confidence_percentage = confidence * 100
            
            # Get disease description
            description = disease_descriptions.get(predicted_class, 'No description available.')
            
            # Get detailed explanation from Ollama API
            detailed_explanation = get_detailed_explanation(predicted_class)
            
            # Format the class name for display
            display_name = predicted_class.replace('-', ' ').replace('_', ' ').title()
            
            result = {
                'prediction': display_name,
                'confidence': f"{confidence_percentage:.1f}%",
                'description': description,
                'class_name': predicted_class,
                'detailed_explanation': detailed_explanation
            }
            
            return jsonify(result)
            
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'})

@app.route('/api/classes')
def get_classes():
    """API endpoint to get all disease classes"""
    classes_info = []
    for class_name in disease_classes:
        display_name = class_name.replace('-', ' ').replace('_', ' ').title()
        classes_info.append({
            'class_name': class_name,
            'display_name': display_name,
            'description': disease_descriptions.get(class_name, 'No description available.')
        })
    return jsonify(classes_info)

@app.route('/explain_disease')
def explain_disease():
    disease_name = request.args.get('name')
    if not disease_name:
        return jsonify({'error': 'No disease name provided.'}), 400
    explanation = get_detailed_explanation(disease_name)
    return jsonify({'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
