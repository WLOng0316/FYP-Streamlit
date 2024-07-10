import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import requests
import json
from ultralytics import YOLO

# Define target classes for counting
target_classes = ['apple', 'beef_rendang', 'kuih_lapis', 'milk', 'nasi_lemak']

# Function to get nutrient information for a given food item from Nutritionix API
def get_nutrient_info(food_item):
    endpoint_url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    query = {"query": food_item}
    app_id = '4d812ede'
    api_key = 'aec831def9a08c4f66a8d99f4597322b'
    headers = {
        'content-type': 'application/json',
        'x-app-id': app_id,
        'x-app-key': api_key
    }
    response = requests.post(endpoint_url, headers=headers, json=query)
    data = json.loads(response.text)
    if "foods" in data:
        food_data = data['foods'][0]
        necessary_info = {
            "Food Name": food_data.get("food_name", "N/A"),
            "Serving Quantity": food_data.get("serving_qty", 0),
            "Serving Unit": food_data.get("serving_unit", "N/A"),
            "Serving Unit": food_data.get("serving_unit", "N/A"),
            "Serving Weight (grams)": food_data.get("serving_weight_grams", 0),
            "Calories": food_data.get("nf_calories", 0),
            "Total Fat (g)": food_data.get("nf_total_fat", 0),
            "Saturated Fat (g)": food_data.get("nf_saturated_fat", 0),
            "Cholesterol (mg)": food_data.get("nf_cholesterol", 0),
            "Sodium (mg)": food_data.get("nf_sodium", 0),
            "Carbohydrates (g)": food_data.get("nf_total_carbohydrate", 0),
            "Dietary Fiber (g)": food_data.get("nf_dietary_fiber", 0),
            "Sugars (g)": food_data.get("nf_sugars", 0),
            "Protein (g)": food_data.get("nf_protein", 0),
            "Potassium (mg)": food_data.get("nf_potassium", 0),
            "Phosphorus (mg)": food_data.get("nf_p", 0)
        }
        return necessary_info
    else:
        return {'Error': 'No food data found'}

# Function to load and preprocess the image from a PIL Image
def preprocess_image(pil_image):
    img = pil_image.resize((224, 224))  # Resize the image to the target size
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the input
    return img_array

# Function to classify the image and get nutrients
def classify_and_count(pil_image, model, yolo_model):
    img_array = preprocess_image(pil_image)
    predictions = model.predict(img_array)
    class_labels = ['apple', 'banana', 'beef_rendang', 'burger', 'cucur_udang', 
                    'currypuff', 'fish_and_chips', 'fried_chicken', 'fried_noodles', 
                    'fried_rice', 'guava', 'kaya_toast', 'kuih_lapis', 'laksa', 
                    'milk', 'nasi_lemak', 'pisang_goreng', 'roti_canai', 
                    'teh_tarik', 'tomato']
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence_score = np.max(predictions)

    if predicted_class in target_classes:
        # Convert PIL Image to format expected by YOLO model
        img_cv = np.array(pil_image.convert('RGB'))
        # Perform detection with YOLO
        results = yolo_model(img_cv)[0]
        count = sum(1 for result in results.boxes.data.tolist() if results.names[int(result[5])].upper() == predicted_class.upper())
        food_item = f"{count} {predicted_class}" if count > 0 else predicted_class
    else:
        food_item = predicted_class

    nutrient_info = get_nutrient_info(food_item)
    return predicted_class, confidence_score, nutrient_info, count if predicted_class in target_classes else None

# Load YOLO model
yolo_model_path = 'best.pt'  # Update this path to where your YOLO weights are stored
yolo_model = YOLO(yolo_model_path)

# Load classification model
model_path = 'fyp2-18-adam-30-mobilenet.h5'
model = tf.keras.models.load_model(model_path)

# Streamlit app
def main():
    st.title('Malaysia Food Image Recognition and Nutritional Facts')
    st.write('Upload a food image, and get the name of the food along with nutritional information. If the food is one of the specified classes, it will also count how many are present in the image.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Process image
        predicted_class, confidence_score, nutrient_info, count = classify_and_count(image, model, yolo_model)

        # Show results
        st.write('Prediction:')
        formatted_class = predicted_class.replace('_', ' ')
        st.write(f'Predicted Class: {formatted_class} (Confidence: {confidence_score:.2f})')
        if count is not None:
            st.write(f'Count: {count}')
        st.write('Nutritional Information:')
        if 'Error' in nutrient_info:
            st.error(nutrient_info['Error'])
        else:
            st.table(nutrient_info)

if __name__ == '__main__':
    main()
