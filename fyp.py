import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import requests
import json

# Function to get nutrient information for a given food item from Nutritionix API
def get_nutrient_info(food_item):
    endpoint_url = 'https://trackapi.nutritionix.com/v2/natural/nutrients'
    query = {"query": food_item}
    app_id = '4d812ede'
    api_key = 'aec831def9a08c4f66a8d99f4597322b'
    headers = {'content-type': 'application/json',
               'x-app-id': app_id,
               'x-app-key': api_key}

    try:
        response = requests.post(endpoint_url, headers=headers, json=query)
        response.raise_for_status()
        data = json.loads(response.text)
        if "foods" in data:
            food_data = data['foods'][0]
            necessary_info = {
                "Food Name": food_data.get("food_name", "N/A"),
                "Serving Quantity": food_data.get("serving_qty", 0),
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
    except requests.exceptions.RequestException as e:
        return {'Error': f'Failed to fetch data: {str(e)}'}

# Function to load and preprocess the image
def preprocess_image(image):
    img = keras_image.load_img(image, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to classify the image and get nutrition information
def classify_and_get_nutrients(image_path, model):
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    class_labels = ['apple', 'banana', 'beef_rendang', 'burger', 'cucur_udang', 
                    'currypuff', 'fish_and_chips', 'fried_chicken', 'fried_noodles', 
                    'fried_rice', 'guava', 'kaya_toast', 'kuih_lapis', 'laksa', 
                    'milk', 'nasi_lemak', 'pisang_goreng', 'roti_canai', 
                    'teh_tarik', 'tomato']
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    nutrient_info = get_nutrient_info(predicted_class)
    return predicted_class, nutrient_info

# Streamlit app
def main():
    st.title('Malaysia Food Image Recognition')
    st.write('Upload a food image, the name of the food and nutrition information will be display.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        model = tf.keras.models.load_model('fyp2-18-adam-30-mobilenet.h5')  # Load your trained model
        predicted_class, nutrient_info = classify_and_get_nutrients(uploaded_file, model)

        st.write('Prediction:')
        formatted_class = predicted_class.replace('_', ' ')
        st.write(f'Predicted Class: {formatted_class}')
        st.write('Nutrition Facts:')
        if 'Error' in nutrient_info:
            st.error(nutrient_info['Error'])
        else:
            st.table(nutrient_info)  # Display nutrient info in table format

if __name__ == '__main__':
    main()
