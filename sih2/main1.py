#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("C:\\Users\\aksha\\OneDrive\\Desktop\\sih2\\trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Disease suggestions dictionary
disease_suggestions = {
    'Apple___Apple_scab': 'Use fungicides like Captan or Myclobutanil. Prune infected branches and avoid overhead watering.',
        'Apple___Black_rot': 'Remove infected fruit and branches. Use fungicide sprays in early spring.',
        'Apple___Cedar_apple_rust': 'Apply fungicides like Myclobutanil in early spring. Remove nearby junipers.',
        'Apple___healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases.',
        'Blueberry___healthy': 'Ensure proper irrigation and monitor humidity levels.',
        'Cherry_(including_sour)___Powdery_mildew': 'Use fungicides like Sulfur. Ensure proper air circulation around the plants.',
        'Cherry_(including_sour)___healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases.',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Apply fungicides and avoid overhead watering.',
        'Corn_(maize)___Common_rust_': 'Use resistant varieties and apply fungicides.',
        'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant varieties and apply fungicides.',
        'Corn_(maize)___healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases.',
        'Grape___Black_rot': 'Use fungicides and remove infected plant parts.',
        'Grape___Esca_(Black_Measles)': 'Remove and destroy infected vines and apply preventative fungicides.',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Use fungicides and practice good vineyard sanitation.',
        'Grape___healthy': 'Ensure proper irrigation and pest management.',
        'Orange___Haunglongbing_(Citrus_greening)': 'Remove and destroy infected trees. Use citrus greening resistant varieties.',
        'Peach___Bacterial_spot': 'Apply copper-based fungicides and avoid overhead watering.',
        'Peach___healthy': 'Maintain good orchard hygiene and manage irrigation.',
        'Pepper,_bell___Bacterial_spot': 'Use copper-based bactericides and avoid overhead watering.',
        'Pepper,_bell___healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases.',
        'Potato___Early_blight': 'Apply fungicides and rotate crops.',
        'Potato___Late_blight': 'Use resistant varieties and apply fungicides.',
        'Potato___healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases.',
        'Raspberry___healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases.',
        'Soybean___healthy': 'Monitor for pests and diseases, and manage irrigation effectively.',
        'Squash___Powdery_mildew': 'Use fungicides and ensure proper air circulation.',
        'Strawberry___Leaf_scorch': 'Apply fungicides and ensure proper irrigation.',
        'Strawberry___healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases and Maintain proper soil moisture and pest control.',
        'Tomato___Bacterial_spot': 'Use copper-based bactericides and avoid overhead watering.',
        'Tomato___Early_blight': 'Apply fungicides and avoid overhead watering.',
        'Tomato___Late_blight': 'Use resistant varieties and apply fungicides.',
        'Tomato___Leaf_Mold': 'Ensure good air circulation and use fungicides.',
        'Tomato___Septoria_leaf_spot': 'Apply fungicides and practice crop rotation.',
        'Tomato___Spider_mites Two-spotted_spider_mite': 'Use miticides and improve irrigation management.',
        'Tomato___Target_Spot': 'Apply fungicides and avoid overhead watering.',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Use resistant varieties and control vector insects.',
        'Tomato___Tomato_mosaic_virus': 'Use resistant varieties and manage plant debris.',
        'healthy': 'Your plant is healthy! Keep monitoring the environmental conditions to prevent diseases.'}

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if (app_mode == "Home"):
    st.header("AI-Driven Crop Disease Prediction and Management System")
    image_path = "C:\\Users\\aksha\\OneDrive\\Desktop\\sih2\\WhatsApp Image 2024-09-08 at 21.56.16_f06b8d7e.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif (app_mode == "About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)
                """)

# Prediction Page
elif (app_mode == "Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if (test_image is not None):
        # Show image
        st.image(test_image, width=300)

        # Predict button
        if (st.button("Predict")):
            st.snow()
            st.write("Our Prediction")
            result_index = model_prediction(test_image)

            # Reading Labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

            predicted_disease = class_name[result_index]
            st.success("Model is Predicting it's a {}".format(predicted_disease))

            # Display suggestions
            if predicted_disease in disease_suggestions:
                st.info("Suggestions: {}".format(disease_suggestions[predicted_disease]))
            else:
                st.warning("No suggestions available for this disease.")

