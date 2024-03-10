import os
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from uses import label_to_uses
import warnings
import pandas as pd


# Ignore all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)


# Load the Model
model = tf.keras.models.load_model('leaf-xception.keras')

# Label Names
labels = ['Aloevera', 'Amla', 'Amruthaballi', 'Arali', 'Astma_weed', 'Badipala', 'Balloon_Vine', 'Bamboo', 'Beans', 'Betel', 'Bhrami', 'Bringaraja', 'Caricature', 'Castor', 'Catharanthus', 'Chakte', 'Chilly', 'Citron lime (herelikai)', 'Coffee', 'Common rue(naagdalli)', 'Coriender', 'Curry', 'Doddpathre', 'Drumstick', 'Ekka', 'Eucalyptus', 'Ganigale', 'Ganike', 'Gasagase', 'Ginger', 'Globe Amarnath', 'Guava', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jackfruit', 'Jasmine',
          'Kambajala', 'Kasambruga', 'Kohlrabi', 'Lantana', 'Lemon', 'Lemongrass', 'Malabar_Nut', 'Malabar_Spinach', 'Mango', 'Marigold', 'Mint', 'Neem', 'Nelavembu', 'Nerale', 'Nooni', 'Onion', 'Padri', 'Palak(Spinach)', 'Papaya', 'Parijatha', 'Pea', 'Pepper', 'Pomoegranate', 'Pumpkin', 'Raddish', 'Rose', 'Sampige', 'Sapota', 'Seethaashoka', 'Seethapala', 'Spinach1', 'Tamarind', 'Taro', 'Tecoma', 'Thumbe', 'Tomato', 'Tulsi', 'Turmeric', 'ashoka', 'camphor', 'kamakasturi', 'kepala']


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Prediction function
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(299, 299))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.sigmoid(predictions[0])

    predicted_class = np.argmax(score)
    confidence = 100 * np.max(score)
    predicted_label = labels[predicted_class]

    uses = label_to_uses.get(predicted_label, 'Unknown uses')

    return predicted_label, confidence, uses


@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/feedback')
# def feedback():
#     return render_template('feedback.html')


@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        feedback_text = request.form.get('feedback_text')

        # Check if the Excel file already exists
        excel_file_path = 'static/feedback_data.xlsx'
        if os.path.exists(excel_file_path):
            # If the file exists, read it into a DataFrame
            feedback_data = pd.read_excel(excel_file_path)
        else:
            # If the file doesn't exist, create an empty DataFrame
            feedback_data = pd.DataFrame(
                columns=['Full Name', 'Feedback Text'])

        # Append new feedback data to the DataFrame
        new_feedback = pd.DataFrame({
            'Full Name': [full_name],
            'Feedback Text': [feedback_text]
        })

        feedback_data = pd.concat(
            [feedback_data, new_feedback], ignore_index=True)

        try:
            # Save the updated feedback data to the Excel file
            feedback_data.to_excel(
                excel_file_path, index=False, sheet_name='Feedback')
            success = True
        except Exception as e:
            print(f"Error: {e}")
            success = False

        # Redirect to feedback page or another page
        return render_template('feedback.html', success=success)

    return render_template('feedback.html')


@app.route('/identify', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):

            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            prediction, confidence, uses = predict_image(file_path)

            return render_template('index.html', prediction=prediction, confidence=confidence, uses=uses, file_path=file.filename)
        else:
            return render_template('index.html', error='Unsupported file extension. Please upload a file with one of the following extensions: png, jpg, jpeg')
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    uses = None
    plant_name = None

    if request.method == 'POST':
        plant_name = request.form.get('plantName', '').strip()
        if plant_name in label_to_uses:
            uses = label_to_uses[plant_name]

    return render_template('search.html', uses=uses, plant_name=plant_name)


if __name__ == '__main__':
    app.run(debug=True, port=9000)
