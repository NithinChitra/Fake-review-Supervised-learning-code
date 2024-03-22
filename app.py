from flask import Flask, render_template, request
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text)

    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text_input']
        
        # Preprocess the input text
        preprocessed_text = preprocess_text(text_input)

        # Transform the preprocessed text using the TF-IDF vectorizer
        text_vectorized = tfidf_vectorizer.transform([preprocessed_text])

        # Make a prediction using the pre-trained model
        prediction = model.predict(text_vectorized)

        result = "Fake" if prediction[0] == 1 else "Genuine"
        return render_template('index.html', result=result, text_input=text_input)

if __name__ == '__main__':
    app.run(debug=True)
