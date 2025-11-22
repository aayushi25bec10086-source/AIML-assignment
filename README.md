# AIML-assignment
the idea pitched about scam detection and priority
#module 1 data processing
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

def clean_text(text):
   
  text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    return text

def fit_vectorizer(data_series):
    
  cleaned_data = data_series.apply(clean_text)
    X = vectorizer.fit_transform(cleaned_data)
    # Save the fitted vectorizer for later use in prediction
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    return X

def transform_data(data_series):
  
  
  global vectorizer
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("Error: Vectorizer not found. Run model_trainer.py first.")
        return None

  cleaned_data = data_series.apply(clean_text)
    return vectorizer.transform(cleaned_data)


    #PART 2 MODEL TRAINING
   import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from src.preprocessor import fit_vectorizer, transform_data

# Configuration settings
MODEL_PATH = 'models/spam_classifier.pkl'
DATA_PATH = 'data/corpus.csv'
TEST_SIZE = 0.2

def train_model():
    print("--- Starting Model Training ---")

    
  try:
        # Assuming the CSV has 'text' and 'label' columns
  data = pd.read_csv(DATA_PATH)
      data = data.dropna(subset=['text', 'label'])
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}. Check your setup.")
        return

  X = data['text']
    y = data['label']

    
  X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"Data split: Train={len(X_train)}, Test={len(X_test)}")

    
  print("Fitting TF-IDF Vectorizer...")
    X_train_vec = fit_vectorizer(X_train)
    X_test_vec = transform_data(X_test)
    print("Vectorization complete.")

    
  print("Training Naive Bayes Classifier...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    print("Model training complete.")

    
  y_pred = model.predict(X_test_vec)
    
  print("\n--- Model Evaluation ---")
    print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    
    
  print("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
    print(report)

    
  joblib.dump(model, MODEL_PATH)
    print(f"\nModel successfully saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_model()


    #MODULE 3 REAL TIME API
  from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.preprocessor import transform_data

app = Flask(__name__)


try:
    MODEL = joblib.load('models/spam_classifier.pkl')
except FileNotFoundError:
    print("ERROR: Model not found. Run model_trainer.py first!")
    MODEL = None


CATEGORIES = {0: "General Ham", 1: "Spam", 2: "High Priority"}

def get_prediction(message):
    
  if MODEL is None:
        return {"error": "Model not loaded."}, 500

    
  message_df = pd.Series([message])
    message_vec = transform_data(message_df)
    
    
  prediction_raw = MODEL.predict(message_vec)[0]
    prediction_proba = MODEL.predict_proba(message_vec)[0]
    confidence = float(prediction_proba[prediction_raw])
    
    
  final_label = prediction_raw
    if prediction_raw == 0:  # If it's Ham
        priority_keywords = ['urgent', 'meeting', 'important']
        if any(kw in message.lower() for kw in priority_keywords):
            final_label = 2  # Assign High Priority

  return {
        "classification": CATEGORIES.get(final_label, "Unknown"),
        "confidence_score": round(confidence, 4)
    }

@app.route('/predict', methods=['POST'])
def predict():
    
  if not request.json or 'message' not in request.json:
        return jsonify({"error": "Invalid input. 'message' field is required."}), 400
    
   message = request.json['message']
    result = get_prediction(message)
    
  if "error" in result:
        return jsonify(result), 500

    
  return jsonify(result)

if __name__ == '__main__':
    # Run the API on the default port
    app.run(debug=True)
