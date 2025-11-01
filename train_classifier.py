import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline # Key tool for professionalism
import joblib

# --- CONFIGURATION ---
DATA_FILE = 'cleaned_medical_data.csv'
MODEL_OUTPUT = 'medical_classifier_pipeline.pkl'

print(f"Loading cleaned data from {DATA_FILE}...")
# The file cleaned_medical_data.csv was created by your previous script!
df = pd.read_csv(DATA_FILE)

# 1. Define X (Features: Text) and Y (Target: Category)
X = df['Text'].astype(str) # The medical transcription text
Y = df['Category']        # The medical specialty

# 2. Define the Pipeline: Vectorizer + Classifier
# This is professional practice: it chains the text-to-number step with the AI step.
model_pipeline = Pipeline([
    # Step 1: Text Vectorization (Converts text to numbers based on word importance)
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    
    # Step 2: Training the Classification Model (The actual AI)
    ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
])

# 3. Train the Model (This might take a minute or two)
print("\n--- TRAINING THE NLP CLASSIFIER (This may take 30-60 seconds) ---")
model_pipeline.fit(X, Y)
print("Training complete.")

# 4. Evaluate the Model
score = model_pipeline.score(X, Y)
print(f"Model Training Accuracy: {round(score * 100, 2)}%")

# 5. Save the ENTIRE Pipeline
joblib.dump(model_pipeline, MODEL_OUTPUT)
print(f"Full NLP pipeline successfully saved as '{MODEL_OUTPUT}'.")