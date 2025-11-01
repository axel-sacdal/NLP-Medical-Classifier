import streamlit as st
import joblib
import pandas as pd

# --- CONFIGURATION ---
MODEL_PATH = 'medical_classifier_pipeline.pkl'

# --- LOAD ASSETS (Uses caching for fast loading) ---
@st.cache_resource 
def load_model():
    """Loads the trained ML pipeline once when the app starts."""
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Did you run train_classifier.py?")
        return None

# Load the model and pipeline
classifier_pipeline = load_model()

# --- STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="Medical NLP Classifier", layout="wide")
st.title("⚕️ AI Medical Specialty Classifier")
st.markdown("##### Powered by Scikit-learn NLP Pipeline - Predicts medical domain from text.")
st.markdown("---")

if classifier_pipeline:

    # Text input box for the user to type/paste text
    user_input = st.text_area("1. Paste Medical Text or Symptoms Here:", 
                            height=200, 
                            placeholder="Example: 'A 65-year-old male with sharp, radiating chest pain, presenting with ST segment elevation on EKG, requiring immediate cardiac catheterization.'")

    if st.button("2. Classify Specialty", use_container_width=True):
        if user_input:

            # 1. Prediction (Using the saved pipeline)
            # predict_proba gives us the confidence score for all categories
            probabilities = classifier_pipeline.predict_proba([user_input])[0]

            # 2. Extract Top Results
            class_names = classifier_pipeline.classes_
            prob_df = pd.DataFrame({
                'Category': class_names, 
                'Probability': (probabilities * 100).round(2)
            }).sort_values(by='Probability', ascending=False)

            top_prediction = prob_df.iloc[0]['Category']
            top_confidence = prob_df.iloc[0]['Probability']

            st.success(f"### Predicted Specialty: {top_prediction}")
            st.subheader(f"Confidence: {top_confidence}%")

            # Display all probabilities in a nice table
            st.markdown("---")
            st.markdown("**Detailed Probability Breakdown**")
            st.dataframe(prob_df.head(5).set_index('Category'))