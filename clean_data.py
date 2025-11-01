import pandas as pd
import nltk

# --- CONFIGURATION ---
INPUT_FILE = 'medical_data.csv' 
OUTPUT_FILE = 'cleaned_medical_data.csv'

# --- NLTK FIX: Ensure stopwords are downloaded ---
# This is the standard, reliable way to download NLTK resources.
try:
    nltk.data.find('corpora/stopwords')
    print("NLTK stopwords found.")
except LookupError:
    print("NLTK stopwords not found. Downloading now...")
    nltk.download('stopwords')
    print("Download complete.")


print(f"Loading data from {INPUT_FILE}...")
try:
    # Load data (assuming your downloaded file has these columns)
    df = pd.read_csv(INPUT_FILE) 
except FileNotFoundError:
    print(f"ERROR: File '{INPUT_FILE}' not found. Did you rename your Kaggle file to medical_data.csv?")
    exit()

# Clean up and select relevant columns
# 'medical_specialty' is the Category and 'transcription' is the Text.
df = df[['medical_specialty', 'transcription',]].dropna()
df.columns = ['Category', 'Text']

# --- Filtering Logic: Crucial for Stable Training ---
# Filter out rare medical specialties (< 100 samples)
category_counts = df['Category'].value_counts()
min_samples = 100 
common_categories = category_counts[category_counts >= min_samples].index

df_clean = df[df['Category'].isin(common_categories)].reset_index(drop=True)

# --- FINAL STEP ---
# Save the prepared, cleaned dataset
df_clean.to_csv(OUTPUT_FILE, index=False)

print("\n--- DATA CLEANING COMPLETE ---")
print(f"Cleaned and filtered samples: {len(df_clean)}")
print(f"Total categories retained: {len(common_categories)}")
print(f"Top 5 Categories (Used for Training): \n{df_clean['Category'].value_counts().head(5)}")
print(f"Cleaned data saved to: {OUTPUT_FILE}")