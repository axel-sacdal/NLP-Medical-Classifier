# ⚕️ AI Medical Specialty Classifier (NLP Application)
🚀 LIVE DEMO:
[Click Here to Test the AI Classifier Now!](YOUR_PUBLIC_STREAMLIT_URL_HERE)

### Project Overview
This project delivers a complete, end-to-end application that uses **Natural Language Processing (NLP)** and Machine Learning to automatically classify raw clinical text (like patient notes or symptom descriptions) into one of several predefined medical specialties (e.g., Cardiology, Orthopedics, Gastroenterology).

This application demonstrates proficiency across the **entire MLOps pipeline**—from data engineering to web deployment.

---

## 🚀 Key Deliverables & Features

* **Applied AI:** Built a high-accuracy, multi-class text classifier using Scikit-learn.
* **Production Pipeline (MLOps):** Implemented an `sklearn.pipeline` to ensure the vectorizer and model are permanently bundled and saved together (`.pkl` file) for seamless production use.
* **Interactive Deployment:** The model is hosted as a user-friendly web application using **Streamlit** for instant testing and presentation.
* **Core Task:** Predicts the medical specialty from unstructured text with a training accuracy of **~\[Your Model Training Accuracy]%** (e.g., 95.8%).

---

## 🛠️ Technology Stack

| Component | Technology | Role in Project |
| :--- | :--- | :--- |
| **Language** | Python (3.x) | Core programming and ML framework. |
| **ML/AI** | **Scikit-learn** | Used for **TF-IDF Vectorization** and **Logistic Regression** classifier. |
| **Deployment** | **Streamlit** | Built the fast, interactive web UI. |
| **Data Handling** | Pandas, `joblib` | Data cleaning and model serialization. |

---

## 🔬 Results & Validation (The Critical Thinking Section)

| Test Case | Predicted Output | Observation |
| :--- | :--- | :--- |
| **Cardiology Text** | [Your Model's Prediction for the Cardiology Text] | Model successfully identified highly specialized terms like EKG, angiography, or stent. |
| **"toothache"** | [Your Model's Prediction, e.g., **Surgery**] | The model identified the text as related to a procedure but was limited by the dataset's lack of a specific **'Dentistry'** category, demonstrating an understanding of **data bias/gaps**. |

**Conclusion:** The model achieves high accuracy on balanced, known categories but accurately exposes its limitations (low confidence) and defaults intelligently when faced with sparse domain data, showing robust real-world behavior.

---

## ⚙️ Setup and Installation

Follow these steps to run the project locally. **Note:** You must have Git and Python installed.

### 1. Clone the Repository & Activate Environment

```bash
git clone [https://github.com/axel-sacdal/NLP-Medical-Classifier](https://github.com/axel-sacdal/NLP-Medical-Classifier)
cd NLP-Medical-Classifier
python -m venv venv
# Activate the environment (e.g., in Windows PowerShell: .\venv\Scripts\Activate.ps1 or cmd /K venv\Scripts\activate.bat)

```

### 2. Install Dependencies
Install all required libraries from requirements.txt:
```bash
pip install -r requirements.txt
```

### 3. Data Acquisition & Training
Download the dataset and train the model:
```bash
# Download mtsamples.csv from Kaggle's Medical Transcriptions Dataset
# Rename it to medical_data.csv
# Move it into the root folder of this project
python train_classifier.py
```

### 4. Launch the Web Application

Run the Streamlit app locally:
```bash
streamlit run app.py
```
<img width="1905" height="945" alt="image" src="https://github.com/user-attachments/assets/df114483-8a9b-4c76-8286-c78ebf7a35c2" />


