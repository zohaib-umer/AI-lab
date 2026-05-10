# EduAI Comprehension: Intelligent Reading Comprehension and Quiz Generation System

EduAI Comprehension is an AI-powered Reading Comprehension and Quiz Generation System built using the RACE (ReAding Comprehension from Examinations) dataset. It utilizes classical machine learning techniques to automatically generate comprehension questions, predict correct answers, extract distractor options, evaluate user responses, and provide graduated hints.
## github link
https://github.com/zohaib-umer/AI-lab-project.git

## 🎯 Project Overview

This project integrates two distinct AI pipelines, exposed through a modern, interactive Streamlit frontend:

1. **Model A — Question & Answer Generator / Verifier:** 
   Utilizes Traditional ML models (Logistic Regression, Naive Bayes, Ensemble Voting) to verify the correct answer given a passage, question, and candidate options. Includes TF-IDF based feature engineering.
2. **Model B — Distractor & Hint Generator:**
   A specialized pipeline that produces plausible but incorrect answer choices (distractors) and extracts graduated hints from the passage to guide users toward the correct answer without revealing it explicitly.

## 📂 Project Structure

```text
race_rc_project/
├── data/
│   ├── raw/                  # Original RACE CSV files (train.csv, dev.csv, test.csv)
│   └── processed/            # Cleaned and feature-engineered datasets
├── models/
│   ├── traditional/          # Pickled scikit-learn models (Model A components)
│   └── ensemble/             # Pickled Voting Classifiers and meta-models
├── src/
│   ├── inference.py          # Unified inference API for Model A and B operations
├── ui/
│   └── app.py                # Streamlit entry point (Modern Premium UI)
├── notebooks/
│   └── EDA.ipynb             # Exploratory Data Analysis & initial data prep
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🚀 Features

- **Interactive Quiz Interface:** Input a custom article or load a random RACE dataset sample, generate a question, and attempt to answer it.
- **Real-Time Verification (Model A):** The system evaluates your selected option against the passage using an ensemble of classical ML models and provides an explanation.
- **Distractor Generation (Model B):** Dynamically extracts and ranks plausible distractor phrases based on TF-IDF semantic similarity to challenge the user.
- **Graduated Hint Engine:** Can't figure out the answer? Use the Hint Engine to extract up to 3 graduated sentence-level clues from the article based on relevance scoring.
- **Internal Analytics Dashboard:** View model confidence, macro metrics, inference latency, and feature engineering details (e.g., Soft-Voting Ensemble, Max Features = 5000).

## 🛠️ Setup & Installation

**1. Clone the repository**
```bash
git clone <repository-url>
cd AI
```

**2. Create a virtual environment (Recommended)**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Ensure Dataset & Models are Present**
- Place the original RACE datasets (`train.csv`, `dev.csv`, `test.csv`) into the `data/raw/` directory.
- Ensure that trained models (e.g., `model_a_rf.pkl`, `tfidf_vectorizer.pkl`) are located in the appropriate `models/` directory.

## 🏃 Running the Application

To launch the Streamlit frontend, simply run:

```bash
streamlit run ui/app.py
```

The application will start a local server and can be accessed in your browser at `http://localhost:8501`.

## 🧠 Under the Hood (Model Architecture)

- **Vectorization:** TF-IDF (`TfidfVectorizer`)
- **Answer Verification (Model A):** Traditional classification models including Support Vector Machines (SVM), Logistic Regression, and Multinomial Naive Bayes. An ensemble method is used to maximize Macro F1 Score.
- **Hint Extraction (Model B):** Sentences are tokenized and scored based on their TF-IDF cosine similarity to the given question, allowing extractive hints sorted by descending relevance.
- **Distractor Generation:** Relies on TF-IDF word retrieval and cosine similarity ranking to find words/phrases that are contextually valid but factually incorrect.

## 📜 Acknowledgements
Based on the RACE Dataset introduced by Lai et al. (2017): *RACE: Large-scale ReAding Comprehension Dataset From Examinations*.
