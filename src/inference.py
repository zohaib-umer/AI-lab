# =========================================================
# FINAL INFERENCE SYSTEM
# Intelligent Reading Comprehension System
# =========================================================

# =========================================================
# 1. IMPORT LIBRARIES 
# =========================================================

import os
import re
import string
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sparse

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import cosine_similarity

import joblib

warnings.filterwarnings("ignore")

# =========================================================
# 2. INFERENCE SYSTEM CLASS
# =========================================================

class RCInferenceSystem:

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self):

        print("Loading trained models...")

        try:
            # -------------------------------------------------
            # Robust Absolute Path Resolution
            # -------------------------------------------------
            # This ensures paths are correct regardless of where Streamlit is run from.
            base_dir = os.path.dirname(os.path.abspath(__file__))
            trad_dir = os.path.join(base_dir, '..', 'models', 'traditional')
            ens_dir = os.path.join(base_dir, '..', 'models', 'ensemble')

            # -------------------------------------------------
            # Traditional ML Models
            # -------------------------------------------------
            self.logreg = joblib.load(
                os.path.join(trad_dir, 'model_a_logistic_regression.pkl')
            )

            self.nb = joblib.load(
                os.path.join(trad_dir, 'model_a_naive_bayes.pkl')
            )

            # Check both possible locations for the ensemble model based on your folders
            ens_path_trad = os.path.join(trad_dir, 'model_a_ensemble.pkl')
            ens_path_soft = os.path.join(ens_dir, 'soft_voting_ensemble.pkl')

            if os.path.exists(ens_path_trad):
                self.ensemble = joblib.load(ens_path_trad)
            elif os.path.exists(ens_path_soft):
                self.ensemble = joblib.load(ens_path_soft)
            else:
                print("Warning: Ensemble model not found in expected directories.")

            # Note: Removed kmeans loading here since it isn't used in the inference functions below
            # and was causing a file not found error if named differently.

            # -------------------------------------------------
            # Vectorizers
            # -------------------------------------------------
            

            self.tfidf_vectorizer = joblib.load(
                os.path.join(trad_dir, 'tfidf_vectorizer.pkl')
            )

            print("All models loaded successfully!")

        except Exception as e:
            print(f"Error loading models: {e}")

    # =====================================================
    # TEXT CLEANING
    # =====================================================

    def clean_text(self, text):

        if not isinstance(text, str):
            return ""

        text = text.lower()

        text = text.translate(
            str.maketrans('', '', string.punctuation)
        )

        return text

    # =====================================================
    # MODEL A — ANSWER VERIFICATION
    # =====================================================

    def verify_answer(self, article, question, options):
        """
        Predicts the correct option label using the Binary Option-Level ensemble.
        It scores each option individually and picks the one with the highest probability.
        """

        clean_article = self.clean_text(article)
        clean_question = self.clean_text(question)

        best_label = 'A'
        highest_prob = -1.0
        confidence_scores = {}

        try:
            for label, opt_text in options.items():
                clean_opt = self.clean_text(opt_text)
                
                combined_text = clean_article + " " + clean_question + " " + clean_opt
                
                features = self.tfidf_vectorizer.transform([combined_text])
                
                prob_correct = self.ensemble.predict_proba(features)[0][1]
                
                confidence_scores[label] = round(float(prob_correct), 4)
                
                if prob_correct > highest_prob:
                    highest_prob = prob_correct
                    best_label = label

            result = {
                'Predicted_Answer': best_label,
                'Predicted_Text': options.get(best_label, "Unknown"),
                'Confidence_Scores': confidence_scores
            }

            return result

        except Exception as e:
            print(f"Prediction error: {e}")
            return {
                'Predicted_Answer': 'A',
                'Predicted_Text': options.get('A', 'Unknown'),
                'Confidence_Scores': None
            }
    # =====================================================
    # MODEL D — DISTRACTOR GENERATION
    # =====================================================

    def extract_candidate_phrases(self, article):

        phrases = re.split(r'[,.!?;:\n]', str(article))
        candidates = []

        for phrase in phrases:
            phrase = phrase.strip()
            words = phrase.split()

            if 3 <= len(words) <= 10:
                candidates.append(phrase)

        return list(set(candidates))

    def generate_distractors(self, article, correct_answer, top_n=3):

        """
        Advanced distractor generation
        using TF-IDF similarity.
        """

        candidates = self.extract_candidate_phrases(article)
        filtered_candidates = []

        for candidate in candidates:
            if (candidate.lower() != str(correct_answer).lower() and
                str(correct_answer).lower() not in candidate.lower()):
                filtered_candidates.append(candidate)

        if len(filtered_candidates) < top_n:
            return ["cannot be determined", "none of the above", "all of the above"]

        try:

            corpus = [str(correct_answer)] + filtered_candidates
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            similarities = cosine_similarity(
                tfidf_matrix[0:1], 
                tfidf_matrix[1:]
            ).flatten()

            ranked_indices = np.argsort(similarities)[::-1]
            distractors = []

            for idx in ranked_indices:
                candidate = filtered_candidates[idx]
                if candidate not in distractors:
                    distractors.append(candidate)
                if len(distractors) == top_n:
                    break

            while len(distractors) < top_n:
                distractors.append("unknown option")

            return distractors

        except Exception:
            return ["option x", "option y", "option z"]

    # =====================================================
    # MODEL E — HINT GENERATION
    # =====================================================

    def split_into_sentences(self, article):

        sentences = re.split(r'(?<=[.!?])\s+', str(article))
        sentences = [
            sentence.strip()
            for sentence in sentences
            if len(sentence.strip()) > 0
        ]

        return sentences

    def generate_hints(self, article, question, top_n=3):

        """
        Intelligent graduated hint generation.
        """

        sentences = self.split_into_sentences(article)

        if len(sentences) < 3:
            return [
                "Read the passage carefully.",
                "Look for important keywords.",
                "Focus on the main topic."
            ]

        try:

            corpus = [str(question)] + sentences
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)

            similarities = cosine_similarity(
                tfidf_matrix[0:1],
                tfidf_matrix[1:]
            ).flatten()

            ranked_indices = np.argsort(similarities)[::-1]
            top_indices = ranked_indices[:top_n]
            
            selected_sentences = [sentences[idx] for idx in top_indices]
            selected_sentences = selected_sentences[::-1]

            hints = []

            for i, sentence in enumerate(selected_sentences):

                # General Hint
                if i == 0:
                    hints.append(
                        "General Hint: " + " ".join(sentence.split()[:12])
                    )

                # Specific Hint
                elif i == 1:
                    hints.append(
                        "Specific Hint: " + " ".join(sentence.split()[:20])
                    )

                # Explicit Hint
                else:
                    hints.append("Explicit Hint: " + sentence)

            return hints

        except Exception:
            return [
                "Re-read the passage.",
                "Check important keywords.",
                "Focus on the conclusion."
            ]

# =========================================================
# 3. SAMPLE TESTING
# =========================================================

if __name__ == "__main__":

    print("\n================ TESTING INFERENCE SYSTEM ================\n")

    system = RCInferenceSystem()

    # -----------------------------------------------------
    # SAMPLE INPUT
    # -----------------------------------------------------

    article = """
    Artificial Intelligence is transforming modern education.
    Students can now use intelligent tutoring systems to
    improve learning outcomes. Machine learning algorithms
    help personalize educational content for each learner.
    """

    question = "What helps personalize educational content?"

    options = {
        'A': 'Traditional books',
        'B': 'Machine learning algorithms',
        'C': 'Classroom furniture',
        'D': 'Manual grading'
    }

    # -----------------------------------------------------
    # MODEL A TEST
    # -----------------------------------------------------

    print("\n1. Testing Answer Verification...\n")
    prediction = system.verify_answer(article, question, options)
    print(prediction)

    # -----------------------------------------------------
    # MODEL D TEST
    # -----------------------------------------------------

    print("\n2. Testing Distractor Generation...\n")
    distractors = system.generate_distractors(article, "Machine learning algorithms")
    print(distractors)

    # -----------------------------------------------------
    # MODEL E TEST
    # -----------------------------------------------------

    print("\n3. Testing Hint Generation...\n")
    hints = system.generate_hints(article, question)
    for hint in hints:
        print(hint)

    print("\nINFERENCE SYSTEM TEST COMPLETED!")