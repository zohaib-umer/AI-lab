import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

class ModelEvaluator:
    """
    Evaluator class to compute performance metrics for 
    Model A (Q&A Verifier) and Model B (Distractor/Hint Generator).
    """

    def __init__(self):
        pass

    # =====================================================
    # 1. CLASSIFICATION METRICS (MODEL A)
    # =====================================================

    def evaluate_classification(self, y_true, y_pred, average='binary'):
        """
        Computes standard classification metrics.
        Useful for Model A binary classification (Correct vs Incorrect option).
        """
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'Recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'F1_Score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        return metrics

    def get_classification_report(self, y_true, y_pred, target_names=None):
        """
        Returns a detailed classification report as a string.
        """
        return classification_report(y_true, y_pred, target_names=target_names, zero_division=0)

    def get_confusion_matrix(self, y_true, y_pred):
        """
        Returns the confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)

    # =====================================================
    # 2. GENERATION/RANKING METRICS (MODEL B)
    # =====================================================

    def evaluate_overlap(self, generated_texts, reference_texts):
        """
        Computes simple word-level overlap/Jaccard similarity.
        Useful for Distractor/Hint baseline evaluation if exact references exist.
        """
        def jaccard_similarity(str1, str2):
            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            if not set1 or not set2:
                return 0.0
            intersection = set1.intersection(set2)
            union = set1.union(set2)
            return len(intersection) / len(union)

        scores = [
            jaccard_similarity(gen, ref) 
            for gen, ref in zip(generated_texts, reference_texts)
        ]
        
        return {
            'Mean_Jaccard_Similarity': np.mean(scores),
            'Max_Jaccard_Similarity': np.max(scores),
            'Min_Jaccard_Similarity': np.min(scores)
        }

# =========================================================
# SAMPLE TESTING
# =========================================================

if __name__ == "__main__":
    print("\n================ TESTING EVALUATOR ================\n")

    evaluator = ModelEvaluator()

    # Dummy Data for Model A
    y_true_dummy = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred_dummy = [1, 0, 1, 0, 0, 1, 1, 0]

    print("1. Classification Metrics (Model A Mock Data):")
    clf_metrics = evaluator.evaluate_classification(y_true_dummy, y_pred_dummy)
    for k, v in clf_metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\n2. Classification Report:")
    print(evaluator.get_classification_report(y_true_dummy, y_pred_dummy, target_names=['Incorrect', 'Correct']))

    # Dummy Data for Model B
    gen_hints = ["Look for important keywords.", "Focus on the main topic."]
    ref_hints = ["Look for keywords in the text.", "Focus on the main idea."]
    
    print("\n3. Generation Overlap Metrics (Model B Mock Data):")
    gen_metrics = evaluator.evaluate_overlap(gen_hints, ref_hints)
    for k, v in gen_metrics.items():
        print(f"   {k}: {v:.4f}")

    print("\nEVALUATOR TEST COMPLETED!")
