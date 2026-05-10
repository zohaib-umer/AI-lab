import pandas as pd
import string
import os
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sparse

def cleantext(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def main():
    print("1. Loading raw data...")
    try:
        traindf = pd.read_csv('data/raw/train.csv')
        testdf = pd.read_csv('data/raw/test.csv')
        
        columnstoclean = ['article', 'question', 'A', 'B', 'C', 'D']
        print("2. Cleaning text (lowercasing & removing punctuation)...")
        for col in columnstoclean:
            if col in traindf.columns:
                traindf[f'clean_{col}'] = traindf[col].apply(cleantext)
            if col in testdf.columns:
                testdf[f'clean_{col}'] = testdf[col].apply(cleantext)
                
        # Save custom train/test if you like, but we'll directly save clean_train.csv
        os.makedirs('data/processed', exist_ok=True)
        traindf.to_csv('data/processed/clean_train.csv', index=False)
        testdf.to_csv('data/processed/clean_test.csv', index=False)
        
        print("3. Building Vectorizer...")
        trained = CountVectorizer(binary=True, max_features=5000)
        traintext = traindf['clean_article'] + " " + traindf['clean_question']
        testtext = testdf['clean_article'] + " " + testdf['clean_question']
        
        print("4. Transforming text...")
        xtrain = trained.fit_transform(traintext)
        xtest = trained.transform(testtext)
        
        print("5. Saving feature matrices and models...")
        os.makedirs('models/traditional', exist_ok=True)
        joblib.dump(trained, 'models/traditional/ohe_vectorizer.pkl')
        sparse.save_npz('data/processed/X_train_ohe.npz', xtrain)
        sparse.save_npz('data/processed/X_test_ohe.npz', xtest)
        print("Preprocessing complete.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == '__main__':
    main()
