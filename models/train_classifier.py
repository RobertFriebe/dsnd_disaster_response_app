import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import time
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def load_data(database_filepath):
    '''Loads the dataset from the database and outputs 
        response and predictor dataframes as well as category names list
    
    Args
        database filepath
    
    Returns
        - X: message text dataframe
        - Y: categories dataframe
        - category_names: list of 36 category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_responses', engine)
    X = df.message.values
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names



def tokenize(text):
    '''
    For every message make the following tokenizing steps
        - removes all special characters
        - creates tokens out of a string
        - removes the stopwords (english)
        - lemmatizes the tokes to its root form, 
            lowers all characters and 
            removes leading and training spaces
        - appends all clean tokens back to a list clean_tokens
    
    Args
        text message
    
    Returns
        clean_tokes: as a list
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)  
    tokens = [w for w in tokens if not w in stopwords.words("english")]   
    lemmatizer = WordNetLemmatizer()    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''Custom transformer that counts the number of characters in a message
    
    Args
        BaseEstimator, TransformerMixing
        
    Returns
        X_len: dataframe with the length of every message as integer
    '''
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        length_checker = np.vectorize(len)
        X_len = length_checker(X)
        return pd.DataFrame(X_len)
    
    


def build_model():
    '''Model Pipeline
    Takes as features the Tfidf-Matrix of the messages
     and the text length of every messages.
    It instanciates a Random Forest Classifier which 
     makes predictions for the 36 classes
     
    Args:
        None
    
    Returns
        model
    
    '''
    model = Pipeline([
    ('features', FeatureUnion([
        
        ('nlp_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize,
                                    max_df = 0.75,
                                    max_features = 5000,
                                    ngram_range = (1,2))),
            ('tfidf', TfidfTransformer())          
        ])),
        
        ('text_len', TextLengthExtractor())
    ])),
    
    ('clf_multi', MultiOutputClassifier(RandomForestClassifier(min_samples_leaf = 1,
                                                              min_samples_split = 6,
                                                              n_estimators = 100,
                                                              max_features = 'auto')))   
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates the model
    
    Args
        model: pipeline output
        X_test: Predictors (Tfidf messages matrix)
        Y_test: Response variables (36 classes)
        category_names: names of the 36 categories
        
    Returns
        Printed Classification Report for every class prediction
        
        https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
        
        Precision: is the ratio tp / (tp + fp) where tp is the 
            number of true positives and fp the number of false 
            positives. The precision is intuitively the ability 
            of the classifier not to label as positive a sample 
            that is negative.
            
        Recall: is the ratio tp / (tp + fn) where tp is the number 
            of true positives and fn the number of false negatives. 
            The recall is intuitively the ability of the classifier 
            to find all the positive samples.
            
        F1-Score: can be interpreted as a weighted harmonic mean of 
            the precision and recall, where an F-beta score reaches 
            its best value at 1 and worst score at 0.
        
        Support: is the number of occurrences of each class in y_true.
        
        Average Score/Total: function computes the average precision 
            (AP) from prediction scores. The value is between 0 and 1 
            and higher is better. 
    
    '''
    
    Y_pred = model.predict(X_test)
    Y_preds = pd.DataFrame(Y_pred, columns=category_names)

    for category in category_names:
        y_true = Y_test[category]
        y_pred = Y_preds[category]    
        print(category)
        print(classification_report(y_true, y_pred))


def save_model(model, model_filepath):
    '''Saves the model as a pickle file 
    
    Args
        model
        filepath for pickle file 
        
    Returns
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():   
    '''Loads the training and testing data, builds the model, fits the model and prints 
       a classification report, saves the model as a pickle file
       
       Args
        None
        
       Returns
        None
    '''
    
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()