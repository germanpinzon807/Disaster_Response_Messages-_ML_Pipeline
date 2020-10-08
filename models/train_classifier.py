# Import libraries:
import sys
import pandas as pd
from sqlalchemy import create_engine

# For preprocessing text purposes:
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# For model pipeline construction and validation:
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Estimators:
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# For estimator building and model serialization:
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    """
    Loads data table from the database, drops unnecessary columns, gets the category labels 
    and splits the data in messages/target arrays.
        
    Params
    --------
        database_filepath (str): 
            database path, created in the process_data.py script.
                                            
    Returns
    --------
        X (Pandas Series): 
            Series with the messages translated to english.
            
        Y (Pandas DataFrame): 
            DataFrame with the sparsed categories classification target data.    

        category_names (list): 
            List with the 36 category labels the model will classify the messages into.  
    """      
    # load data from database
    database_filename_ext = 'sqlite:///' + database_filepath
    engine = create_engine(database_filename_ext)
    df = pd.read_sql_table('disaster_response_data', engine)

    # I am not going to use the original messages, so let´s drop the 'original' column:
    df.drop(['original'], axis=1, inplace = True)
    # df.isnull().sum()

    # There is 138 messages that have no classification. Let's hold these as a test set for later prediction (df_test), 
    # and split the messages that do have classification into features/target (validation) dataframes for model training                (df_train).
    df_train = df[~(df.isnull().any(axis=1))]
    df_test = df[df.isnull().any(axis=1)]
    
    # Messages and target dataframes
    X = df_train.iloc[:,1]
    Y = df_train.iloc[:, 3:]
    #X.head()
    #Y.head()
    
    # Category names:
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizer, punctuation/numbers filter and lemmatizer function for texts in English language.
        
    Params
    --------
        text (str): 
            String to be tokenized.
                                            
    Returns
    --------
        tokens (list): 
            List of the clean tokens found in text.  
    """    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())    
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens    


def build_model():
    """
    MultiOutput Text Classifier model building with pipeline technique and grid search over multiple parameters.
        
    Params
    --------
    
                                            
    Returns
    --------
        cv (GridSearchCV object): 
            Grid search pipeline MultiOutput Text Classifier model to be trained.  
    """
    # Pipeline:
    pipeline = Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=tokenize, max_df=0.5, max_features=10000)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(KNeighborsClassifier(), n_jobs = -1))
        ])

    # Parameters for grid searching (Reduced version for speed purposes, see README)
    parameters = parameters = {
            #'vectorizer__ngram_range': ((1, 1), (1, 2)),
            #'vectorizer__max_df': (0.5, 0.75, 1.0),
            #'vectorizer__max_features': (None, 5000, 10000),
            #'tfidf__use_idf': (True, False),
            'clf__estimator__leaf_size': [20, 30],
            #'clf__estimator__metric': ('minkowski', 'chebyshev'),
            'clf__estimator__n_neighbors': [4, 5],
            #'clf__estimator__weights': ('uniform', 'distance')
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the predictions made by the model builded in build_model() 
    function over the test set with three metrics per category: F1_score, 
    precision and recall and prints the results.
        
    Params
    --------
        model (sklearn model): 
            Model to be evaluated.
            
        X_test (array): 
            Array with the messages test set.            

        Y_test (Pandas DataFrame): 
            Array with the target test set.
            
        category_names (list): 
            Target labels provided by load_data() function.            
            
    Returns
    --------

    """ 
    # Predict over validation set:
    y_pred = model.predict(X_test)
    
    # Evaluate the performance: 
    y_test_arr = np.array(Y_test) # Turn y_test to a numpy array
    for i in range(y_test_arr.shape[1]):
        y_true = np.vstack(y_test_arr[:,i])
        y_predictions = np.vstack(y_pred[:,i]) 
        classification_report_str = classification_report(y_true, y_predictions)
        precision = classification_report_str[-35:-30]
        recall = classification_report_str[-25:-20]
        f1 = classification_report_str[-15:-10]
        print(category_names[i] + ':')
        print('F1_score: {}'.format(f1), 
              '   % Precision: {}'.format(precision),
              '   % Recall: {}'.format(recall)
             )
        print('\n')


def save_model(model, model_filepath):
    """
    Serializes and saves the model builded.
        
    Params
    --------
        model (sklearn model): 
            Model to be serialized and saved.
            
        model_filepath (str): 
            File path of the pickled model.            

    Returns
    --------

    """ 
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
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