# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy import create_engine

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump


def load_data(database_filepath):
    '''
    Loads data from database
    Defines feature and target variables X and Y

    database_filepath: 
    return: Features, labels and label names    
    '''

    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('InsertTableName', engine)

    X = df.message.values
    Y = df.iloc[:,4:].copy()
    category_names = Y.columns.values.tolist()

    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizes the given text
    
    text: Text to be tokenize
    return: List of tokens
    '''
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Builds a classification model

    return: Classifier
    '''

    # build a machine learning pipeline
    pipeline = Pipeline([        
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])

    # use grid search to find better parameters
    parameters = {
        'vect__max_df': (0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 3],
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Prints the classification_report values for each category

    model: Model to test
    X_test: Feature values for testing
    Y_test: Label values for testing
    category_names: Label names
    return: None
    '''

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=category_names)
    for category in category_names:
        print(category, classification_report(Y_test[category], y_pred_df[category]))    


def save_model(model, model_filepath):
    ''' 
    Saves a model as a pickle file

    model: Model to be saved
    model_filepath: Filepath to the pickle file
    return: None
    '''

    dump(model, model_filepath) 


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