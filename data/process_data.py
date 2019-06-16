# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' 
    Load data from given csv files

    messages_filepath: Filepath to the messages
    categories_filepath: Filepath to the categories
    return: Messages and categoires merged in a DataFrame
    ''' 

    # load messages dataset   
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    return messages.merge(categories, on='id')


def clean_data(df):
    '''
    Converts category values to just numbers 0 or 1
    Replaces categories column in df with new category columns
    Removes duplicates

    df: Dataframe to clean
    return: Cleaned DataFrame
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True) 

    # drop duplicates
    return df.drop_duplicates()


def save_data(df, database_filename):
    '''
    Saves the dataframe into an sqlite database

    df: Dataframe to be saved
    database_filename: The name of the database filename
    return: None
    '''

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('InsertTableName', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()