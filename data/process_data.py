import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):    
    '''Loads the messages and categories datasets and merges them
    
    Args 
        messages filepath, csv file
        categories filepath, csv file
        
    Returns
        df: dataframe, merged dataset    
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)    
    df = messages.merge(categories, left_on="id", right_on="id")
    
    return df


def clean_data(df):
    '''Data cleaning step
        
        - splits the category column into separate category columns
          and stores them into a new dataframe
        - converts the the category values to just number 0 and 1
          (binary encoding: if value = 0, then 0; if value 1 or greater 1, then 1)
        - replaces the category column in df with the new category columns
        - removes duplicates
         
    Args
        takes the merged dataframe df
        
    Returns
        cleaned dataframe df   
    '''
   
    categories = df['categories'].str.split(pat=";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:len(x)-2]).tolist()
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = np.where(categories[column].astype(int) < 0.5, 0, 1)
   
    df.drop(columns=['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''Saves the cleaned dataset and stores it into an SQLite database
    
    Args
        df: cleaned dataset
        database_filename
        
    Returns
        None
    '''
    
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('disaster_responses', engine, index=False, if_exists='replace')


def main():
    '''
    - Loads the datasets
    - Cleans the data
    - Stores the cleaned dataset into an SQLite database
    
    Args
        None
    Returns
        print statements: Loading data, Cleaning data, Saving data
        if no file paths and database name are provided, it drop a message
    
    '''
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