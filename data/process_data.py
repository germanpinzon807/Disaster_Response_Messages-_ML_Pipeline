# Import modules:
import sys
import pandas as pd
from sqlalchemy import create_engine    


def load_data(messages_filepath, categories_filepath):
    """
    Loads datasets, merges them in a single dataframe and transforms 
    the categories data in a suited form for the ML model training.
        
    Params
    --------
        messages_filepath (str): 
            Messages original dataset path.

        categories_filepath (str):
            Categories original dataset path.
                                            
    Returns
    --------
        df (Pandas DataFrame): 
            Dataframe with the original messages and a sparse structure of 0's and 1's encoded classification per category.  
    """    
    # Step 1. Load datasets:
        # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
        # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Step 2. Merge datasets:
        # merge datasets
    df = messages.merge(categories, left_on='id', right_on='id')
    
    # Step 3. Split categories into separate category columns:
        # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(';',expand=True)
    
        # select the first row of the categories dataframe
    row = categories.iloc[0]
    row[32][:-2]

        # Extract from this row a list of new column names for categories.
    category_colnames =list(row.apply(lambda x: x[:-2]))
    
        # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Step 4. Convert category values to just numbers 0 or 1:
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
 
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')    
    #categories.info()
    
    # Step 5. Replace categories column in df with new category columns:
        # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    #df.head()
    
        # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
        
    return df

def clean_data(df):
    """
    Check and drop any duplicates that may be present in the dataframe provided by load_data() function.
        
    Params
    --------
        df (Pandas DataFrame): 
            Dataframe provided by load_data() function.
                                            
    Returns
    --------
        df (Pandas DataFrame): 
            Dataframe provided by load_data() function with no duplicates.  
    """    
    # Step 6. Remove duplicates
        # check number of duplicates
    #df.duplicated(keep='last').sum()
    
        # drop duplicates
    df.drop_duplicates(keep='last', inplace=True)
    
        # check number of duplicates after the drop
    #df.duplicated(keep='last').sum()
    
    return df   


def save_data(df, database_filename):
    """
    Database setup and table creation with the dataframe provided by clean_data() function.
        
    Params
    --------
        df (Pandas DataFrame): 
            Dataframe provided by clean_data() function.
            
        database_filename (str):
            Database file path to create.                              
    Returns
    --------
 
    """
    # Step 7. Database setup and table creation:
    database_filename_ext = 'sqlite:///' + database_filename
    engine = create_engine(database_filename_ext)
    df.to_sql('disaster_response_data' , engine, index=False) 


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