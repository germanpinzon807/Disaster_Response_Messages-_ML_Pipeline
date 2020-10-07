# Disaster Response Pipeline Project

Due to the great amount of emergency messages that can pile up subsequent to a disaster situation from many sources: news agencies, first responders or the same people at risk, is necessary to organize this information and send it to the correct aid offices as soon as possible. This repo contains a disaster response messages classification tool that, besides displaying stats about the messages as a whole in different plots through an app dashboard, classifies emergency messages into 36 different categories, ranging from the type of aid needed (water, shelter) to the specifics of the risk situation (the presence of a child alone, the type of disaster, etc).

This tool was developed with machine learning algorithms and its available in a webpage, specifically trained with disaster related messages collected by Figure Eight (https://www.appen.com).

All the data files, data preprocessing, ML training, dashboard deploying and model results are contained in this repository.

Fundamentally, this project was developed in four stages:
- Preparation of the data and setup of a SQLite database where the web dashboard can extract info from.  
- A machine learning, multioutput text classification model building, trained with the data provided by the database.
- An evaluation of the model performance through different metrics (F1 score, Recall, Precision). 
- A Flask app deploying, displaying the dashboard and the message classification tool.

For any question, suggestion or discrepancy, feel free to write me at german.pinzon@davivienda.com


## Libraries needed:

You can run this project in a Python 3.7 environment with the following modules installed:

- pandas (1.0.1)
- numpy (1.18.1)
- re (2020.7.14)
- sys
- SQLAlchemy (1.3.13) 
- sklearn (0.19.1)
- nltk (3.4.5)
- pickle
- json
- plotly (2.0.15)

If you want to install these modules in an environment, you can run the installation of **requirements.txt** file in your environment.

## Repository Contents:

- app (folder):
    - templates (folder): HTML templates for the Flask web app.
        - go.html
        - master.html
    - run.py: Script that builds the web app, connects to the database and deploys the classification model. 
        
- data (folder):
    - DisasterResponse.db: Database with **'disaster_response_data'** SQL table used in the dashboard and the ML model training.
    - disaster_categories.csv: Disaster data provided by Figure Eight.
    - disaster_data.csv: Raw disaster messages data provided by Figure Eight.
    - process_data.py: Raw data preprocessing and database building script for later ML training.
    
- models (folder):
    - classifier.pkl: Serialized ML model developed and used in the web app. 
    - train_classifier.py: ML text classification building, training and evaluation script.

- requirements.txt: Set of Python modules needed to run this project.
- README.md


## User Instructions: 

If you just want to run the web app:

1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/


Although there is the files needed to run the web app in the repo already, if you want to run the entire pipeline instead, follow these steps:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Summary:

- In the data preparation stage, I found 41 duplicate rows that were dropped from the final table in the database. On the other hand, I think that the 'original' column in the preliminary message data table (the message in its original language) is not relevant for the developed classification tool (besides not all of the messages have their correspondent original language version, only 10184 of them), so I dropped this column from the final table too. 
- The final database table have 138 messages that have no classification. Thus, these messages were filtered for the train and test set of the ML text classification model. However, these messages were saved in another, alternative dataframe called 'df_test' within the ML training script, if it is required for model testing in later validations.
- The tokenization preprocessing step gets rid of english stopwords and applies lemmatization to every message string.
- Train/Test training sets rate was 80/20.
- The model was trained through a grid search technique, with 20 alternatives feeding 8 different parameters of the ML Pipeline, ranging from the n_grams and max permitted features in the Count Vectorizer preprocessing stage to the number of neighbors in the K-Neighbors Classifier used in the Scikit Learn's Multioutput Classifier. Of course, due to the fact this training took a lot of time, the final version of the Grid Search you will going to find in this repo is a simplified version with the best parameters found.
- I also explored a Random Forests Classifier with a custom made text lenght extractor sklearn estimator, with similar performance results compared to the shown model.
- The worst results in performance were in the 'related', 'aid_related', 'weather_related' and 'direct_report' categories. A summary of the model performance can be seen in the verbose of 'train_classifier.py' script run.
- Because the dataset is imbalanced, I recommend special attention to the F1-score metric, which balances how well the model classifies positive cases and the fraction of positive cases within a category.
- Most of the disaster messages come from news reports, followed by direct aid request messages and social media posts (as you can see in the web dashboard).
- The vast majority of messages are aid requests. Less than 3% are aid offerings (as you can see in the web dashboard).
- Earthquakes, storms and floods are the most frequent natural disasters identified in the dataset (as you can see in the web dashboard).


## Acknowlegdments:

I want to thank Figure Eight for collect and provide the data that made possible the development of this project.