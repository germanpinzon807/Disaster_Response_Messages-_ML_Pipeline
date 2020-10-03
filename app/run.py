import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Graph 1 data (given example):
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Graph 2 data: Distribution of Natural Disaster Messages
    natural_disaster_counts = pd.Series({'Cold': int(df.cold.sum()), 
                                      'Flood': int(df.floods.sum()),
                                      'Storm': int(df.storm.sum()),
                                      'Fire': int(df.fire.sum()),
                                      'Earthquake': int(df.earthquake.sum())})
    
    natural_disaster_names = list(natural_disaster_counts.index)
    
    # Graph 3 data: Distribution of Help Requests/Help Offerings
    request_offering_counts = pd.Series({'Requests': int(df.request.sum()), 
                                       'Offerings': int(df.offer.sum())})
    
    request_offering_names = list(request_offering_counts.index)
    
    # create visuals
    colors = ['lightslategray',] * 2
    colors[1] = 'crimson'
    graphs = [
        # First graph (given example):
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # Second graph: Distribution of Natural Disaster Messages
        {
            'data': [
                Pie(
                    labels=natural_disaster_names,
                    values=natural_disaster_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Natural Disaster Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Type of Disaster"
                }
            }
        },
        # Third graph: Distribution of Help Requests/Help Offerings
        {
            'data': [
                Bar(
                    x=request_offering_names,
                    y=request_offering_counts,
                    marker_color=colors
                )
            ],

            'layout': {
                'title': 'Distribution of Help Requests/Help Offerings ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Requests/Help"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()