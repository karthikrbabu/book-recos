from flask import Flask, request

# from __future__ import print_function
# import sys

#Library Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import string
import warnings
import json

#Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Tags Clusters
from sklearn.cluster import KMeans

#Stats
from scipy import stats

#Tuircreate
import turicreate as tc

app = Flask(__name__)

#Main Books Data
books = pd.read_csv("ultimate_books.csv")

#Color Books Data
bk_plus_color = pd.read_csv("books_colors.csv")

#Train and test data
train_user_data = pd.read_csv("user_train_data.csv")
test_user_data = pd.read_csv("user_test_data.csv")

#Turicerate model
item_similarity_model = tc.load_model('item_similarity_cluster.model')


################# AUTHOR STUFF ################
tf_author = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tf_matrix_authors = tf_author.fit_transform(books['authors'])
cosine_sim_authors = cosine_similarity(tf_matrix_authors, tf_matrix_authors)

# All the books 
titles = books['normalized']
indices = pd.Series(books.index, index=books['normalized'])

def similarity_by_authors(book, num_recs=10):
    try:
        index = indices[book]
        scores = list(enumerate(cosine_sim_authors[index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if num_recs > len(scores):
            scores = scores[1:len(scores)]  
        else:
            scores = scores[1:num_recs]
        return scores
    except:
        print("We don't have that book!")



################# COLOR STUFF ################

color_titles = bk_plus_color['normalized']

#NEEd to get normalized titles for them to match!
color_indices = pd.Series(bk_plus_color.index, index=bk_plus_color['normalized'])

# Distance Matrix
cosine_sim_colors = cosine_similarity(
    bk_plus_color[['pct_blue', 'pct_light', 'pct_green', 'pct_yellow', 'pct_red', 'pct_magenta', 'pct_cyan', 'pct_dark']],
    bk_plus_color[['pct_blue', 'pct_light', 'pct_green', 'pct_yellow', 'pct_red', 'pct_magenta', 'pct_cyan', 'pct_dark']])

def similarity_by_colors(book, num_recs=10):
    try:           
        index = color_indices[book]     
        scores = list(enumerate(cosine_sim_colors[index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if num_recs > len(scores):
            scores = scores[1:len(scores)]  
        else:
            scores = scores[1:num_recs+1]            
            # scores = scores[1:num_recs]
        # book_indices = [book[0] for book in scores]
        # return bk_plus_color.iloc[book_indices]

        return scores
    except:
        print("We don't have that book!")


################ OVERALL GENRE BASED ################
tf_overall = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tf_matrix_overall = tf_overall.fit_transform(books['tag_name'])
cosine_sim_overall = cosine_similarity(tf_matrix_overall, tf_matrix_overall)


# Function that get book recommendations based on the cosine similarity score of books tags
def overall_recommendations(book, num_recs = 10):
    try:
        index = indices[book]
        scores = list(enumerate(cosine_sim_overall[index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        if num_recs > len(scores):
            scores = scores[1:len(scores)]  
        else:
            scores = scores[1:num_recs]                        
        # book_indices = [book[0] for book in scores]
        # return books.iloc[book_indices]
        return scores
    except:
        print("Book doesn't exist")



################ BOOK PARSING ################
def process_recs(recs, scores):
    result = []

    if recs:
        for ind,row in enumerate(recs):
            result.append(make_book_object(row, scores[ind]))
    return result


def make_book_object(row, score):

    rounded_score = str(np.round(score, 4))

    if row is not None:
        book_link = "https://www.goodreads.com/book/show/"+str(row["goodreads_book_id"])
        authors = []
        for author in row["authors"].split(","):
            authors.append({ "text": author, "action": '' })

        if row["book_id"] is None:
            image_url = 'https://c.saavncdn.com/873/6-Weeks-Wiggle--English-2018-20181112191820-150x150.jpg'
        else:
            image_url = row["image_url"]
            
        title = row["title"]
        
        return { "title": { "text": title, "action": book_link },
        "author": authors, 
        "book_id": str(row["book_id"]), 
        "cos_score": rounded_score,
        "type": "book",
        "image": [image_url]
            }
    else:
        return { "title": { "text": '', "action": '' },
        "author": [], 
        "book_id": -99, 
        "type": "book",
        "image": ['']
            }        



@app.route("/author_recos", methods=["GET"])
def get_author_recos(): 

    #Pass to run in global context to pull in pre filled in data
    # and function calls
    book = request.args.get('book')
    num_recs = int(request.args.get('num_recs'))

    if book is None:
        return { 'author_recos': "ERROR" }
    else:
        books_and_scores = similarity_by_authors(book, num_recs)
        book_indices = [book[0] for book in books_and_scores]
        scores = [book[1] for book in books_and_scores]
        
        book_table = books.iloc[book_indices]
        recs_list = book_table.to_dict(orient='records')
        author_recos = process_recs(recs_list, scores)
        return { 'author_recos': author_recos }


@app.route("/genre_recos", methods=["GET"])
def get_genre_recos(): 

    #Pass to run in global context to pull in pre filled in data
    # and function calls
    book = request.args.get('book')
    num_recs = int(request.args.get('num_recs'))

    if book is None:
        return { 'genre_recos': "ERROR" }
    else:

        books_and_scores = overall_recommendations(book, num_recs)
        book_indices = [book[0] for book in books_and_scores]
        scores = [book[1] for book in books_and_scores]
        
        book_table = books.iloc[book_indices]
        recs_list = book_table.to_dict(orient='records')
        genre_recos = process_recs(recs_list, scores)
        return { 'genre_recos': genre_recos }




@app.route("/color_recos", methods=["GET"])
def get_color_recos(): 

    #Pass to run in global context to pull in pre filled in data
    # and function calls
    book = request.args.get('book')
    num_recs = int(request.args.get('num_recs'))

    if book is None:
        return { 'color_recos': "ERROR" }
    else:

        books_and_scores = similarity_by_colors(book, num_recs)
        book_indices = [book[0] for book in books_and_scores]
        scores = [book[1] for book in books_and_scores]
        
        book_table = bk_plus_color.iloc[book_indices]
        recs_list = book_table.to_dict(orient='records')
        
        color_recos = process_recs(recs_list, scores)      
        return { 'color_recos': color_recos }
   

@app.route("/get_book", methods=["GET"])
def get_book(): 

    #Pass to run in global context to pull in pre filled in data
    # and function calls
    book = request.args.get('book')

    if book is None:
        print("BOOK NONE")
        return { 'color_recos': "ERROR" }
    else:
        index = indices[book]
        book_row = books.iloc[index]
        book_data = make_book_object(book_row, 1.0)
        return {'book' : book_data }


# Tuircreate stuff
# list of test users
@app.route('/test_users', methods=["GET"])
def get_test_users():
    return {'test_users': test_user_data['user_id'].unique().tolist()}

# list of books already read by user
# cos_score is the rating the user gave
@app.route('/training_books')
def get_training_books():
    user_id = request.args.get('user', default=7416, type=int)
    user_books_ids = train_user_data[train_user_data['user_id']==user_id]['book_id'].to_list()
    user_books = books[books['book_id'].isin(user_books_ids)]
    user_books_list = user_books.to_dict(orient='records')
    user_books_recos = process_recs(user_books_list, train_user_data[train_user_data['user_id']==user_id]['rating'].to_list() )
    return {'user_books': user_books_recos}

# list of correct picks for user (held back data)
# cos_score is the rating the user gave
@app.route('/testing_books')
def get_testing_books():
    user_id = request.args.get('user', default=7416, type=int)
    user_books_ids = test_user_data[test_user_data['user_id']==user_id]['book_id'].to_list()
    user_books = books[books['book_id'].isin(user_books_ids)]
    user_books_list = user_books.to_dict(orient='records')
    user_books_recos = process_recs(user_books_list, test_user_data[test_user_data['user_id']==user_id]['rating'].to_list() )
    return {'user_books': user_books_recos}

# recommend a list of 10 books to a user using collaborative filtering
# the cluster # column was used as side data
# cos_score is the confidence score
# the cos_score is positive or negative
# to indicate whether the pick matches the test data
# (the raw score is always positive)
@app.route('/similar_books')
def recommend_similar_books():
    user_id = request.args.get('user', default=7416, type=int)
    recommended_book_ids = item_similarity_model.recommend([user_id])
    recommended_books = books[books['book_id'].isin(recommended_book_ids['book_id'])]
    recommended_books_list = recommended_books.to_dict(orient='records')

    book_is_recommended = recommended_book_ids['book_id'].is_in(test_user_data[test_user_data['user_id']==user_id]['book_id'])
    scores = [score[0] if score[1] == 1 else score[0]*-1  for score in zip(recommended_book_ids['score'], book_is_recommended)]

    recommended_books_recos = process_recs(recommended_books_list, scores)
    return {'recommended_books': recommended_books_recos}


if __name__ == "__main__":
    app.run(debug=True, port=5000)


