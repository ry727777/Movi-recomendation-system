import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import ast

movies = pd.read_csv('tmdb_5000_movies.csv')
credit = pd.read_csv('tmdb_5000_credits.csv')

#merge these two data set
movies = movies.merge(credit,on='title')
# print(movies.info())

#on which we basis we are going to recommend
#genres id title overview cast crew keywords

# we are going to select all these column together
movies = movies[['genres','id','keywords','title','cast','crew','overview']]

#now using this dat set i am going to make a new data set
#having column id title tag->megrge (genres,keywords,title,cast,crew,overview)

movies.dropna(inplace=True)

#No column is duplicated
#print(movies.duplicated().sum())

# print(type(movies.iloc[0].genres))
# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
# #convert this into simple list having name only
# ['action','Adventure','Fantasy','scince fiction']

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert2(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            L.append(i['name'])
            count+=1
        else:
            break
    return L

def fetc_Dir(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

#Now change the column of movies['genres'] with list L using function convert
movies['genres'] = movies['genres'].apply(convert)
#Do the dame thing for rest of the column
movies['keywords'] = movies['keywords'].apply(convert)
#to cahge the the column of csat into list we top top 3 charachter
movies['cast'] = movies['cast'].apply(convert2)
#make crew table of single list containing director nameof that movie
movies['crew'] = movies['crew'].apply(fetc_Dir)

movies['overview'] = movies['overview'].apply(lambda x:x.split())

def remove_space(obj):
    l = []
    for i in obj:
        l.append(i.replace(" ",""))
    return l

movies['genres'] =  movies['genres'].apply(remove_space)
movies['cast'] =  movies['cast'].apply(remove_space)
movies['crew'] =  movies['crew'].apply(remove_space)
movies['keywords'] =  movies['keywords'].apply(remove_space)

#making new tags and add all column genres cast crew keywords into this
movies['tags'] = movies['genres']+movies['cast']+movies['crew']+movies['keywords']+movies['overview']

new_df = movies[['id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

def stem(text):
    ps = PorterStemmer()
    l = []
    for i in text.split():
        l.append(ps.stem(i))
    return " ".join(l)

new_df['tags'] = new_df['tags'].apply(stem)

'''Now we are ready to with correct data on which we are going to recommend movies
Now here we are going to implement some algorithm to relate tags of one movies to another moviw tags'''

# Now we are goimg to make vector of movie using scikit learn countvectorization
vc =  CountVectorizer( max_features=5000,stop_words='english')
x = vc.fit_transform(new_df['tags'])
#convert into numpy array
vectors = x.toarray()

#calculate cosine distance from one movie to another movies using sklearn
similarity = cosine_similarity(vectors)
# print(similarity)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse = True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

movie_name = input("Enter movie name:-")
print("Similar Movies")
recommend(movie_name)