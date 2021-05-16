import os, json
import requests
import pandas as pd
import numpy as np
from collections import Counter

import locale # to format currency as USD
locale.setlocale( locale.LC_ALL, '' )

def loading_data(data_path):
    
    if os.path.isfile(os.path.join(data_path, "ratings.csv")):
        ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
        movies = pd.read_csv(os.path.join(data_path, "final_dataset.csv"))
        links = pd.read_csv(os.path.join(data_path, "links.csv"))
        metadata = pd.read_csv(os.path.join(data_path, "movies_metadata.csv"))
        return ratings, movies, links, metadata, True
    else:
        return [], [], [], [], False


def analysis(ratings, movie_data):

    print("Information about the data : \n")
    print("*"*50)
    print("Number of Users :", len(np.unique(ratings["userId"])))
    print("Number of movies :", len(np.unique(movie_data["movieId"])))

    print("*"*50)
    print("\nMovies with highest number of user ratings :\n")
    
    for i in sorted(
        Counter(ratings["movieId"]).items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(movie_data["title"][i[0]])

    print("*"*50)
    print("\nUser who gave more ratings  :")
    print(
        ratings.groupby("userId")
        .count()
        .sort_values(["movieId"], ascending=False)["movieId"]
        .head(5)
    )

def get_data(API_key, Movie_ID):
    query = 'https://api.themoviedb.org/3/movie/'+str(Movie_ID)+'?api_key='+API_key+'&language=en-US'
    response =  requests.get(query)
    if response.status_code==200: 
        array = response.json()
        text = json.dumps(array)
        return (text)
    else:
        return ("error")


def write_file(data, API_key):
    
    for i in range(data.shape[0]):
        result = get_data(API_key, data["tmdbId"].iloc[i])
        
        if result == "error":
            overview = "None"
        else:
            dataset = json.loads(result)
            try:
                overview = dataset['overview']
            except:
                overview = "None"
        data["Overview"].iloc[i] = overview
    return data


def using_metadata_and_API(metadata, movies_data, links, API_key):
  
  movies_data = pd.merge(links, movies_data, on="movieId")
  movies_data = movies_data.dropna()

  for i in range(movies_data.shape[0]):
    
    try:
      temp = metadata[metadata["id"] == movies_data["movieId"].iloc[i]]
      if len(temp) > 0:
           overview = temp["Overview"].iloc[0]
           print("Overview : ", overview)
      else:
          result = get_data(API_key, movies_data["tmdbId"].iloc[i])
          
          if result == "error":
              overview = "None"
          else:
              dataset = json.loads(result)
              try:
                  overview = dataset['overview']
              except:
                  overview = "None"
          print("Overview using API : ", overview)
      movies_data["Overview"].iloc[i] = overview
    except Exception as w:
      print("Error : ",w)
  movies_data.to_csv(r"/content/drive/MyDrive/Colab Notebooks/Movie/Movie-Recommendation/dataset/preprocessed.csv")
  return movies_data

def give_recomendations(title, movies_data, sig):
    title = title.lower()
    idx = movies_data.index[movies_data["title"] == title].tolist()
    if len(idx) == 0:
      return("Movie is not registered with us!!!!!!!!!!")
    else:
      idx = idx[0]
      sig_scores = list(enumerate(sig[idx]))
      sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
      sig_scores = sig_scores[1:11]
      movie_indices = [i[0] for i in sig_scores]
      return movies_data['title'].iloc[movie_indices]
