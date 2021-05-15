import os
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

from constant import API_key
from utils import loading_data, analysis, get_data, using_metadata_and_API, give_recomendations


def training(path):
  ratings, movies_data, links, metadata, status = loading_data(path)
  
  print("#"*100)
  print("Some analysis : ")
  analysis(ratings, movies_data)

  movies_data["title"] = list(movies_data["title"].apply(lambda x: " ".join(x.split("(")[:-1]).strip()))
  movies_data = movies_data.dropna()

  if "Overview" not in movies_data.columns: 
    movies_data["Overview"] = "Nan"
    movies_data = using_metadata_and_API(metadata, movies_data, links, API_key)
  else:
    movies_data = movies_data.drop("Unnamed: 0", axis=1)

  tfv = TfidfVectorizer(min_df=3,  max_features=None,
              strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
              ngram_range=(1, 3),
              stop_words = 'english')

  tfv_matrix = tfv.fit_transform(movies_data['Overview'])
  sig = sigmoid_kernel(tfv_matrix, tfv_matrix)

  return movies_data, sig

if __name__=="__main__":
  movies_data, sig = training(path=r"C:\Users\capricious\Desktop\Movie\Movie-Recommendation\dataset")
  print("*"*50)
  print("Recommenations : ")
  print(give_recomendations("I Love Trouble", movies_data, sig))
  print("*"*50)