import os
import pickle
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

from constant import API_key, mat_path, movies_data_path, data_path
from utils import loading_data, analysis, get_data, using_metadata_and_API, give_recomendations

def training(path):
  ratings, movies_data, links, metadata, status = loading_data(path)

  if "Overview" not in movies_data.columns: 
    movies_data["Overview"] = "Nan"
    movies_data = using_metadata_and_API(metadata, movies_data, links, API_key)
  else:
    movies_data["Overview"] = movies_data["Overview"] +" " + movies_data["genres"].apply(lambda x : " ".join(x.split("|")))
  tfv = TfidfVectorizer(min_df=3,  max_features=None,
              strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
              ngram_range=(1, 3),
              stop_words = 'english')

  tfv_matrix = tfv.fit_transform(movies_data['Overview'].values.astype('U'))
  sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
  with open(mat_path, "wb") as file:
    pickle.dump(sig, file)

  return movies_data, sig

if __name__=="__main__":
  if os.path.exists(mat_path):
    movies_data = pd.read_csv(movies_data_path)
    with open(mat_path, 'rb') as file:
      sig = pickle.load(file)
  else:
    movies_data, sig = training(path=data_path)
  print(movies_data.shape)
  print("*"*50)
  print("Recommenations : ")
  print(give_recomendations("The avengers", movies_data, sig))
  print("*"*50)
