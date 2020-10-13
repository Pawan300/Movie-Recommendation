import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_error(error):

    if len(error)==0:
        return False

    plt.plot(list(range(1, len(error)+1)),error)
    plt.xlabel("Iterations")
    plt.ylabel("Error rates")
    plt.title("Error rates vs Iterations")
    plt.savefig("Error.png")
    return True

def indicator_matrix(data):

    I = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] != 0:
                I[i][j] = 1
    return I


def loading_data(rating_path, movie_path):

    if os.path.isfile(rating_path) and os.path.isfile(movie_path):
        ratings = pd.read_csv(rating_path)
        movies = pd.read_csv(movie_path)
        return ratings, movies, True
    else:
        return [], [], False


def show_rating(data):

    plt.hist(data["rating"], rwidth=0.5)
    plt.xlabel("Ratings")
    plt.ylabel("Counts")
    plt.title("Ratings counts")
    plt.savefig("Rating.png")


def analyis(ratings, movie_data):

    print("Information about the data : \n")
    print("*********************************************************")
    print("Number of Users :", len(np.unique(ratings["userId"])))
    print("Number of movies :", len(np.unique(movie_data["movieId"])))

    print("\n*********************************************************")
    print("\nMovies with highest number of user ratings :\n")
    for i in sorted(
        Counter(ratings["movieId"]).items(), key=lambda x: x[1], reverse=True
    )[:5]:
        print(movie_data["title"][i[0]])

    print("\n*********************************************************")
    print("\nUser who gave more ratings  :")
    print(
        ratings.groupby("userId")
        .count()
        .sort_values(["movieId"], ascending=False)["movieId"]
        .head(5)
    )


def train_test_split(user_rating, train, test):
    t = int(np.count_nonzero(user_rating) * 80 / 100)
    non_zero = []
    for i in range(user_rating.shape[0]):
        for j in range(user_rating.shape[1]):
            if user_rating[i, j] != 0:
                non_zero.append([i, j])
    k = 0
    while k <= t:
        i = random.randint(1, len(non_zero) - 1)
        train[non_zero[i][0], non_zero[i][1]] = user_rating[
            non_zero[i][0], non_zero[i][1]
        ]
        non_zero.remove(non_zero[i])
        k = k + 1

    for i in non_zero:
        test[i[0], i[1]] = user_rating[i[0], i[1]]


def rating_matrix(ratings, user_rating):
    for i in np.unique(ratings["userId"]):
        k = ratings[ratings["userId"] == i]
    for j in range(len(k)):
        if k["movieId"].iloc[j] - 1 <= 9742:
            user_rating[i - 1, k["movieId"].iloc[j] - 1] = k["rating"].iloc[j]
