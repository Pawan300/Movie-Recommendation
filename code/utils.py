import math
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def gradient(users, movies, result, alpha):

    users = users + 2 * alpha * (result.dot(movies))
    movies = movies + 2 * alpha * (result.T.dot(users))

    return (users, movies)


def accuracy(users, movies, data, I):

    data = np.multiply(data, I)  # indicator matrix problem
    result = users.dot(movies.T)
    e = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j] != 0:
                e += (data[i, j] - result[i, j]) ** 2
    return math.sqrt(e / np.count_nonzero(data))


def recommend(users, movies, user_id, user_rating):

    result = users.dot(movies.T)
    result = result[user_id]
    user_rating = user_rating[user_id]

    temp = []
    for i, j in enumerate(result):
        temp.append((i, j))

    result = sorted(temp, key=lambda x: x[1], reverse=True)
    count = 5  # only top 5 recommendation
    movies_index = []

    for i in result:
        if count == 0:
            break
        if user_rating[i[0]] == 0:
            movies_index.append(i[0])
            count = count - 1
    return movies_index


def plot_error(error, name):

    if len(error) == 0:
        return False

    plt.plot(list(range(1, len(error) + 1)), error)
    plt.xlabel("Iterations")
    plt.ylabel("Error rates")
    plt.title("Error rates vs Iterations")
    plt.savefig(os.path.join("Images/", name))
    return True


def indicator_matrix(data):

    I = np.zeros(data.shape)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i][j] != 0:
                I[i][j] = 1
    return I


def loading_data(data_path):

    if os.path.isfile(os.path.join(data_path, "ratings.csv")):
        ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
        movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
        return ratings, movies, True
    else:
        return [], [], False


def show_rating(data):

    plt.hist(data["rating"], rwidth=0.5)
    plt.xlabel("Ratings")
    plt.ylabel("Counts")
    plt.title("Ratings counts")
    plt.savefig("Images/Rating.png")


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
            if user_rating[i][j] != 0:
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
