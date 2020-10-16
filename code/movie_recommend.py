import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import (
    analyis,
    indicator_matrix,
    loading_data,
    train_test_split,
    plot_error,
    show_rating
)

from movie_recommend_regularization import gradient_r, MMF_r


beta = 0.0001


def gradient(users, movies, result):

    users=users+2*0.000001*(result.dot(movies))
    movies=movies+2*0.000001*(result.T.dot(users))

    return (users, movies)


def MMF(user_rating, train, Itrain, k):
    
    users = np.random.rand(user_rating.shape[0], 5)
    movies = np.random.randint(4, size=(user_rating.shape[1], 5))

    error = []
    for i in range(k):
        result = np.multiply((train - users.dot(movies.T)), Itrain)
        users, movies = gradient(users, movies, result)
        error.append(np.sum(np.square(result)))

    return error, users, movies


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

def worker(user_rating, train, test, Itrain, Itest, movies_data, epochs):

    error, users, movies = MMF(user_rating, train, Itrain, epochs)

    ploting = plot_error(error)
    if ploting == False: 
        print("Something is wrong with errors.")
        print("Error you have : ", error)

    print("Training error : ", accuracy(users, movies, train, Itrain))
    print("Testing error : ", accuracy(users, movies, test, Itest))

    print("\n**********************************************************\n")
    print("It's time to recommend : \n")
    print("Enter User ID : ")

    user_id = 2

    movie_index = recommend(users, movies, user_id, user_rating)
        
    for i in movie_index:
        temp = movies_data.iloc[i]
        print("\n", temp["movieId"], "\t\t\t", temp["title"])


def main():

    rating_path = "dataset/ratings.csv"
    movie_path = "dataset/movies.csv"

    try:
        ratings, movies_data, status = loading_data(rating_path, movie_path)
        if status == False:
            return "Path doesn't exist"

        user_rating = ratings.pivot(index='userId', columns='movieId', values='rating')
        user_rating = user_rating.fillna(0)
        user_rating = user_rating.values

        train = np.zeros(user_rating.shape)
        test = np.zeros(user_rating.shape)

        show_rating(ratings)

        analyis(ratings, movies_data)
        train_test_split(user_rating, train, test)

        Itrain = indicator_matrix(train)
        Itest = indicator_matrix(test)
        
        print("#"*100)
        print("\n\nWithout Regularization : \n")
        worker(user_rating, train, test, Itrain, Itest, movies_data, 10000)
        print("#"*100)
        print("\n\nWith Regularization : \n")
        worker(user_rating, train, test, Itrain, Itest, movies_data, 1000)
        return "Successfully build"

    except Exception as e:
        print("Caught an Exception : ", e)
        print("Build Failed !!!!!!!!!!!!!!")

if __name__ == "__main__":
    print(main())