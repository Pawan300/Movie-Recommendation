import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from movie_recommend_regularization import MMF_r
from optimizer import optimizer_function
from utils import (
    accuracy,
    analyis,
    gradient,
    indicator_matrix,
    loading_data,
    plot_error,
    recommend,
    show_rating,
    train_test_split,
)

beta = 0.0001


def MMF(user_rating, train, Itrain, k):

    users = np.random.rand(user_rating.shape[0], 5)
    movies = np.random.randint(4, size=(user_rating.shape[1], 5))

    error = []
    for i in range(k):
        result = np.multiply((train - users.dot(movies.T)), Itrain)
        users, movies = gradient(users, movies, result, 0.000001)
        error.append(np.sum(np.square(result)))

    return error, users, movies


def worker(user_rating, train, test, Itrain, Itest, movies_data, epochs, status):

    if status == "GD":
        error, users, movies = MMF(user_rating, train, Itrain, epochs)
        ploting = plot_error(error, "Gradient_error.png")

    elif status == "R_GD":
        error, users, movies = MMF_r(user_rating, train, Itrain, beta, epochs)
        ploting = plot_error(error, "Randomize_gradient_error.png")

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


def argument_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--d",
        "--data_path",
        dest="data_path",
        help="path to data file",
        default="dataset/",
    )

    args = parser.parse_args()
    return args


def main():

    args = argument_parser()
    try:
        ratings, movies_data, status = loading_data(args.data_path)
        if status == False:
            return "Path doesn't exist"

        user_rating = ratings.pivot(index="userId", columns="movieId", values="rating")
        user_rating = user_rating.fillna(0)
        user_rating = user_rating.values

        train = np.zeros(user_rating.shape)
        test = np.zeros(user_rating.shape)

        show_rating(ratings)

        analyis(ratings, movies_data)
        train_test_split(user_rating, train, test)

        Itrain = indicator_matrix(train)
        Itest = indicator_matrix(test)

        # print("#" * 100)
        # print("\n\nNon Negative Matrix Factorization  : \n")
        # worker(user_rating, train, test, Itrain, Itest, movies_data, 10000, "GD")

        # print("#" * 100)
        # print("\n\nNon Negative Matrix Factorization With Regularization : \n")
        # worker(user_rating, train, test, Itrain, Itest, movies_data, 5000, "R_GD")
       
        print("#" * 100)
        print("\n\n!!!!!!!!!!!!! Different type of Optimizer !!!!!!!!!!!!")
        print("\n\nSliding Window protocol for optimizer : ")
        optimizer_function(user_rating, train, test, Itrain, Itest, movies_data)

        return "Successfully build"

    except Exception as e:
        print("Caught an Exception : ", e)
        print("Build Failed !!!!!!!!!!!!!!")


if __name__ == "__main__":
    print(main())
