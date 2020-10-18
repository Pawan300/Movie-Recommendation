import numpy as np
import pandas as pd
from utils import accuracy, gradient, plot_error, recommend


def MMF_line_search(user_rating, train, Itrain, k, threshold=0.001):

    users = np.random.rand(user_rating.shape[0], 5)
    movies = np.random.randint(4, size=(user_rating.shape[1], 5))

    error = []
    alpha = 1 / 9000
    cost_old = 0
    cost_new = 0

    for i in range(k):
        result = np.multiply((train - users.dot(movies.T)), Itrain)
        users, movies = gradient(users, movies, result, alpha)
        cost_new = np.sum(np.square(result))
        error.append(cost_new)

        if abs(cost_old - cost_new) > threshold:
            alpha = alpha / 10
            users, movies = gradient(users, movies, result, alpha)
        else:
            break
        cost_old = cost_new

    return error, users, movies


def MMF_sliding_window(user_rating, train, Itrain, k, win_size=5):

    users = np.random.rand(user_rating.shape[0], 5)
    movies = np.random.randint(4, size=(user_rating.shape[1], 5))

    errors = []
    window_error = []

    mean1 = 0
    mean2 = 0
    for i in range(k):
        result = np.multiply((train - users.dot(movies.T)), Itrain)
        users, movies = gradient(users, movies, result, 0.000001)
        error = np.sum(np.square(result))
        errors.append(error)

        if i != 0 and i % win_size == 0:

            if i == win_size:
                mean1 = np.mean(window_error)
            else:
                mean2 = np.mean(window_error)
                if mean2 < mean1:
                    break
                mean1 = mean2
            window_error = []
        else:
            window_error.append(error)

    return errors, users, movies


def optimizer_function(user_rating, train, test, Itrain, Itest, movies_data):

    print("\nInitiating sliding window optimizer : \n")
    errors, users, movies = MMF_sliding_window(user_rating, train, Itrain, 10000, 5)

    ploting = plot_error(errors, "Sliding_window_error.png")

    if ploting == False:
        print("Something is wrong with errors.")
        print("Error you have : ", errors)

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

    print("\nInitiating line search optimizer : \n")
    errors, users, movies = MMF_line_search(user_rating, train, Itrain, 10000)

    ploting = plot_error(errors, "Line_search_error.png")

    if ploting == False:
        print("Something is wrong with errors.")
        print("Error you have : ", errors)

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
