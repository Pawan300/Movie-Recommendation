import math

import numpy as np
import pandas as pd


def gradient_r(users, movies, result, beta):

    users = users + 2 * 0.00000001 * (result.dot(movies) - 2 * beta * (np.sum(users)))
    movies = movies + 2 * 0.00000001 * (
        result.T.dot(users) - 2 * beta * (np.sum(movies))
    )

    return (users, movies)


def MMF_r(user_rating, train, Itrain, beta, k):
    users = np.random.rand(user_rating.shape[0], 5)
    movies = np.random.randint(4, size=(user_rating.shape[1], 5))

    error = []
    for i in range(k):
        cost = np.sum(
            np.square(np.multiply((train - users.dot(movies.T)), Itrain))
        ) + beta * ((np.sum(np.square(users)) + (np.sum(np.square(movies)))))
        result = np.multiply((train - users.dot(movies.T)), Itrain)
        users, movies = gradient_r(users, movies, result, beta)
        error.append(np.sum(np.square(cost)))

    return error, users, movies
