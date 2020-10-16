# Recommendation-System-based-on-Nonnegative-Matrix-Factorization

## Goal:
 To recommend the movies that user might want to watch.
### Recommender systems
<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/netflix.PNG" alt="netflix" width="600" height="300" style="vertical-align:top; margin:10px">
 </p>
 
* Recommender systems aim to predict users’ interests and recommend product items that quite likely are interesting for them. They are among the most powerful machine learning systems that online retailers implement in order to drive sales.
* Data required for recommender systems stems from explicit user ratings after watching a movie or listening to a song, from implicit search engine queries and purchase histories, or from other knowledge about the users/items themselves.<br><br>
Examples of Recommendation systems are Netflix or YouTube that suggest playlists or make video recommendations 

## Types of recommender systems:
- Content-based systems, which use characteristic information.

- Collaborative filtering systems, which are based on user-item interactions.

- Hybrid systems, which combine both types of information with the aim of avoiding problems that are generated when working with just one kind.

## Dataset : 

Movie lens Dataset consists of ([link](https://grouplens.org/datasets/movielens/)):
* 100000 ratings (~ 1 lakh) 
* 600 users

# Some Analysis : 

### Information about the data : 

*********************************************************
Number of Users : 610 <br>
Number of movies : 9742

*********************************************************

### Movies with highest number of user ratings :

Age of Innocence, The (1993)<br>
I Love Trouble (1994)<br>
Virtuosity (1995)<br>
Cemetery Man (Dellamorte Dellamore) (1994)<br>
Teenage Mutant Ninja Turtles II: The Secret of the Ooze (1991)<br>

*********************************************************

### User who gave more ratings  :
  UserId   |   Movie ID   |
-----------|--------------|
  414      |   2698       | 
  599      |   2478       |
  474      |   2108       |
  448      |   1864       |
  274      |   1346       |

**********************************************************


## Matrix factorization
<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/factorization.PNG" alt="netflix" width="600" height="300" style="vertical-align:top; margin:10px">
</p>
 
Where,
-  R (users,movies) 
-  U (users,d)
-  VT (d,movies)
Here, d is the number of latent features.
## Cost Function
<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/cf1.PNG" alt="netflix" width="600" height="300" style="vertical-align:top; margin:10px">
</p>

<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/costfunc.PNG" alt="netflix" width="600" height="300" style="vertical-align:top; margin:10px">
</p>

## Prediction of rating of movies using gradient descent algorithm.

### Gradient descent
Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. To find a local minimum of a function using gradient descent, we take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.
<p align="center">
<img src="https://miro.medium.com/max/1000/0*ZppAJQdr9FnrnrGG.jpg" width="600" height="300" style="vertical-align:top; margin:10px">
</p>

### Optimization techniques used in Gradient descent
 
- <b>Regularized Gradient descent</b>
 <p align="center">
 <img src="https://i.stack.imgur.com/8qNaM.png" alt="netflix" width="600" height="250" style="vertical-align:top; margin:10px">
 </p>
 
- <b>Sliding Window Gradient descent</b>
 <p align="center">
 <img src="https://miro.medium.com/max/666/1*wytgFSuRwFp82yQ9kxD_-A.gif" alt="netflix" width="400"  height="300" style="vertical-align:top; margin:10px">
 </p>
 
- <b>Line Search Gradient descent</b>
 <p align="center">
 <img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/line%20search.PNG" alt="netflix" width="400"  height="300" style="vertical-align:top; margin:10px">
 </p>
 
- <b>Particle Swarm Optimization(PSO) Gradient descent</b>
 <p align="center">
 <img src="https://static-01.hindawi.com/articles/ijap/volume-2013/649049/figures/649049.fig.003.jpg"  width="400" height="300" style="vertical-align:top; margin:10px">
 </p>
 
## Results :

### Root Mean Square Error (RMSE)

  Optimization               |   Epochs        |    Train error   |    Test error    |
-----------------------------|-----------------|------------------|------------------|
  Gradient Descent (MMF)     |    10000        |     1.09         |     1.14         |
  Regularised (MMF)          |    1000         |     1.43         |     1.46         |


### It's time to recommend : 

<b>For User Id : 2</b>

  Movie ID     |         Movie                                                  |
---------------|----------------------------------------------------------------|
  26776        |   Porco Rosso (Crimson Pig) (Kurenai no buta) (1992)           | 
  104879       |   Prisoners (2013)                                             |
  5358         |   Mountains of the Moon (1990)                                 |
  175569       |   Wind River (2017)                                            |
  626          |   Thin Line Between Love and Hate, A (1996)                    |

