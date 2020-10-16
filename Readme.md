# Recommendation-System-based-on-Nonnegative-Matrix-Factorization

## Goal:
 To recommend the movies that user might want to watch.
### Recommender systems
- Recommender systems aim to predict users’ interests and recommend product items that quite likely are interesting for them. They are among the most powerful machine learning systems that online retailers implement in order to drive sales.
<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/netflix.PNG" alt="netflix" width="600" height="300" style="vertical-align:top; margin:10px">
 </p>
- Data required for recommender systems stems from explicit user ratings after watching a movie or listening to a song, from implicit search engine queries and purchase histories, or from other knowledge about the users/items themselves.
Examples of Recommendation systems are Netflix or YouTube that suggest playlists or make video recommendations 

## Types of recommender systems:
- Content-based systems, which use characteristic information.

- Collaborative filtering systems, which are based on user-item interactions.

- Hybrid systems, which combine both types of information with the aim of avoiding problems that are generated when working with just one kind.

## Dataset

link(https://grouplens.org/datasets/movielens/)
Movie lens Dataset consists of :b100000 ratings (1 lakh) , 600 users

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

Optimization techniques used in Gradient descent
 
- Regularized Gradient descent
 <p align="center">
 <img src="https://i.stack.imgur.com/8qNaM.png" alt="netflix" width="600" height="300" style="vertical-align:top; margin:10px">
 </p>
 
- Sliding Window Gradient descent
 <p align="center">
 <img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/sliding%20window.PNG" alt="netflix" width="600"  height="300" style="vertical-align:top; margin:10px">
 </p>
 
- Line Search Gradient descent
 <p align="center">
 <img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/line%20search.PNG" alt="netflix" width="600"  height="300" style="vertical-align:top; margin:10px">
 </p>
 
- Particle Swarm Optimization(PSO) Gradient descent
 <p align="center">
 <img src="https://static-01.hindawi.com/articles/ijap/volume-2013/649049/figures/649049.fig.003.jpg"  width="600" height="300" style="vertical-align:top; margin:10px">
 </p>
 
 ## Results :
 ### Root Mean Square Error (RMSE)
<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/results.PNG" alt="netflix" width="600" height="300" style="vertical-align:top; margin:10px">
</p>
 
### Movies Recommendations: 

<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/re1.PNG" width="600" height="300" style="vertical-align:top; margin:10px">
</p>
<p align="center">
<img src="https://github.com/Oprishri/Recommendation-system-based-on-Nonnegative-Matrix-Factorization/blob/master/images/re2.PNG" width="600" height="300" style="vertical-align:top; margin:10px">
</p>

## Conclusion

- In this project, we are trying to find the movies that we can recommend to user as per their interest. By using Matrix Factorization and some of the variate of Gradient Descent we able to do so.
- In our project Swarm optimization works so well that it minimises the error to 0.87.
