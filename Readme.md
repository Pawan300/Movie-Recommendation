<img src="https://www.vshsolutions.com/wp-content/uploads/2020/02/recommender-system-for-movie-recommendation.jpg" widht=700 height=700>

# Dataset :
I am using Movielens Data set for recommending the movies to user.<br>
link(https://grouplens.org/datasets/movielens/)

# Algorithm Used:
  * Matrix Factorization with Gradient Descent 
  * Regularised Matrix Factorisation
  * Line Search method In Gradient Descent
  * Particle Swarm Optimization

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

### Errors (epochs 10000):

  * Training error :  1.0980925016874177
  * Testing error :  1.1524514692496157
  
### Training error :<br>
<img src="https://github.com/Pawan300/Movie-Recommendation/blob/master/Images/Error.png" width=400 height=400>

**********************************************************

### It's time to recommend : 

<b>For User Id : 2</b>

  Movie ID     |         Movie                                                  |
---------------|----------------------------------------------------------------|
  26776        |   Porco Rosso (Crimson Pig) (Kurenai no buta) (1992)           | 
  104879       |   Prisoners (2013)                                             |
  5358         |   Mountains of the Moon (1990)                                 |
  175569       |   Wind River (2017)                                            |
  626          |   Thin Line Between Love and Hate, A (1996)                    |

                    
