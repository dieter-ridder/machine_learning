# Machine Learning - Udacity NanoDegree certification

This repository collects my submissions during Udacity NanoDegree certification.

## Overview

- **titanic survival exploration** <br>
List of Titanic Passengers, containing several features, and as label the information if they survived or not.Objective was to find a set of rules predicting if a passenger survives.<br>


- **boston housing** <br> 
Data source was from [UCI machine learning repositary](https://archive.ics.uci.edu/ml/datasets/Housing). It contains a list with prices of houses as labels as well as some features(number of rooms, as well as 2 features indicating the social class of the neighbor hood and the quality of schools nearby) on which the price may depend on. Data preparation was already done by Udacity team. Objective was to predict house prices in the Boston area.


- **finding donors** <br>
Objective was to accurately model individuals' income using data collected from the 1994 U.S. Census. 

## Runtime
For each project the Jupyter notebook is stored.

## Details

###Titanic Survival Exploration
_used data_: <br>
List of Titanic Passengers, containing several features, and as label the information if they survived or not.Objective was to find a set of rules predicting if a passenger survives.


_Homework_: <br>
Find manually a decision tree predicting the survival/death of the passengers<br>


_personal summary_:<br>
It is an interesting game, to try to find rules, which predict who survived - but very soon you start defining rules w/o any link to real-world understanding. From the very beginning this exercise showed how easy you can overfit to the training data.


###Boston Housing 
_used data_: <br>
Data source was from [UCI machine learning repositary](https://archive.ics.uci.edu/ml/datasets/Housing). It contains a list with prices of houses as labels as well as some features(number of rooms, as well as 2 features indicating the social class of the neighbor hood and the quality of schools nearby) on which the price may depend on. Data preparation was already done by Udacity team. Objective was to predict house prices in the Boston area.


_Homework_: <br>
This project is at the end of the lesson discussing performance modell, error metrics and the idea of cross validation. So it focussed on implemting these: r2-score was used as performance model, test/training data where seperated, and for training we had to use k-fold shuffle split, and run a cross validation grid search. As regressor a decision tree was used. Grid search alternated the maxDepth size. The learning curve show clearly, how overfitting increased with high depth of the decision tree. furthermore they are a nice example for high bias, indicating to little complexity of the model, and for variance, indicating to high complixity.Last figure summarized it by displaying the performance of training/test in dependency of the max depth.


_personal summary_:<br>
Nice to find the theory we discussed in this lesson practically implemented.

####finding donors

_used data_: <br>
Objective was to accurately model individuals' income using data collected from the 1994 U.S. Census. 


_Homework_:<br>
Data has to be prepared (normalized and encoded). For the prediction we had to choice 3 different estimator, and to find arguments, why I decided for them, and which real life application exist. After the initial run, the best of them had to optimized.


_personal summary_:<br>
I found it extremly difficult to decide which algorithm to start with. There are few cheat sheets, an dif they are they propose completly different estimators to start with. Same with finding a concrete link between estimator and real life application. It was quite interesting to play around, and to tune - and to visualize the results. I gained a lot of additional insights by implementing an additional plot.



## Licence

None, as these are just project home works!


