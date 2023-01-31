#Chapter 7 Multivariate Adaptive Regression Splines

#load in the necessary packages 
library(pacman)
pacman::p_load(tidyverse, earth, caret, vip, pdp, AmesHousing, install = T)

#traditionally use polynomials for non-linear relationships but they can often become unintepretable 

#Multivariate adaptive regression splines 

#Multivariate adaptive regression splines (MARS) provide a convenient approach to capture the nonlinear relationships in the data by assessing cutpoints (knots) similar to step functions.

#v The procedure assesses each data point for each predictor as a knot and creates a linear regression model with the candidate feature(s)

#once the full set of knots has been identified, we can sequentially remove knots that do not contribute significantly to predictive accuracy.

#Fitting a basic MARS model

#We can fit a direct engine MARS model with the earth package

#earth::earth() will assess all potential knots across all supplied features and then will prune to the optimal number of knots based on an expected change in  

#fit a basic MARS model on the ames data

mars1 <- earth(
Sale_Price ~., 
data = ames_train
)

#print the summary 
print(mars1)

#It also shows us that 36 of 39 terms were used from 27 of the 307 original predictors. But what does this mean? If we were to look at all the coefficients, we would see that there are 36 terms in our model (including the intercept).

#Looking at the first 10 terms in our model, we see that Gr_Liv_Area is included with a knot at 2787

summary(mars1) %>%  .$coefficients %>% head(10)

#numbers in teh model tells us where notes were included

#plot it to visualize where the optimal number of tems is at 
plot(mars1, which = 1)

#can see that the optimal number of terms is 43, where changed in R^2 is < 0.001

#we can also assess potential interactiosn betwen different hinge functions 

#Fit the basic mars model
mars2 <- earth(
  Sale_Price ~., 
  data = ames_train,
  degree = 2) # what me modify to include interactions

summary(mars2) %>% .$coefficients %>% head(10)


#Tuning