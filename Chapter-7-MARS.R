#Chapter 7 Multivariate Adaptive Regression Splines

#load in the necessary packages 
library(pacman)
pacman::p_load(tidyverse, earth, caret, vip, pdp, AmesHousing, rsample, install = T)

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

#wo important tuning parameters associated with our MARS model: the maximum degree of interactions and the number of terms retained in the final model. 

#need to perform CV to find the optinal combo of hyperparameters and minimize prediction error. 

#create tuning grid 

hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

head(hyper_grid)

#use the caret package to cross-validate the model 
set.seed(123) #for repro :) 
cv_mars <- train(
  x = subset(ames_train, select = -Sale_Price),#select everything sans Sale_Price
  y = ames_train$Sale_Price,
  method = "earth",
  metric = "RMSE", #metric we want to evaluate on 
  trControl = trainControl(method = "cv", number = 10), 
  tuneGrid = hyper_grid
)

#find the best tuned model 
cv_mars$bestTune

#get more summart results 
cv_mars$results %>% 
  filter(nprune == cv_mars$bestTune$nprune, degree == cv_mars$bestTune$degree)

#plot it 
ggplot(cv_mars)


#let's refine our grid search  to 45-65 terms retrained 
new_hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(45, 65, length.out = 10) %>% floor()
)

head(new_hyper_grid)

#new cross-validated model
set.seed(123)  # for reproducibility
new_cv_mars <- train(
  x = subset(ames_train, select = -Sale_Price),
  y = ames_train$Sale_Price,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = new_hyper_grid
)

#find the best tuned model 
new_cv_mars$bestTune

#get more summart results 
new_cv_mars$results %>% 
  filter(nprune == new_cv_mars$bestTune$nprune, degree == new_cv_mars$bestTune$degree)

#plot it 
ggplot(new_cv_mars)



#Feature interpretation 

#mars also kicks out unimportant values
p1 <- vip(cv_mars, num_features = 40, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(cv_mars, num_features = 40, geom = "point", value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)

#Its important to realize that variable importance will only measure the impact of the prediction error as features are included; 
#however, it does not measure the impact for particular hinge functions created for a given feature.

#look for interaction in our models to see different hinge functions

cv_mars$finalModel %>%
  coef() %>%  
  broom::tidy() %>%  
  filter(stringr::str_detect(names, "\\*"))


#we can also construct partial dependence plots to visualize the relationship between these features 

#construct partial dependence plots
p1 <- partial(cv_mars, pred.var = "Gr_Liv_Area", grid.resolution = 10) %>% 
  autoplot()
p2 <- partial(cv_mars, pred.var = "Year_Built", grid.resolution = 10) %>% 
  autoplot()
p3 <- partial(cv_mars, pred.var = c("Gr_Liv_Area", "Year_Built"), 
              grid.resolution = 10) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))

#display plots side by side

gridExtra::grid.arrange(p1,p2,p3, ncol = 3)

#Partial dependence plots to understand the relationship between Sale_Price and the Gr_Liv_Area and Year_Built features. The PDPs tell us that as Gr_Liv_Area increases and for newer homes, Sale_Price increases dramatically.

#Attrition data babyyyy

df <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE) #read in the data

#Create the training and test set 

set.seed(123)
churn_split <- rsample::initial_split(df, prop = 0.7, strata = "Attrition")
churn_test <- rsample::testing(churn_split)
churn_train <- rsample::training(churn_split)

#build our tuning grid

hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

#for reproducibility 
set.seed(123)
#cross-validated model 
tunded_mars <- train(
  x = subset(churn_train, select = -Attrition),
  y = churn_train$Attrition, 
  method = 'earth',
  trControl= trainControl(method = 'cv', number = 10),
  tuneGrid = hyper_grid #object we built earlier 
  
)

#find the best model
tunded_mars$bestTune

#Plot the results 
ggplot(tunded_mars)
