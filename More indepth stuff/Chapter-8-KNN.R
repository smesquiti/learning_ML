# Chapter 8 K-Nearest Neighbors
#load in the necessary packages 
library(pacman)
pacman::p_load(tidyverse, earth, caret, vip, pdp, AmesHousing, recipes, rsample, install = T)

#K-nearest neighbor (KNN) is a very simple algorithm in which each observation is predicted based on its “similarity” to other observations

#load in the data

# create training (70%) set for the rsample::attrition data.
attrit <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE)
set.seed(123)
churn_split <- initial_split(attrit, prop = .7, strata = "Attrition")
churn_train <- training(churn_split)

# import MNIST training data
mnist <- dslabs::read_mnist()
names(mnist)

#look at the two features 
(two_houses <- ames_train[1:2, c("Gr_Liv_Area", "Year_Built")])

#euclidean distance 
dist(two_houses, method = "euclidean")


#Preprocessing the data

#euclidean distance is sensitive to outliers and the scale of features, for example comparing years to square footage

#Choosing K 

#When using KNN for classification, it is best to assess odd numbers for  
#k to avoid ties in the event there is equal proportion of response levels 
#(i.e. when k = 2 one of the neighbors could have class “0” while the other neighbor has class “1”).

#create our blueprint

blueprint <- recipe(Attrition ~ ., data = churn_train) %>% 
  step_nzv(all_nominal()) %>% 
  step_integer(contains("Satisfaction")) %>% 
  step_integer(WorkLifeBalance) %>%
  step_integer(JobInvolvement) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())


#create a resampling method 
cv <- trainControl(
  method = 'repeatedcv', 
  number = 10,
  repeats = 5, 
  classProbs = T, 
  summaryFunction = twoClassSummary
)

#create a hyperparameter grid search
hyper_grid <- expand.grid(
  k = floor(seq(1, nrow(churn_train)/3, length.out = 20))
)

#fit knn model and perfom grid search 

#this takes a while to run 

knn_grid <- train(
  blueprint, 
  data = churn_train, 
  method = 'knn',
  trControl = cv, 
  tuneGrid = hyper_grid, 
  metric = "ROC"
)

#graph grid search 
ggplot(knn_grid)


#MNIST example