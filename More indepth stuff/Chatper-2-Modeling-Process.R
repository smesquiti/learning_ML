#learning how to use machine learning with R 

#load the necessary packages 

library(pacman)
pacman::p_load(tidyverse, rsample, caret, h2o,AmesHousing,rsample,dslabs) #need the last three to get our data

# h2o set-up 
h2o.no_progress()  # turn off h2o progress bars
h2o.init()         # launch h2o

#get our data 
# access data
ames <- AmesHousing::make_ames()
mnist <- dslabs::read_mnist()

# Ames housing data
ames <- AmesHousing::make_ames()
ames.h2o <- as.h2o(ames)

# Job attrition data
churn <- rsample::attrition %>% 
  mutate_if(is.ordered, .funs = factor, ordered = FALSE)
churn.h2o <- as.h2o(churn)


# response variable
head(ames$Sale_Price)

#regression prediciton: predicting a continuous outcome 

#classification: predicting a binary or multinomial response
  #we often want to predict the probability of a particular class

#Data splitting

#strive for generalizability in our model, that is the one that most accurately predicts our values 

#when splitting data, need to save enough to test unbiasedly 

#simple random sampling

#Sampling is a random process so setting the random number generator with a common seed allows for reproducible results. 

#using 70-30 split
set.seed(123)
index_1 <- sample(1:nrow(ames), round(nrow(ames) * 0.7))
train_1 <- ames[index_1, ]
test_1  <- ames[-index_1, ]

# Using caret package
set.seed(123)  # for reproducibility
index_2 <- createDataPartition(ames$Sale_Price, p = 0.7, 
                               list = FALSE)
train_2 <- ames[index_2, ]
test_2  <- ames[-index_2, ]

# Using rsample package
set.seed(123)  # for reproducibility
split_1  <- initial_split(ames, prop = 0.7)
train_3  <- training(split_1)
test_3   <- testing(split_1)

# Using h2o package
split_2 <- h2o.splitFrame(ames.h2o, ratios = 0.7, 
                          seed = 123)
train_4 <- split_2[[1]]
test_4  <- split_2[[2]]


#Stratified Sampling 

#This is more common with classification problems where the response variable may be severely imbalanced 
#(e.g., 90% of observations with response “Yes” and 10% with response “No”).

#The easiest way to perform stratified sampling on a response variable is to use the rsample package,
#where you specify the response variable to stratafy

# orginal response distribution
table(churn$Attrition) %>% prop.table()
set.seed(123)
split_strat  <- initial_split(churn, prop = 0.7, 
                              strata = "Attrition")
train_strat  <- training(split_strat)
test_strat   <- testing(split_strat)

#crea# Sale price as function of neighborhood and year sold
#model_fn represents our function to fit whatever model... and is not an actual function 

model_fn(Sale_Price ~ Neighborhood + Year_Sold, 
         data = ames)


#meta engine (aggregator) that allows you to apply almost any direct engine with method = "<method-name>".

lm_lm    <- lm(Sale_Price ~ ., data = ames)
lm_glm   <- glm(Sale_Price ~ ., data = ames, 
                family = gaussian)
lm_caret <- train(Sale_Price ~ ., data = ames, 
                  method = "lm")

#Resampling methods 

#validation approach, which involves splitting the training set further to create two parts (as in Section 2.2): a training set and a validation set (or holdout set). 

#k-fold cross validation 

#fit on  k−1 folds and then the remaining fold is used to compute model performance

#the k-fold CV estimate is computed by averaging the k test errors, providing us with an approximation of the error we might expect on unseen data.

#an example

h2o.cv <- h2o.glm(
  x = x, 
  y = y, 
  training_frame = ames.h2o,
  nfolds = 10  # perform 10-fold CV
)

vfold_cv(ames, v = 10)

#bootstrapping
  #sampling with replacement: after a data point is selected for inclusion in the subset, it’s still available for further selection

#bootstrap sampling will contain approximately the same distribution of values (represented by colors) as the original data set.

#Since observations are replicated in bootstrapping, there tends to be less variability in the error measure compared with k-fold CV

#However, this can also increase the bias of your error estimate. This can be problematic with smaller data sets; however, for most average-to-large data sets 

#creating bootstrap samples 
bootstraps(ames, times = 1e4)

#Bias variance trade-off 
#Prediction errors can be decomposed into two important subcomponents: error due to “bias” and error due to “variance”.

#Bias: Bias is the difference between the expected (or average) prediction of our model and the correct value which we are trying to predict.

#Linear models are classical examples of high bias models as they are less flexible and rarely capture non-linear, non-monotonic relationships.

#Variance: the variability of a model prediction for a given data point

#can use grid searching to find optimal number of hyperparameters

#A grid search is an automated approach to searching across many combinations of hyperparameter values.

#model evaluation 

#assess the predictive accuracy via loss functions. 
#Loss functions are metrics that compare the predicted values to the actual value (the output of a loss function is often referred to as the error or pseudo residual). 

#Putting it all together 

## Stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.7, #70/30 on sale price 
                       strata = "Sale_Price")

ames_train  <- training(split) #training dataset
ames_test   <- testing(split) #testing dataset 

#next we'll appply a k-nearest neighbor regressor to our data

#To do so, we’ll use caret, which is a meta-engine to simplify the resampling, grid search, and model application processes. The following defines
# Specify resampling strategy
cv <- trainControl(
  method = "repeatedcv", 
  number = 10, #number or cross validation splits
  repeats = 5 #repeat this 5x
)

# Create grid of hyperparameter values
hyper_grid <- expand.grid(k = seq(2, 25, by = 1)) #go from 2 to 25 by 1 #tells ys

# Tune a knn model using grid search
knn_fit <- train(
  Sale_Price ~ ., 
  data = ames_train, 
  method = "knn", 
  trControl = cv, #using the CV object we specified earlier 
  tuneGrid = hyper_grid, #hypergrid we created earlier 
  metric = "RMSE" #what metric we wanna evaluate on
)
#plot it to see the optimal number of parameters
ggplot(knn_fit)

#can see that 6 is the optimal number of hyperparameters


