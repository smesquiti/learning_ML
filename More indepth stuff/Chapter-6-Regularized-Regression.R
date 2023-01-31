#Chapter 6 - Regularized Regression

library(pacman)
pacman::p_load(tidyverse, recipes, glmnet, caret, vip, AmesHousing, install = T)

#Regularization methods provide a means to constrain or regularize the estimated coefficients, which can reduce the variance and decrease out of sample error.

#Using Ames training data 

#regularized regression is great for datasets that have a high degree of dimensional-ity (text data)

#often called penalized regression 

#This penalty parameter constrains the size of the coefficients such that the only way the coefficients can increase is if we experience a comparable decrease in the sum of squared errors (SSE)

#implementation 

#creating our datasets we can use 
# Stratified sampling with the rsample package
set.seed(123)
split <- initial_split(ames, prop = 0.7, #70/30 split
                       strata = "Sale_Price")
ames_train  <- training(split)
ames_test   <- testing(split)

# Create training  feature matrices
# we use model.matrix(...)[, -1] to discard the intercept
X <- model.matrix(Sale_Price ~ ., ames_train)[, -1] #creating out x variable discarding the intercept 

# transform y with log transformation
Y <- log(ames_train$Sale_Price) #creating our y variable 

#we need to ensure our coefficients are on a common scale.
#If not, then predictors with naturally larger values (e.g., total square footage) will be penalized more than predictors with naturally smaller values (e.g., total number of rooms).

# Apply ridge regression to ames data, function also standardizes stuff before hand 
ridge <- glmnet(
  x = X,
  y = Y,
  alpha = 0 # ridge (alpha = 0), lasso (alpha = 1), or elastic net (0 < alpha < 1)
)

plot(ridge, xvar = "lambda")

# lambdas applied to penalty parameter
ridge$lambda %>% head()

# small lambda results in large coefficients
coef(ridge)[c("Latitude", "Overall_QualVery_Excellent"), 100]

# large lambda results in small coefficients
coef(ridge)[c("Latitude", "Overall_QualVery_Excellent"), 1] 

#Tuning 

#we can also tune things using cross-validation 

#use this function to perform 10-fold cross validation 

# Apply CV ridge regression to Ames data
ridge <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 0
)

# Apply CV lasso regression to Ames data
lasso <- cv.glmnet(
  x = X,
  y = Y,
  alpha = 1
)

# plot results
par(mfrow = c(1, 2))
plot(ridge, main = "Ridge penalty\n\n")
plot(lasso, main = "Lasso penalty\n\n")

# Ridge model
min(ridge$cvm)       # minimum MSE
ridge$lambda.min     # lambda for this min MSE

# Lasso model
min(lasso$cvm)       # minimum MSE
lasso$lambda.min     # lambda for this min MSE

lasso$nzero[lasso$lambda == lasso$lambda.min] # No. of coef | Min MSE

lasso$cvm[lasso$lambda == lasso$lambda.1se]  # 1-SE rule

lasso$nzero[lasso$lambda == lasso$lambda.1se] # No. of coef | 1-SE MSE

#recall we defined x and y earlier in the script 
# Ridge model
ridge_min <- glmnet(
  x = X,
  y = Y,
  alpha = 0
)

# Lasso model
lasso_min <- glmnet(
  x = X,
  y = Y,
  alpha = 1
)

par(mfrow = c(1, 2))
# plot ridge model
plot(ridge_min, xvar = "lambda", main = "Ridge penalty\n\n")
abline(v = log(ridge$lambda.min), col = "red", lty = "dashed")
abline(v = log(ridge$lambda.1se), col = "blue", lty = "dashed")

# plot lasso model
plot(lasso_min, xvar = "lambda", main = "Lasso penalty\n\n")
abline(v = log(lasso$lambda.min), col = "red", lty = "dashed")
abline(v = log(lasso$lambda.1se), col = "blue", lty = "dashed")

#alpha value between 0–1 will perform an elastic net. 
#When alpha = 0.5 we perform an equal combination of penalties whereas alpha  
#<  0.5 will have a heavier ridge penalty applied and alpha   > 0.5 will have a heavier lasso penalty.

#Often, the optimal model contains an alpha somewhere between 0–1, thus we want to tune both the λ and the alpha parameters.

#use the caret package to automate tuning


# for reproducibility
set.seed(123)

# grid search across 
cv_glmnet <- train(
  x = X,
  y = Y,
  method = "glmnet",
  preProc = c("zv", "center", "scale"), #stuff for us to preprocess the data
  trControl = trainControl(method = "cv", number = 10), # our cross validation information 
  tuneLength = 10
)

#check to see whivh model has the best tune
cv_glmnet$bestTune

# results for model with lowest RMSE
cv_glmnet$results %>%
  filter(alpha == cv_glmnet$bestTune$alpha, lambda == cv_glmnet$bestTune$lambda)

# plot cross-validated RMSE
ggplot(cv_glmnet)

# predict sales price on training data
pred <- predict(cv_glmnet, X)

# compute RMSE of transformed predicted
RMSE(exp(pred), exp(Y))

#need to do feature interpretation using the VIP package 

vip(cv_glmnet, num_features = 20, geom = "point")

#Attrition data 
df <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE) #read in the data

# Create training (70%) and test (30%) sets for the
# rsample::attrition data. Use set.seed for reproducibility
set.seed(123) #for repro
churn_split <- initial_split(df, prop = .7, strata = "Attrition") #stratify on the DV
train <- training(churn_split) #training data
test  <- testing(churn_split) #testing data

# train logistic regression model (classic logit)
set.seed(123)
glm_mod <- train(
  Attrition ~ ., #training all variables
  data = train, 
  method = "glm",#standard GLM 
  family = "binomial",
  preProc = c("zv", "center", "scale"), #center, scale, and remove zero variance variables 
  trControl = trainControl(method = "cv", number = 10)
)

# train regularized logistic regression model
set.seed(123)
penalized_mod <- train(
  Attrition ~ ., 
  data = train, 
  method = "glmnet", #penalized
  family = "binomial",
  preProc = c("zv", "center", "scale"),
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


#now compare the two models 
# extract out of sample performance measures
summary(resamples(list(
  logistic_model = glm_mod, 
  penalized_model = penalized_mod
)))$statistics$Accuracy
