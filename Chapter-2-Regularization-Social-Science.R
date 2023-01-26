#Chapter-2-Regularization 

library(pacman)
pacman::p_load(tidyverse, tidymodels, tidyflow)

data_link <- "https://raw.githubusercontent.com/cimentadaj/ml_socsci/master/data/pisa_us_2018.csv"
pisa <- read.csv(data_link)


#residual sum of squares falls into the loss function family 

#The strength of the ridge regression comes from the fact that it compromises fitting the training data really well for improved generalization

#n other words, the whole gist behind ridge regression is penalizing very large coefficients for better generalization on new data.

# Specify all variables and scale
rcp <-
  # Define dependent (math_score) and independent variables
  ~ recipe(math_score ~ MISCED + FISCED + HISEI + REPEAT + IMMIG + DURECEC + BSMJ, data = .) %>% 
  # Scale all predictors (already knows it's the independent variables)
  step_scale(all_predictors()) #scale them baby, so none get penalized more than others

#build our tidyflow 

tflow <- 
  tidyflow(seed = 231141) %>% 
  plug_data(pisa) %>% 
  plug_split(initial_split, prop = .7) %>% #do a 70/30 split on the data
  # add the recope with all variables 
  plug_recipe(rcp) #plug the recipe 
tflow

#ridge regression 

############################# Ridge regression ################################
###############################################################################
regularized_reg <-
  set_engine(
    # mixture specifies the type of penalized regression: 0 is ridge regression
    linear_reg(penalty = 0, mixture = 0), #controls how much weight do we want to give to the “shrinkage penalty” if it is 0 then it's no weight
    "glmnet"
  )

model1 <- 
  tflow %>% 
  plug_model(regularized_reg) %>% 
  fit()

#get ridge coefficients 
mod <- model1 %>% pull_tflow_fit() %>% .[["fit"]]
ridge_coef <- predict(mod, s = 0, type = "coefficients")

############################# Linear model ####################################
###############################################################################

model2 <-
  tflow %>%
  plug_model(set_engine(linear_reg(), "lm")) %>%
  fit()

lm_coef <- model2 %>% pull_tflow_fit() %>% .[["fit"]] %>% coef()

############################# Comparing model #################################
###############################################################################

comparison <-
  data.frame(coefs = names(lm_coef),
             `Linear coefficients` = unname(round(lm_coef, 2)),
             `Ridge coefficients` = round(as.vector(ridge_coef), 2))

knitr::kable(comparison)

#now let's use CV to tune and find the best models 

#build tidyflow to add CV and grid
tflow <- 
  tflow %>% 
  #cross validation 
  plug_resample(vfold_cv, v = 5) %>%  #five folds
  #grid
  plug_grid(grid_regular)

#regularized regression 
regularized_reg <- update(regularized_reg, penalty = tune()) #update our past model with the tune

res <- 
  tflow %>% 
  #update the model adn specify that the penalty will be tuned 
  plug_model(regularized_reg) %>% 
  fit()

final_ridge <- complete_tflow(res, metric = 'rmse') #evaluating on RMSE

#graph it
final_ridge %>% 
  pull_tflow_fit() %>% 
  .[["fit"]] %>% 
  plot(xvar = "lambda", label = T)

#figure out the best lambda now 
best_tune <- 
  res %>% #pass the model through our pipeline
  pull_tflow_fit_tuning() %>% 
  select_best(metric = "rmse")
best_tune

#we dont need to calculate this, we can use functions to find this 

#complete_tflow can help us 

#now see the RMSE for the training models and see which is the best 
train_rmse_ridge <-
  final_ridge %>%
  predict_training() %>% #training dataset 
  rmse(math_score, .pred)

holdout_ridge <-
  final_ridge %>%
  predict_testing() %>% #testing dataset 
  rmse(math_score, .pred)

train_rmse_ridge$type <- "training"
holdout_ridge$type <- 'testing'

ridge <- as.data.frame(rbind(train_rmse_ridge, holdout_ridge))
ridge$model <- 'ridge'
ridge

#The testing error (RMSE) is higher than the training error, as expected, as the training set nearly always memorizes the data better for the training.


#lasso ridge regression 
#The Lasso regularization is very similar to the ridge regularization where only one thing changes: the penalty term. 
#Instead of squaring the coefficients in the penalty term, the lasso regularization takes the absolute value of the coefficient.
#can force a coefficient to be exactly zero if they dont add anything to the model 

regularized_reg <- update(regularized_reg, mixture = 1) #build the model 

res <-
  tflow %>%
  plug_model(regularized_reg) %>%
  fit()

final_lasso <- complete_tflow(res, metric = "rmse")

final_lasso %>%
  pull_tflow_fit() %>%
  .[["fit"]] %>%
  plot(xvar = "lambda", label = TRUE)

#lasso also removes redundant variables 

#lets check the fit of the final model and it's error
train_rmse_lasso <- 
  final_lasso %>% 
  predict_training() %>% 
  rmse(math_score, .pred)

holdout_lasso <-
  final_lasso %>% 
  predict_testing() %>% 
  rmse(math_score, .pred)

train_rmse_lasso$type <- "training"
holdout_lasso$type <- "testing"

lasso <- as.data.frame(rbind(train_rmse_lasso, holdout_lasso))
lasso$model <- "lasso"
lasso

#check which model is performing better 
model_comparison <- rbind(lasso,ridge)
model_comparison

#while the difference is miniscule, if there were more predicitors lasso would probably be better

#Elastic Net regularization 
#combines both penalties to form a single equation 
#basically ridge and lasso

#Essentially, you now have two tuning parameters. 
#In the grid of values, instead of specifying a mixture of 0 (ridge) or 1 (lasso), 
#tidyflow will slide through several values of mixture ranging from 0 to 1 and compare that to several values of lambda.

#construct the tidy flow and models 

regularized_reg <- update(regularized_reg, mixture = tune())

res <- 
  tflow %>% 
  plug_model(regularized_reg) %>% 
  fit()

final_elnet <- complete_tflow(res, metric = 'rmse') #fis the best model from the search 

train_rmse_elnet <- #training model 
  final_elnet %>%
  predict_training() %>%
  rmse(math_score, .pred)

holdout_elnet <- #testing model 
  final_elnet %>%
  predict_testing() %>%
  rmse(math_score, .pred)

train_rmse_elnet$type <- "training"
holdout_elnet$type <- "testing"

elnet <- as.data.frame(rbind(train_rmse_elnet, holdout_elnet))
elnet$model <- "elnet"
elnet

model_comparison <- rbind(model_comparison, elnet)

model_comparison %>%
  ggplot(aes(model, .estimate, color = type, group = type)) +
  geom_point(position = "dodge") +
  geom_line() +
  scale_y_continuous(name = "RMSE") +
  scale_x_discrete(name = "Models") +
  theme_minimal()

#working with the example 

#read in the data
data_link <- "https://raw.githubusercontent.com/cimentadaj/ml_socsci/master/data/pisa_us_2018.csv"
pisa <- read.csv(data_link)

#create the tidyflow with a split
tflow <- 
  pisa %>% 
  tidyflow(seed = 2341) %>% 
  plug_split(initial_split, prop = 0.7)
tflow

#run a ridge regression with noncognitive as a DV 

#first need to build our recipe 
# Specify all variables and scale

ridge_mod <- set_engine(linear_reg(penalty = 0.001, mixture = 0), "glmnet") #build out ridge_model

#build the tidy flow 
tflow <- 
  tflow %>% 
  plug_formula(noncogn ~ .) %>% #regress on all variables 
  plug_model(ridge_mod)

#fit our model 
m1 <- fit(tflow)

#get the metrics 
m1_rsq <- predict_training(m1) %>%  rsq(noncogn, .pred)
m1_rmse <- predict_training(m1) %>%  rmse(noncogn, .pred)

#add a recipe to scale all of the predictors and then rerun our model 
# Specify all variables and scale
rcp <-
  # Define dependent (math_score) and independent variables
  ~ recipe(math_score ~ ., data = .) %>% 
  # Scale all predictors (already knows it's the independent variables)
  step_scale(all_predictors()) #scale them baby, so none get penalized more than others

#build our tidyflow

tflow <- 
  tidyflow(seed = 2314) %>% 
  plug_data(pisa) %>% 
  plug_split(initial_split, prop = .7) %>% #do a 70/30 split on the data
  plug_model(ridge_mod) %>% 
  # add the recope with all variables 
  plug_recipe(rcp) #plug the recipe 
tflow

m2 <- fit(tflow)

m2_rsq <- predict_training(m2) %>% rsq(noncogn, .pred)
m2_rmse <- predict_training(m2) %>% rmse(noncogn, .pred)


#adapt the previous model to do a grid search of pnealt values 

ridge_mod <- update(ridge_mod, penalty = tune()) #update with the penalty 

tflow <-
  tflow %>%
  replace_model(ridge_mod) %>% #specify the model we want to modify 
  plug_resample(vfold_cv) %>% #we are using CV
  plug_grid(grid_regular, levels = 10)

m3 <- fit(tflow)

#compare the two types
m3 %>%
  pull_tflow_fit_tuning() %>%
  autoplot()

#run a lasso regression with the same specification 
lasso_mod <- update(ridge_mod, mixture = 1 ) # the mixture part specifies we want lasso 

#now rund the grid search 
m4 <-
  tflow %>%
  replace_model(lasso_mod) %>% #need to specift we replaced the model here!
  fit()
m4 %>% 
  pull_tflow_fit_tuning() %>%
  autoplot()

#run an elastic fit regression 
elnet_mod <- update(lasso_mod, mixture = tune())

m5 <-
  tflow %>%
  replace_model(elnet_mod) %>%
  fit()

m5 %>%
  pull_tflow_fit_tuning() %>%
  autoplot()

# Additional plot with standard error
library(tidyr)
m5 %>%
  pull_tflow_fit_tuning() %>%
  collect_metrics() %>%
  pivot_longer(penalty:mixture) %>%
  mutate(low = mean - (std_err * 2),
         high = mean + (std_err * 2)) %>% 
  ggplot(aes(value, mean)) +
  geom_point() +
  geom_errorbar(aes(ymin = low, ymax = high)) +
  facet_grid(.metric ~ name)


# Since we will be repeating the same process many times
# let's write a function to predict on the training/testing
# and combine them. This function will accept a single
# model and produce a data frame with the RMSE error for
# training and testing. This way, we can reuse the code
# without having to copy everything many times
calculate_err <- function(final_model, type_model = NULL) {
  final_model <- complete_tflow(final_model, metric = "rmse")
  err_train <-
    final_model %>%
    predict_training() %>%
    rmse(noncogn, .pred)
  
  err_test <-
    final_model %>%
    predict_testing() %>%
    rmse(noncogn, .pred)
  
  err_train$type <- "train"
  err_test$type <- "test"
  res <- as.data.frame(rbind(err_train, err_test))
  res$model <- type_model
  res
}

final_res <-
  rbind(
    calculate_err(m3, "ridge"),
    calculate_err(m4, "lasso"),
    calculate_err(m5, "elnet")
  )

final_res %>%
  ggplot(aes(model, .estimate, color = type)) +
  geom_point() +
  theme_minimal()

## BONUS
## Fit a linear regression and compare the four models
## What is the best model to pick considering both accuracy and simplicity?