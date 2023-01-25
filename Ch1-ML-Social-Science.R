#Ch1 Machine Learning for Social Scientists 

library(pacman)
pacman::p_load(tidyverse, patchwork, scales, tidymodels, install = T)

## Data creation
set.seed(2313)
n <- 500
x <- rnorm(n)
y <- x^3 + rnorm(n, sd = 3)
age <- rescale(x, to = c(0, 100))
income <- rescale(y, to = c(0, 5000))
age_inc <- data.frame(age = age, income = income)
## Data creation

y_axis <- scale_y_continuous(labels = dollar_format(suffix = "€", prefix = ""),
                             limits = c(0, 5000),
                             name = "Income")

x_axis <- scale_x_continuous(name = "Age")

bad_fit <-
  ggplot(age_inc, aes(age, income)) +
  geom_point() +
  geom_smooth(method = "lm") +
  y_axis +
  x_axis +  
  ggtitle("Underfit") +
  theme_linedraw()

overfit <-
  ggplot(age_inc, aes(age, income)) +
  geom_point() +
  geom_smooth(method = "loess", span = 0.015) +
  y_axis +
  x_axis +  
  ggtitle("Overfit") +
  theme_linedraw()

goodfit <-
  ggplot(age_inc, aes(age, income)) +
  geom_point() +
  geom_smooth(method = "loess", span = 0.9) +
  y_axis +
  x_axis +  
  ggtitle("Ideal fit") +
  theme_linedraw()

bad_fit + overfit + goodfit

#install tidyflow 

install.packages("devtools")
devtools::install_github("cimentadaj/tidyflow")

library(tidyflow)
ml_flow <-
  age_inc %>%
  tidyflow(seed = 2313) %>%
  plug_split(initial_split)

ml_flow

ml_flow <-
  ml_flow %>%
  plug_resample(vfold_cv)

ml_flow


#an example 

rescale <- function(x, to = c(0, 1), from = range(x, na.rm = TRUE, finite = TRUE)) {
  (x - from[1])/diff(from) * diff(to) + to[1]
}

set.seed(2313)
n <- 500
x <- rnorm(n)
y <- x^3 + rnorm(n, sd = 3)
age <- rescale(x, to = c(0, 100))
income <- rescale(y, to = c(0, 5000))

age_inc <- data.frame(age = age, income = income)

#plot the relationship between age and income
age_inc %>% 
  ggplot(aes(age, income)) +
  geom_point() +
  geom_smooth()
  theme_linedraw()
  
#now create a tidyflow to depsarate the data into training data
m1_flow <- 
  age_inc %>% 
  tidyflow(seed = 2313) %>% 
  plug_split(initial_split) #using initial split, divides the data
m1_flow


#now let's start to run the data

#run a simple model
m1 <- 
  m1_flow %>% 
  plug_recipe(~recipe(income ~ age, data = .)) %>% #add the formula 
  plug_model(linear_reg() %>% set_engine('lm')) %>%  #define the linear regression 
  fit()
#predict on the training data

m1_res <- 
  m1 %>% 
  predict_training()
m1_res

#The result of predict_training is the training data from age_inc with one new column: the predicted values of the model. Let’s visualize the predictions:

#visualize the predictions
m1_res %>% 
  ggplot(aes(age, income)) +
  geom_line(aes(y = .pred), color = 'red', size = 2) +
  geom_point() + 
  scale_x_continuous(name = "Age") +
  scale_y_continuous(name = "Income",
                     label = dollar_format(suffix = "$", prefix = "")) +
  theme_linedraw()

#starting to apply more 

#Define the formula of your model and specify that the 'polynomial' 
#value will be 'tuned'. That is, we will try several values instead of just one 

rcp <- 
  ~recipe(income ~ age, data = .) %>% 
  step_poly(age, degree = tune()) #specifying that we wanna use a polynomial 

#model 2
m2 <- 
  m1 %>% 
  plug_resample(vfold_cv) %>% #using the vfold_cv validation function 
  #replace the initial recipe with one of the several polynomials 
  replace_recipe(rcp) %>% 
  #define the values we will try, from 2 to 10 
  plug_grid(expand.grid, degree = 2:10) %>% 
  fit()
#takes a few to run 

# Visualize the result
m2 %>%
  pull_tflow_fit_tuning() %>% # Extract all models with different 'degree' values
  autoplot() +
  theme_linedraw()

#Given that most of the polynomial terms have similar error terms, we usually would go for the simplest model, that is, the model with age^3

#fit the final with the third degree polynomial 
res_m2 <- complete_tflow(m2, best_params = data.frame(degree = 3))

#visualize 
res_m2 %>% 
  predict_training() %>% 
  ggplot(aes(age,income)) +
  geom_line(aes(y = .pred), color = "red", size = 2) +
  geom_point() +
  scale_x_continuous(name = "Age") +
  scale_y_continuous(name = "Income", label = dollar_format(suffix = "", prefix = "$")) +
  theme_linedraw()

#comapre the RMSE of the training data to the testing data 
res_m2 %>% 
  predict_testing() %>% #note this is what changed 
  ggplot(aes(age, income)) +
  geom_line(aes(y = .pred), color = 'red', size = 2) + 
  geom_point()+
  scale_x_continuous(name = "Age") +
  scale_y_continuous(name = "Income", label = dollar_format(suffix = "", prefix = "$")) +
  theme_linedraw()
