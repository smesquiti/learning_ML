#Chapter 3: Feature and Target Engineering 

library(pacman)
pacman::p_load(tidyverse, visdat, caret, recipes, forecast,AmesHousing)

#sometimes it can be helpful to transform data, esp when using parametric techniques 

transfromed_response <- log(ames_train$Sale_Price)

#can also use the recipe function to preprocess

ames_recipe <- recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_log(all_outcomes())
ames_recipe #saving as an object; the log of all outcomes

#can also apply box cox transformation 
# The “optimal value” is the one which results in the best transformation to an approximate normal distribution.

# Log transform a value
y <- log(10)

# Undo log-transformation
exp(y)

#box cox transform a value 
y <- forecast::BoxCox(10,lambda = "auto")
y


# Inverse Box Cox function
inv_box_cox <- function(x, lambda) {
  # for Box-Cox, lambda = 0 --> log transform
  if (lambda == 0) exp(x) else (lambda*x + 1)^(1/lambda) 
}

# Undo Box Cox-transformation
inv_box_cox(y, lambda)

#missing data can sometimes be a pain. should ID if it is missing at random or if there is some information that may be derived from it

#get a sense of how many rows have missing data 
sum(is.na(AmesHousing::ames_raw))

#can visualize the missing data

AmesHousing::ames_raw %>%
  is.na() %>%
  reshape2::melt() %>%
  ggplot(aes(Var2, Var1, fill=value)) + 
  geom_raster() + 
  coord_flip() +
  scale_y_continuous(NULL, expand = c(0, 0)) +
  scale_fill_grey(name = "", 
                  labels = c("Present", 
                             "Missing")) +
  xlab("Observation") +
  theme(axis.text.y  = element_text(size = 4))

AmesHousing::ames_raw %>% 
  filter(is.na(`Garage Type`)) %>% 
  select(`Garage Type`, `Garage Cars`, `Garage Area`)

#Can also use another package with less effort to visualize 
vis_miss(AmesHousing::ames_raw, cluster = TRUE)

#can use data imputations to deal with missing data 

#we should do data imputation within the resampling process 

#we can impute using descriptive statistics (not recommended)
ames_recipe %>%
  step_medianimpute(Gr_Liv_Area)

#can also use k-nearest neighbor
#imputes values by identifying observations with missing values, then identifying other observations that are most similar based on the other available features, and using the values from these nearest neighbor observations to impute missing values.

ames_recipe %>%
  step_knnimpute(all_predictors(), neighbors = 6) #can change the # of neighbors

#can use some approaches for decision trees as well i.e., Bagged trees

#feature filtering 

#at times it's important to be utilitarian in selecting parameters

#easy to eliminate variables that don't really contribute to our model


#A rule of thumb for detecting near-zero variance features is:
#The fraction of unique values over the sample size is low (say  ≤ 10%)
#The ratio of the frequency of the most prevalent value to the frequency of the second most prevalent value is large (say >20%

caret::nearZeroVar(ames_train, saveMetrics = TRUE) %>% 
  tibble::rownames_to_column() %>% 
  filter(nzv)
#inspect adn see that we have a few variables that meet this criteria                                                                                                                     

caret::nearZeroVar(ames_train, saveMetrics = TRUE) %>% 
  tibble::rownames_to_column() %>% 
  filter(nzv)

#dealing with skewed distributions: Normalizing and standardizing heavily skewed features can help minimize these concerns
#helps deal with some of the issues that skewed data introduce

#dealing with skewness 

#When normalizing many variables, it’s best to use the Box-Cox (when feature values are strictly positive) or 
#Yeo-Johnson (when feature values are not strictly positive) procedures as these methods will identify if a transformation is required and what the optimal transformation will be

