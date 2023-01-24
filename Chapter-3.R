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


#normalize all numeric columns 

recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_YeoJohnson(all_numeric())

#standardizing to help with data interpretation 

ames_recipe %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes())

#lumping can also be used but be careful when you do so 

#see how things are dispersed 
count(ames_train, Screen_Porch) %>% arrange(n)

#lump stuff together 
# Lump levels for two features
lumping <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_other(Neighborhood, threshold = 0.01, 
             other = "other") %>%
  step_other(Screen_Porch, threshold = 0.1, 
             other = ">0")

apply_2_training <- prep(lumping, training = ames_train) %>%
  bake(ames_train)

#new distribution of neighborhoods
count(apply_2_training, Neighborhood) %>% arrange(n)

# New distribution of Screen_Porch
count(apply_2_training, Screen_Porch) %>% arrange(n) #number of porches with and with Screens 

#one hot coding and dummy coding                                                   

# Lump levels for two features together 
recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_dummy(all_nominal(), one_hot = T) #step_dummy() will create a full rank encoding

#dummy and one hot coding can increase the dimensionality of your dataset. so be weary of how you label things 

#Label encoding is a pure numeric conversion of the levels of a categorical variable
#If a categorical variable is a factor and it has pre-specified levels then the numeric conversion will be in level order. 
#If no levels are specified, the encoding will be based on alphabetical order

#recode some factors in the dataset to numeric 
count(ames_train, MS_SubClass)

# Label encoded
recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(MS_SubClass) %>%
  prep(ames_train) %>%
  bake(ames_train) %>%
  count(MS_SubClass)

#need to be careful with the order though 

#when there is a rank order (e.g., very bad -- very good) it's better to work with 
#that is, ordinal data is easier to work with 

ames_train %>% select(contains("Qual"))

#original categories
count(ames_train, Overall_Qual)


#now make it numeric 
recipe(Sale_Price ~., data = ames_train) %>% 
  step_integer(Overall_Qual) %>% 
  prep(ames_train) %>% 
  bake(ames_train) %>% 
  count(Overall_Qual)

#now we can see how it was converted to integers

#there are also some alternatives 

#target encoding is the process of replacing a categorical value with the mean (regression) or proportion (classification) of the target variable

#Target encoding runs the risk of data leakage since you are using the response variable to encode a feature

#dimension reduction is an alternative approacb to filter out non-informative features without manually removing them 

#using PCA on the data 
recipe(Sale_Price ~ ., data = ames_train) %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric()) %>% 
  step_pca(all_numeric(), threshold = 0.95)

#putting the processes together 
blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>% #upply the formula of interest (the target variable, features, and the data these are based on) 
  step_nzv(all_nominal())  %>% #Remove near-zero variance features that are categorical (aka nominal).
  step_integer(matches("Qual|Cond|QC|Qu")) %>% #Ordinal encode our quality-based features (which are inherently ordinal).
  step_center(all_numeric(), -all_outcomes()) %>% #Center and scale (i.e., standardize) all numeric features.
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_pca(all_numeric(), -all_outcomes()) #Perform dimension reduction by applying PCA to all numeric features.

#train the bluepront on some data. but remember we don't want to train the data on some of these features 

prepare <- prep(blueprint, training = ames_train)

#Lastly, we can apply our blueprint to new data (e.g., the training data or future test data) with bake().
baked_train <- bake(prepare, new_data = ames_train) #make our training dataset 
baked_test <- bake(prepare, new_data = ames_test) #make testing ataset 
baked_train

#First, we create our feature engineering blueprint to perform the following tasks:

blueprint <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_nzv(all_nominal()) %>%#Filter out near-zero variance features for categorical features.
  step_integer(matches("Qual|Cond|QC|Qu")) %>% #Ordinally encode all quality features, which are on a 1–10 Likert scale.
  step_center(all_numeric(), -all_outcomes()) %>% #Standardize (center and scale) all numeric features.
  step_scale(all_numeric(), -all_outcomes()) %>% #One-hot encode our remaining categorical features.
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

# Specify resampling plan 
cv <- trainControl(
  method = "repeatedcv", 
  number = 10, 
  repeats = 5
)

hyper_grid <- expand.grid(k = seq(2, 25, by = 1)) # Construct grid of hyperparameter values

knn_fit2 <- train(
  blueprint, 
  data = ames_train, 
  method = "knn", 
  trControl = cv, 
  tuneGrid = hyper_grid,
  metric = "RMSE"
)

