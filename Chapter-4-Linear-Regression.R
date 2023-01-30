#Chapter 4 - Linear Regression 

library(pacman)
pacman::p_load(tidyverse, caret, vip, forecast,pdp,AmesHousing, install = T)

#run a simple model 

model1 <- lm(Sale_Price ~ Gr_Liv_Area, data = ames_train)
summary(model1)

plot(model1)


#getting the RMSE 
sigma(model1)

#Note that the RMSE is also reported as the Residual standard error in the output from summary().

#getting the MSE 
sigma(model1)^2

#get confidence intervals
confint(model1)

#progressing to multiple regression 
(model2 <- lm(Sale_Price ~ Gr_Liv_Area + Year_Built, data = ames_train))
summary(model2)


#we can use update() to update the model formula 

# The new formula can use a . as shorthand for keep everything on either the left or right hand side of the formula, and a + or - can be used to add or remove terms 

#update the model with everything in the dataframe 
(model2 <- update(model1, . ~ . + Year_Built))

#specifying interaction effects
lm(Sale_Price ~ Gr_Liv_Area + Year_Built + Gr_Liv_Area:Year_Built, data = ames_train)

#short hand 
lm(Sale_Price ~ Gr_Liv_Area*Year_Built, data = ames_train)


#include all possible effects
model3 <- lm(Sale_Price ~ ., data = ames_train)

# print estimated coefficients in a tidy data frame
broom::tidy(model3)  

#Assessing model accuracy 

#Use RMSE to assess model accuarcy 

#use caret to assess fit 
set.seed(123) #set for reproducability 

(cv_model1 <- train(
  form = Sale_Price ~ Gr_Liv_Area,
  data = ames_train,
  method = "lm",
  trControl = trainControl(method = 'cv', number = 10)
))

#gives us the average RMSE across the 10 models 

#we can now perform CV on the other  models we specified 
# model 2 CV
set.seed(123)
cv_model2 <- train(
  Sale_Price ~ Gr_Liv_Area + Year_Built, 
  data = ames_train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

# model 3 CV
set.seed(123)
cv_model3 <- train(
  Sale_Price ~ ., 
  data = ames_train, 
  method = "lm",
  trControl = trainControl(method = "cv", number = 10)
)

#Extract out the performance measures 
summary(resamples(list(
  model1 = cv_model1,
  model2 = cv_model2,
  model3 = cv_model3
)))

#model 3 has the best fit 
#more info -> more predicitors 

#issues with linear regression 

#broom::augment function is an easy way to add model results to each observation

p1 <- ggplot(ames_train, aes(Year_Built, Sale_Price)) + 
  geom_point(size = 1, alpha = .4) +
  geom_smooth(se = FALSE) +
  scale_y_continuous("Sale price", labels = scales::dollar) +
  xlab("Year built") +
  ggtitle(paste("Non-transformed variables with a\n",
                "non-linear relationship."))

p2 <- ggplot(ames_train, aes(Year_Built, Sale_Price)) + 
  geom_point(size = 1, alpha = .4) + 
  geom_smooth(method = "lm", se = FALSE) +
  scale_y_log10("Sale price", labels = scales::dollar, 
                breaks = seq(0, 400000, by = 100000)) +
  xlab("Year built") +
  ggtitle(paste("Transforming variables can provide a\n",
                "near-linear relationship."))

gridExtra::grid.arrange(p1, p2, nrow = 1)

#no autocorrelation of the errors 
df1 <- mutate(df1, id = row_number())
df2 <- mutate(df2, id = row_number())

p1 <- ggplot(df1, aes(id, .resid)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Correlated residuals.")

p2 <- ggplot(df2, aes(id, .resid)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Row ID") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Uncorrelated residuals.")

gridExtra::grid.arrange(p1, p2, nrow = 1)


#also, constant variance among residuals (homoscedasticity)
df1 <- broom::augment(cv_model1$finalModel, data = ames_train)

p1 <- ggplot(df1, aes(.fitted, .resid)) + 
  geom_point(size = 1, alpha = .4) +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 1", subtitle = "Sale_Price ~ Gr_Liv_Area")

df2 <- broom::augment(cv_model3$finalModel, data = ames_train)

p2 <- ggplot(df2, aes(.fitted, .resid)) + 
  geom_point(size = 1, alpha = .4)  +
  xlab("Predicted values") +
  ylab("Residuals") +
  ggtitle("Model 3", subtitle = "Sale_Price ~ .")


#PCA regression 

#reducing the number of factors 

#Performing PCR with caret is an easy extension from our previous model. We simply specify method = "pcr" within train()


# perform 10-fold cross validation on a PCR model tuning the 
# number of principal components to use as predictors from 1-100
set.seed(123)
cv_model_pcr <- train(
  Sale_Price ~ ., 
  data = ames_train, 
  method = "pcr", #modify to use PCA 
  trControl = trainControl(method = "cv", number = 10), #setting our CV method 
  preProcess = c("zv", "center", "scale"), #setting preprocess stuff, center scale and ignore zero variance predictors 
  tuneLength = 100 #tune length, using 100 principal components 
)

# model with lowest RMSE
cv_model_pcr$bestTune


# results for model with lowest RMSE
cv_model_pcr$results %>%
  dplyr::filter(ncomp == pull(cv_model_pcr$bestTune))

#can see that it takes ~100 components to reach lowest RMSE
ggplot(cv_model_pcr)

#By controlling for multicollinearity with PCR, we can experience significant improvement in our predictive accuracy 
#compared to the previously obtained linear models

#Partial Least Squares
#Partial least squares (PLS) can be viewed as a supervised dimension reduction procedure (Kuhn and Johnson 2013)
#also uses the response variable to aid the construction of the principal components

#supervised dimension reduction procedure that finds new features that not only captures most of the information in the original features, but also are related to the response.

# perform 10-fold cross validation on a PLS model tuning the 
# number of principal components to use as predictors from 1-30


#set our cv stuff for all the models using an object 
cv <- trainControl(method = "cv", number = 10)

set.seed(123)
cv_model_pls <- train(
  Sale_Price ~., 
  data = ames_train,
  method = "pls", #change to appropriate estimation method 
  trControl = cv,
  preProcess = c("zv", "center", "scale"),
  tuneLength = 30
)

#model with lowest RMSE
cv_model_pls$bestTune


# results for model with lowest RMSE
cv_model_pls$results %>%
  dplyr::filter(ncomp == pull(cv_model_pls$bestTune))

#plot cross-validated RMSE
ggplot(cv_model_pls)

#can see that 3 is the ideal # of parameters

#Feature interpretation 

#need to get the most inmportant features out 

vip::vip(cv_model_pls, num_features = 20, method = 'model')

#The importance measure is normalized from 100 (most important) to 0 (least important)

#construct partial dependency plots to viz the change in the average predicted value 
pdp::partial(cv_model_pls, "Gr_Liv_Area", grid.resolution = 20, plot = TRUE) #can prune to only display a certain amount 
pdp::partial(cv_model_pls, "First_Flr_SF", grid.resolution = 20, plot = TRUE)
pdp::partial(cv_model_pls, "Garage_Cars", grid.resolution = 20, plot = TRUE)
pdp::partial(cv_model_pls, "Garage_Area", grid.resolution = 20, plot = TRUE)

