#Chapter 5 - Logistic Regression 

library(pacman)
pacman::p_load(tidyverse, rsample, caret, vip, modeldata, install = T)

#prep the data
df <- attrition %>% mutate_if(is.ordered, factor, ordered = FALSE) #need to install modeldata package to get the attrition dataset


# Create training (70%) and test (30%) sets for the 
set.seed(123)
churn_split <- initial_split(df, prop = 0.7, strata = "Attrition") #split the data, stratafying on our outcome variable "Attrition"
churn_train <- training(churn_split) #designating our training dataset
churn_test <- testing(churn_split) #designation our testing dataset

#simple logistic regression 

model1 <- glm(Attrition ~ MonthlyIncome, family = "binomial", data = churn_train)
model2 <- glm(Attrition ~ OverTime, family = "binomial", data = churn_train)
tidy(model1)
tidy(model2)

#sometimes it's easier to interpret coefficient when exponentiating them 
exp(coef(model1))

#we can also fit our model to binary responses 
model3 <- glm(
  Attrition ~ MonthlyIncome + OverTime,
  family = "binomial", 
  data = churn_train
)

tidy(model3)


#assess model accuracy 

#specify the three models 
set.seed(123)
cv_model1 <- train(
  Attrition ~ MonthlyIncome, 
  data = churn_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model2 <- train(
  Attrition ~ MonthlyIncome + OverTime, 
  data = churn_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

set.seed(123)
cv_model3 <- train(
  Attrition ~ ., 
  data = churn_train, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10)
)

#now extract their performance
summary(
  resamples(
    list(
      model1 = cv_model1,
      model2 = cv_model2,
      model3 = cv_model3
    )
  )
)

#model 3 has the best accuracy of the three, which makes sense 

#now use the confusion matrix to actually see the accuracy 

#predict class
pred_class <- predict(cv_model3, churn_train)

#confusion matrix 
confusionMatrix(
  data = relevel(pred_class, ref = "Yes"),
  reference = relevel(churn_train$Attrition, ref = "Yes")
)

#plotting simple model to full model 


library(ROCR)

# Compute predicted probabilities
m1_prob <- predict(cv_model1, churn_train, type = "prob")$Yes # we want to predict "Yes" for each model
m3_prob <- predict(cv_model3, churn_train, type = "prob")$Yes

# Compute AUC metrics for cv_model1 and cv_model3
perf1 <- prediction(m1_prob, churn_train$Attrition) %>%
  performance(measure = "tpr", x.measure = "fpr")
perf2 <- prediction(m3_prob, churn_train$Attrition) %>%
  performance(measure = "tpr", x.measure = "fpr")

# Plot ROC curves for cv_model1 and cv_model3
plot(perf1, col = "black", lty = 2)
plot(perf2, add = TRUE, col = "blue")
legend(0.8, 0.2, legend = c("cv_model1", "cv_model3"),
       col = c("black", "blue"), lty = 2:1, cex = 0.6)

# Perform 10-fold CV on a PLS model tuning the number of PCs to 
# use as predictors
set.seed(123)
cv_model_pls <- train(
  Attrition ~ ., 
  data = churn_train, 
  method = "pls", #partial least squares 
  family = "binomial", #binonmial distributions
  trControl = trainControl(method = "cv", number = 10), #CV method
  preProcess = c("zv", "center", "scale"), #center, scale, and cut zero variance predictors
  tuneLength = 16 #hunt for 16 predictors 
)

#get the model with the optimal number of parameters 
cv_model_pls$bestTune

#6 appears the be the best 

# results for model with lowest loss
cv_model_pls$results %>%
  dplyr::filter(ncomp == pull(cv_model_pls$bestTune))

#model with the lowest loss is also 11

#visualize 
ggplot(cv_model_pls)

#also see that the number of components is 11 based on the graph 

#Now need to see which features are especially meaningful 

#Using vip::vip() we can extract our top 20 influential variables. 

vip::vip(cv_model_pls, num_features = 20)

vip::vip(cv_model3, num_features = 20) # looking at the most meaningful features 
