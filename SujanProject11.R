##1. Loading Required Libraries 
# Loading required libraries
library(tidyverse)          # Pipe operator (%>%) and other commands
library(caret)              # Random split of data/cross validation
library(olsrr)              # Heteroscedasticity Testing (ols_test_score)
library(car)                # Muticolinearity detection (vif)
library(broom)              # Diagnostic Metric Table (augment)

##2. Loading Data set
# Loading Data set
library(readr)

data=read_csv("C:/Users/sujan/Downloads/advertising.csv")
View(data)
##3. Exploring Data set
# Inspection of top 5-rows of data
head(data)

# Inspection of bottom 5-rows of data
tail(data)

# Getting Structure of whole data set
str(data)
# Checking Outliers
boxplot(data)
##The above plot shows that two outliers are present in the variable "Newspaper".
# Removing Outliers
data <- data[-which(data$Newspaper %in% boxplot.stats(data$Newspaper)$out), ]

# Again Checking Outliers
boxplot(data)

# Checking Missing Values
table(is.na(data))

# Creating scatter plot matrix 
pairs(data , upper.panel = NULL)

# Scatter Plot between TV and Sales
plot(data$TV , data$Sales)

# Scatter Plot between Radio and Sales
plot(data$Radio , data$Sales)

# Scatter Plot between Newspaper and Sales
plot(data$Newspaper , data$Sales)

# Scatter Plot between TV and Radio
plot(data$TV , data$Radio)

# Scatter Plot between Newspaper and TV
plot(data$TV , data$Newspaper)

# Scatter Plot between Newspaper and Radio
plot(data$Radio , data$Newspaper)
##4. Splitting Data Set
# Randomly Split the data into training and test set
set.seed(123)
training.samples <- data$Sales %>%
  createDataPartition(p = 0.75, list = FALSE)
train.data  <- data[training.samples, ]
test.data <- data[-training.samples, ]

##5. Fitting Simple Linear Regression
# Fitting Sales ~ TV
sm1 <- lm(Sales ~ TV , data = train.data)

# Take a look on summary of the model
summary(sm1)

# Fitting Sales ~ Radio
sm2 <- lm(Sales ~ Radio , data = train.data)

# Take a look on summary of the model
summary(sm2)

# Fitting Sales ~ Newspaper
sm3 <- lm(Sales ~ Newspaper , data = train.data)

# Take a look on summary of the model
summary(sm3)

# Scatter plot with Simple Linear Regression Line
plot(train.data$TV , train.data$Sales)

# Adding Regression Line
abline(lm(train.data$Sales ~ train.data$TV) , col = "blue")

##6. Fitting Multiple Linear Regression with Diagnostic Plot

# Fitting MLR model with predictors TV and Radio 
mm1 <- lm(Sales ~ TV + Radio , data = train.data)

# Take a look on summary of the model
summary(mm1)

# Performing ANOVA to test the above stated null hypothesis
anova(sm1 , mm1)
# Extending further the MLR including the predictor Newspaper
mm2 <- lm(Sales ~ TV + Radio + Newspaper , data = train.data)

# Take a look on summary of the model
summary(mm2)

# Residual Plot
plot(mm1 , 1)

# Score Test for Heteroscedasticity
ols_test_score(mm1)

# Checking effect of Auto-correlation
durbinWatsonTest(mm1)
# Detecting Multicolinearity
vif(mm1)
# Checking Normality of Errors
shapiro.test(mm1$residuals)
##Normality does not hold since p-value < 0.05
# Plotting Histogram for Residuals
hist(mm1$residuals)

##7. Fitting Orthogonal Polynomial Regression with Diagnostic Plot
# Fitting second order orthogonal polynomial model in two variables to avoid multicolinearity
pm1 <- lm(Sales ~ poly(TV , 2) + poly(Radio , 2) + TV:Radio  , data = train.data)

# Take a look on summary of the model
summary(pm1)

# Performing ANOVA to test the above stated null hypothesis
anova(mm1 , pm1)

# Fitting third order (orthogonal) polynomial model in two variables to avoid multicolinearity
pm2 <- lm(Sales ~ poly(TV , 3) + poly(Radio , 3) + TV:Radio  , data = train.data)

# Take a look on summary of the model
summary(pm2)

# Fitting third order (orthogonal) polynomial model in two variables to avoid multicolinearity but after removing third order of TV predictor
pm3 <- lm(Sales ~ poly(TV , 2) + poly(Radio , 3) + TV:Radio  , data = train.data)

# Take a look on summary of the model
summary(pm3)

# Performing ANOVA to test the above stated null hypothesis
anova(pm1 , pm3)
##Diagnostic Plots -
#Now, again check all the assumptions of Linear Regression are satisfied or not.
#Checking Linearity Assumption -

# Residual Plot
plot(pm3 , 1)

# Score Test for Heteroscedasticity
ols_test_score(pm3)

# Checking effect of Auto-correlation
durbinWatsonTest(pm3)
# Checking Normality of Errors
shapiro.test(pm3$residuals)
# Detecting Multicolinearity
vif(pm3)
# Creating Diagnostic metrics Table for model pm3
dm = augment(pm3)

# See the Table
head(dm)
# Checking minimum value of last column (Studentized Residuals)
min(dm$.std.resid)

# Checking the index of that observation in train data
which(dm$.std.resid  %in% "-3.654809")
# Info. about 98th row of train data set
train.data[98,]
# Removing 98th row of outlier
train.data1 = train.data %>% filter(train.data$Sales !=  1.6)

# Checking number of rows in old train data set
nrow(train.data)

# Checking number of rows in new train data set (train.data1)
nrow(train.data1)

# Fitting third order orthogonal polynomial model in two variables TV and Radio but after removing third order of TV predictor using train.data1
pm4 <- lm(Sales ~ poly(TV , 2) + poly(Radio , 3) + TV:Radio  , data = train.data1)

# Take a look on summary of the model
summary(pm4)
# Linearity Assumption
plot(pm4 ,1)

# Homoscedasticity Assumption 
ols_test_score(pm4)

# Autocorrelation Assumption 
durbinWatsonTest(pm4)

# Normality Assumption
shapiro.test(pm4$residuals)

# Multicolinearity Assumption
vif(pm4)
# Creating Diagnostic metric table for model pm4
dm1 = augment(pm4)

# Checking minimum and maximum value of Studentized Residuals
min(dm1$.std.resid)
max(dm1$.std.resid)

# Making Predictions
prediction = pm4 %>% predict(test.data)

# Checking performance by calculating R2 , RMSE and MAE
data.frame( R2 = R2(prediction, test.data$Sales),
            RMSE = RMSE(prediction, test.data$Sales),
            MAE = MAE(prediction, test.data$Sales))
##9. Repeated 10-fold Cross Validation
# Removing outlier, i.e., the row that contains Sales = 1.6
data <- data %>% filter(Sales != 1.6)
# Define training control
set.seed(123)
train.control <- trainControl(method = "repeatedcv", 
                              number = 10, repeats = 3)
# Train the model
model_cv <- train(Sales ~ poly(TV , 2) + poly(Radio , 3) + TV:Radio , data = data, method="lm",
                  trControl = train.control)

# Summarize the results
print(model_cv)
