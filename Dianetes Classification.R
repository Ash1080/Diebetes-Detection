#' ---
#' title: "Diabetes Classification"
#' subtitles: "Regression Analysis HW2"
#' author: "Sharif Ashna - 99103633"
#' date: "2023 May"
#' output: html_document
#' ---

#' # Introduction
#' Nearly half a billion people worldwide live with diabetes, and nearly 80% of those live in low- and middle-income countries. Nine in ten people with diabetes have Type 2 diabetes, which is increasing fastest in low- and middle-income countries.
#' Diabetes increases the risk of early death, and diabetes-related complications can lower quality of life. The high global burden of diabetes has a negative economic impact on individuals, health care systems, and nations.
#' 
#' Complications like heart disease, vision loss, lower-limb amputation, and kidney disease are associated with chronically high levels of sugar remaining in the bloodstream for those with diabetes.
#' While there is no cure for diabetes, strategies like losing weight, eating healthily, being active, and receiving medical treatments can mitigate the harms of this disease in many patients.
#' Early diagnosis can lead to lifestyle changes and more effective treatment, making predictive models for diabetes risk important tools for public and public health officials.
#' The scale of this problem is also important to recognize. The Centers for Disease Control and Prevention has indicated that as of 2018, 34.2 million Americans have diabetes and 88 million have prediabetes. 
#' Furthermore, the CDC estimates that 1 in 5 diabetics, and roughly 8 in 10 pre diabetics are unaware of their risk.
#' While there are different types of diabetes, type II diabetes is the most common form and its prevalence varies by age, education, income, location, race, and other social determinants of health.
#' Much of the burden of the disease falls on those of lower socioeconomic status as well.
#' Diabetes also places a massive burden on the economy, with diagnosed diabetes costs of roughly $327 billion dollars and total costs with undiagnosed diabetes and prediabetes approaching $400 billion dollars annually.
#' So, it’s important to identify and detect the people who might have diabetes or are in risk of having one. To do so, we gather information on the patients and probable cases.
#' We consider 21 factors that may relate to having diabetes. Here we have records of 253680 of patients who lives in America based on the survey conducted by CDC – Center of Disease Control and Prevention – annually. 
#' Another dataset we use here is a clean dataset of 70,692 survey responses which has an equal 50-50 split of respondents with no diabetes and with either prediabetes or diabetes. Both of the dataset have binary responses.
#' 
#' 
#' # Problems
#'
#' 1. The first problem is "is this worth trying"? does any of the features have any connection with diabetes?
#'
#' 2. The second problem that we need to address here is what are the most important factors for detecting diabetes?
#'
#' 3. The Third problem is if we can use a subset of features instead of using all of them?
#'
#' 4. the fourth problem is how we actually model and what features we use for prediction?
#'
#' 
#' before we start, we load the datasets and required libraries on to our code
#' 

# required libraries

library(rmarkdown)
library(data.table)
library(grid)
library(caret)
library(tidyverse)
library(MASS)
library(ISLR2)
library(pls)
library(FSelector)
library(rpart)
library(rpart.plot)
library(glmnet)

#loading datasets

path = choose.files()
d = read.csv(path)
path2 = choose.files()
d2 = read.csv(path2)

#' Before starting, let's take a glance at datasets in hand. as you can see, all the predictors except 'BMI' are qualitative.
#' for further information about the features you can visit the source page [here.](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
#' 
head(d, 5)
head(d2, 5)

#' # First Problem
#' To address the first problem, we use basic graphs and plots to show a connection between response and some of the features
#' 
#' Based on these simple plots we can see that Diabetes at least is connected with some of features here.
#' 

ggplot(d, aes(x = BMI, fill = as.factor(Diabetes_binary))) + geom_density(alpha = 0.75)
e = ggplot(d, aes(x = Age, fill = as.factor(Diabetes_binary))) + geom_density(alpha = 0.7)
e + scale_fill_brewer(palette = 'Blues')
ggplot(d, aes(x = Income, fill = as.factor(Diabetes_binary))) + geom_boxplot(alpha = 0.75)+
  facet_grid(Diabetes_binary ~ .) + labs(x = 'Income')
ggplot(d, aes(x = as.factor(HighChol), y = as.factor(Diabetes_binary))) + geom_count(color = 'darkred') + 
  theme_dark() + labs(x = 'High Cholestrol', y = 'Diabetes', title = 'Number of Cases') + scale_radius(range = c(2,20))

#'  Correlation Matrix of Features
a = data.frame(cor(d))  
a[1:5,1:5]


#' # Second Problem
#' The easiest way to find that which of the features is important is to use the Logistic Regression coefficients of each feature

glm.fit = glm(Diabetes_binary ~ HighBP + HighChol + CholCheck + BMI + Smoker + Stroke + HeartDiseaseorAttack + 
                PhysActivity + Fruits + Veggies + HvyAlcoholConsump + AnyHealthcare + NoDocbcCost + GenHlth + MentHlth + 
                PhysHlth + DiffWalk + Sex + Age + Education + Income,family = binomial, data = d)

# another method for full model

glm.fit = glm(d$Diabetes_binary ~ ., family = binomial, data = d)
summary(glm.fit)

#' here we can see which features are not actually related to the response
#'
q = data.frame(summary(glm.fit)$coef)
q$Pr...z.. = round(q$Pr...z.., 9)
head(q, 10)
q[c(which(q$Pr...z.. > 0.05)),]




#after discarding features
glm.fit2 = glm(Diabetes_binary ~ HighBP + HighChol + CholCheck + BMI + Stroke + HeartDiseaseorAttack + 
                 Veggies + HvyAlcoholConsump + GenHlth + MentHlth + 
                 PhysHlth + DiffWalk + Sex + Age + Education + Income,family = binomial, data = d)

summary(glm.fit2)
q2 = data.frame(summary(glm.fit2)$coef)
q2$Pr...z.. = round(q2$Pr...z.., 9)
head(q2, 10)
q2[c(which(q2$Pr...z.. > 0.05)),]

#' highest estimate
q2[which.max(abs(q$Estimate[-1]))+1,]

#' # Third Problem

#' another method for feature importance is to use Chi square test and information gain criteria

# Fselector for feature importance
#Chi_square 
chi = chi.squared(Diabetes_binary ~ ., data = d)
features = cbind(chi,names(d)[-1])
features = features[order(features$attr_importance, decreasing = TRUE),]
features = data.table(features)
names(features) = c('Importance', 'attribute')
head(features,5)
print('based on Chi-square')

#information gain
IG = information.gain(Diabetes_binary ~ ., data = d)
features2 = cbind(IG,names(d)[-1])
features2 = features2[order(features2$attr_importance, decreasing = TRUE),]
features2 = data.table(features2)
names(features2) = c('Importance', 'attribute')
head(features2,5)
print('based on IG')


#' Shrinkage is a method for estimating coefficients and reducing variance at the same time
#' 
#' here we use 'The Lasso' for such a thing.
# lasso coef
x = model.matrix(Diabetes_binary ~., data = d, family = binomial)
y = d$Diabetes_binary
grid = 10^seq(10,-3, length = 100)
lasso.mod = glmnet(x,y,alpha = 1, lambda = grid,family = 'binomial', standardize = FALSE)
plot(lasso.mod, lwd = 2)

#' In the mean time we need to determine the best tuning parameter for lasso.
#' To do so, we use cross validation method.
cv.out = cv.glmnet(x,y, family = 'binomial', alpha = 1)
plot(cv.out)
#' Here, you can see the estimation of coefficients.
bestlam = cv.out$lambda.min
predict(lasso.mod, s = bestlam, x, type = 'coefficients')

#' # Fourth Problem
#' 
#' We use 4 different method to predict.
#' for training the model we use the 'd' as the training set. It is a clear and balance set so our model would be more stable
#' . then we test the model on 'd2' which has approximately 270000 records. 
#' 
#' before we start, we need to determine the 'Threshold' which we use to address the diabetes.

glm.prob2 = predict(glm.fit2,d, type = 'response')
cv.error = data.frame(matrix(data = NA, nrow = 9, ncol = 4))



for (i in 1:9){
  res = rep(0, nrow(d))
  res[glm.prob2 > i/10] = 1
  con_fit = confusionMatrix(reference = factor(d$Diabetes_binary), data = factor(res))
  cv.error[i, 2] = as.numeric(con_fit$overall[1])
  cv.error[i, 3] = con_fit$byClass[1]
  cv.error[i, 4] = con_fit$byClass[2]
  
  
  
  
}  

names(cv.error) = c('Threshold', 'Accuracy', 'Specifity', 'Sensitivity')
cv.error$Threshold = 1:9/10
cv.error


#' based on the table we use 0.45 as threshold.
#' 
#' ## Prediction with Logistic Regression
#####
# logistic
glm.prob = predict(glm.fit2,newdata = d2, type = 'response')
glm.prob[1:10]
w = data.table(matrix(data = 0,nrow = nrow(d2), ncol = 2))
w$V1 = d2$Diabetes_binary
w$V2[glm.prob > 0.45] = 1
con_logistic = confusionMatrix(data = factor(w$V2),reference = factor(w$V1))
con_logistic$table
con_logistic$overall[1]
con_logistic$byClass[1:2]




#' ##Prediction with Partial Least Square
########
# pls

set.seed(1)
pls.fit = plsr(Diabetes_binary ~ ., data = d, scale = FALSE, validation = 'CV', type = 'binomial')
summary(pls.fit)
validationplot(pls.fit, val.type = 'R2')
pls.pred = predict(pls.fit, newdata = d2, ncomp = 10)
head(pls.pred,5)
w$V3 = 0
w$V3[pls.pred > 0.45] = 1
con_pls = confusionMatrix(reference = factor(w$V1), data = factor(w$V3))
con_pls

#' ## Prediction with Lasso
##########
#lasso
x2 = model.matrix(Diabetes_binary ~., data = d2, family = binomial)
lasso.fit = predict(lasso.mod, newx = x2, s = bestlam, type = 'response')
w$V4 = 0
w$V4[lasso.fit > 0.45] = 1
con_lasso = confusionMatrix(reference = factor(w$V1), data = factor(w$V4))
con_lasso
names(w) = c('y', 'logistic', 'pls', 'lasso')


#' ## Prediction with Regression Tree
#######
# tree

tree.fit = rpart(Diabetes_binary ~ ., method = 'class', data = d, control = rpart.control(maxdepth = 12))
rpart.plot(tree.fit, type = 1, extra = 2, cex = 0.7)
tree.pred = predict(tree.fit, newdata = d2, type = 'class')
w$tree = tree.pred
con_tree = confusionMatrix(reference = factor(w$y), data = factor(w$tree))
con_tree


#' ## Comparison of Methods

ACC = data.frame(matrix(NA, nrow = 4, ncol = 4))
names(ACC) = c('Method', 'Accuracy', 'Sensitivity', 'Specifity')


ACC[1,] = c(Method = 'Logistic', Accuracy = round(con_logistic$overall[1],4), Sensitivity = round(con_logistic$byClass[1],4),
            Specifity = round(con_logistic$byClass[2],4))
ACC[2,] = c(Method = 'PLS', Accuracy = round(con_pls$overall[1],4), Sensitivity = round(con_pls$byClass[1],4),
            Specifity = round(con_pls$byClass[2],4))
ACC[3,] = c(Method = 'Lasso', Accuracy = round(con_lasso$overall[1],4), Sensitivity = round(con_lasso$byClass[1],4),
            Specifity = round(con_lasso$byClass[2],4))
ACC[4,] = c(Method = 'Tree', Accuracy = round(con_tree$overall[1],4), Sensitivity = round(con_tree$byClass[1],4),
            Specifity = round(con_tree$byClass[2],4))
ACC










