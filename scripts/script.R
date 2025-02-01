rm(list=ls())

# Useful libraries
library(tidyverse)
library(magrittr)
library(corrplot)
library(modelsummary)
library(dplyr)
library(margins)
library(ROCR)
library(MASS)
library(caret)
library(car)
library(e1071)
library(class)


# Setting working directory
setwd("")
getwd()

# Loading the data
data <- read_csv("faults.csv")

# Data cleaning
clean_data <- data %>%
  mutate(
    spt_pa = Steel_Plate_Thickness / Pixels_Areas,
    range_of_x = abs(X_Maximum - X_Minimum),
    range_of_y = abs(Y_Maximum - Y_Minimum),
    area_perimeter_ratio = Pixels_Areas / (X_Perimeter + Y_Perimeter),
    luminosity_range = abs(Maximum_of_Luminosity - Minimum_of_Luminosity),
    luminosity_ratio =  abs(Minimum_of_Luminosity / Maximum_of_Luminosity),
    fault_type = Bumps + Other_Faults,  
    steel_type = TypeOfSteel_A300)

attach(clean_data)
clean_data <- subset(clean_data, select = -c(X_Minimum,X_Maximum,Y_Minimum,Y_Maximum,X_Perimeter,Y_Perimeter,Sum_of_Luminosity,Maximum_of_Luminosity,Minimum_of_Luminosity,
                                             Length_of_Conveyer,Edges_Index,Empty_Index,Square_Index,Outside_X_Index,Edges_X_Index,Edges_Y_Index,
                                             Outside_Global_Index,LogOfAreas,Log_X_Index,Log_Y_Index,Orientation_Index,Luminosity_Index,SigmoidOfAreas,
                                             TypeOfSteel_A300,TypeOfSteel_A400,Pastry,Z_Scratch,K_Scatch,Stains,Dirtiness,Bumps,Other_Faults))

# Changing notation
options(scipen = 999)
# Training and test sets
n <- nrow(clean_data)
set.seed(2024)
train.ind <- sample(1:n, size = 0.75*n)
train <- clean_data[train.ind,]
test <- clean_data[-train.ind,]

##### Logistic Regression 1
Logi1 <- glm(fault_type ~ luminosity_ratio + steel_type, data = train, family = binomial)
summary(Logi1)
vif(Logi1)

#### Logistic Regression 2
Logi2 <- glm(fault_type ~ Pixels_Areas + Steel_Plate_Thickness + 
               area_perimeter_ratio + range_of_y 
             + range_of_x + steel_type
             , data = train, family = binomial)
summary(Logi2)
step(Logi2)
Logi2Step <- glm(formula = fault_type ~ Pixels_Areas + Steel_Plate_Thickness + 
                   range_of_x + steel_type, family = binomial, data = train)
summary(Logi2Step)
vif(Logi2Step)

# Cohen's pseudo-R2
1 - summary(Logi1)$deviance/summary(Logi1)$null.deviance
1 - summary(Logi2Step)$deviance/summary(Logi2Step)$null.deviance

#### Confusion Matrix: Logistic Regressions
# Logi1
Logi1.probs <- predict(Logi1, test, type="response")
Logi1.pred <- rep("0", nrow(test))
Logi1.pred[Logi1.probs > 0.5] = "1"

table(test$fault_type,Logi1.pred)
addmargins(table(test$fault_type,Logi1.pred))
confusionMatrix(data=as.factor(Logi1.pred),reference=as.factor(test$fault_type),positive="1") 

predLogi1 <- prediction(Logi1.probs,as.factor(test$fault_type))
perfLogi1 <- performance(predLogi1,"tpr","fpr")


AUC1 <- performance(predLogi1,"auc")@y.values[[1]]
AUC1


# Logi2Step
Logi2.probs <- predict(Logi2Step, test, type="response")
Logi2.pred <- rep("0", nrow(test))
Logi2.pred[Logi2.probs > 0.5] = "1"

table(test$fault_type,Logi2.pred)
addmargins(table(test$fault_type,Logi2.pred))
confusionMatrix(data=as.factor(Logi2.pred),reference=as.factor(test$fault_type),positive="1") #provare a cambiare threshold commenti: k un po' basso, spcifit? un po' bassa

predLogi2Step <- prediction(Logi2.probs,as.factor(test$fault_type))
perfLogi2Step <- performance(predLogi2Step,"tpr","fpr")

AUC2 <- performance(predLogi2Step,"auc")@y.values[[1]]
AUC2

#####LDA 1
lda.fit1 <- lda(fault_type ~ luminosity_ratio + steel_type, data = train)
lda.fit1

lda.pred.test1 <- predict(lda.fit1, newdata = test)
lda.class.test1 <- lda.pred.test1$class
table(test$fault_type,lda.class.test1)
addmargins(table(test$fault_type,lda.class.test1))
confusionMatrix(data=as.factor(lda.class.test1),reference=as.factor(test$fault_type),positive="1")
# Changing the classification threshold 
# Predictions on test data
lda.post1 <- lda.pred.test1$posterior
lpred.class.02.1 <- ifelse(lda.post1[,2]>0.55,"1", "0")
confusionMatrix(data=as.factor(lpred.class.02.1),reference=as.factor(test$fault_type),positive="1")

lda.pred.test1 <- predict(lda.fit1, newdata = test)
lda.post1 <- lda.pred.test1$posterior
pred1 <- prediction(lda.post1[,2],as.factor(test$fault_type))
perfLDA1 <- performance(pred1,"tpr","fpr")
AUC3.1 <- performance(pred1,"auc")@y.values[[1]]  #AUC

#####LDA2
lda.fit2 <- lda(fault_type ~ Pixels_Areas + Steel_Plate_Thickness + 
                  range_of_x + steel_type, data = train)
lda.fit2

lda.pred.test2 <- predict(lda.fit2, newdata = test)
lda.class.test2 <- lda.pred.test2$class
table(test$fault_type,lda.class.test2)
addmargins(table(test$fault_type,lda.class.test2))
confusionMatrix(data=as.factor(lda.class.test2),reference=as.factor(test$fault_type),positive="1")
# Changing the classification threshold 
# Predictions on test data
lda.post2 <- lda.pred.test2$posterior
lpred.class.02.2 <- ifelse(lda.post2[,2]>0.5,"1", "0")
confusionMatrix(data=as.factor(lpred.class.02.2),reference=as.factor(test$fault_type),positive="1")

lda.pred.test2 <- predict(lda.fit2, newdata = test)
lda.post2 <- lda.pred.test2$posterior
pred2 <- prediction(lda.post2[,2],as.factor(test$fault_type))
perfLDA2 <- performance(pred2,"tpr","fpr")
AUC3.2 <- performance(pred2,"auc")@y.values[[1]]  #AUC
AUC3.2
# We compared the 2 LDAs and we chose to include the number 2 in the poster



#####QDA2
# We directly did the QDA with all 4 features

qda.fit2 <- qda(fault_type ~ Pixels_Areas + Steel_Plate_Thickness + 
                  range_of_x + steel_type, data = train)

# Predictions
qda.pred.test2 <- predict(qda.fit2, newdata = test)
qda.class.test2 <- qda.pred.test2$class

# Confusion matrix
table(test$fault_type,qda.class.test2)
addmargins(table(test$fault_type,qda.class.test2))
confusionMatrix(data=as.factor(qda.class.test2),reference=as.factor(test$fault_type),positive="1")

# Changing the classification threshold 
# Predictions on test data
qda.post2 <- qda.pred.test2$posterior
qpred.class.02.2 <- ifelse(qda.post2[,2]>0.55,"1", "0")
confusionMatrix(data=as.factor(qpred.class.02.2),reference=as.factor(test$fault_type),positive="1")

qda.pred.test2 <- predict(qda.fit2, newdata = test)
qda.post2 <- qda.pred.test2$posterior
qpred2 <- prediction(qda.post2[,2],as.factor(test$fault_type))
perfQDA2 <- performance(qpred2,"tpr","fpr")
AUC4.2 <- performance(qpred2,"auc")@y.values[[1]]  #AUC
AUC4.2



##### Naive bayes 2 

nb.fit2 <- naiveBayes(fault_type ~ Pixels_Areas + Steel_Plate_Thickness + 
                        range_of_x + steel_type, data = train)
nb.fit2

nb.post2 <- predict(nb.fit2, newdata=test, type="raw")
# Predict classes
nb.class.test2 <- ifelse(nb.post2[,2]>0.6,"1","0") # We got better results with 0.6 classification threshold
# Confusion matrix
confusionMatrix(data=as.factor(nb.class.test2),reference=as.factor(test$fault_type),positive="1")


nb.post2 <- predict(nb.fit2, newdata=test, type="raw")
pred.nb2 <- prediction(nb.post2[,2],as.factor(test$fault_type))
perf.nb2 <- performance(pred.nb2,"tpr","fpr")
AUC5.2 <- performance(pred.nb2,"auc")@y.values[[1]]
AUC5.2


#####KNN
# K-Nearest Neighbour 
# Normalization

train_selected <- train[, c("Pixels_Areas", "Steel_Plate_Thickness", "range_of_x", "steel_type")]
test_selected <- test[, c("Pixels_Areas", "Steel_Plate_Thickness", "range_of_x", "steel_type")]

train_scaled <- as.data.frame(scale(train_selected[, -ncol(train_selected)]))  
test_scaled <- as.data.frame(scale(test_selected[, -ncol(test_selected)]))    

train_scaled$fault_type <- train$fault_type
test_scaled$fault_type <- test$fault_type

knn.fit <- knn(
  train = train_scaled[ , -ncol(train_scaled)],  
  test = test_scaled[ , -ncol(test_scaled)],    
  cl = train_scaled$fault_type,               
  k = 100)                                     

knn.fit

confusionMatrix(data = as.factor(knn.fit),reference = as.factor(test_scaled$fault_type), positive = "1") 


table(Predicted = knn.fit, Actual = test$fault_type) %>% addmargins()


# PLOTS

# CORRPLOT
clean_data_corr <- subset(clean_data, select = -c(steel_type, fault_type))


clean_data_corr %>% 
  cor %>% 
  corrplot(method = "square",
           type = "lower",
           bg = "white",
           tl.col = "black",
           tl.cex = 0.7)




# MODEL SUMMARY
# LOGISTIC REGRESSIONS
modelsummary(list('Model1'= Logi1,'Model2 (improved via  back. stepwise)'= Logi2Step), 
             fmt = fmt_decimal(digits = 4, pdigits = 4),
             gof_map = c("nobs","r.squared","f"),
             stars = TRUE,
             title = 'Model Summary: Logistic regressions')


# ROC CURVES
# Logistic regressions
plot(perfLogi1, main = 'ROC Curve', col = "blue", grid = TRUE)
plot(perfLogi2Step, add =TRUE, col = "red")
legend(0.742, 0.38, legend = c("Logi1", "Logi2Step"),
       lty = 1,
       col = c("blue", "red"),
       box.lwd = 1,         
       box.col = "black",   
       cex = 0.9)  
text(0.85, 0.19, paste("AUC Logi1: ", round(AUC1, 2)), col = "blue", cex = 0.8)
text(0.87, 0.07, paste("AUC Logi2Step", round(AUC2, 2)), col = "red", cex = 0.8)

# Generative models

plot(perfLDA2, main = 'ROC Curve', col = 'cyan')
plot(perfQDA2, add =TRUE, col = "magenta")
plot(perf.nb2, add=TRUE, col='green1')
legend(0.7,0.5, legend = c("LDA","QDA",'NB'),
       lty=1,
       col = c("cyan","magenta",'green1'))
text(0.85, 0.19, paste("LDA: ", round(AUC1, 2)), col = "cyan", cex = 0.8)
text(0.87, 0.07, paste("QDA: ", round(AUC2, 2)), col = "magenta", cex = 0.8)
text(0.85, 0.19, paste("NB", round(AUC1, 2)), col = "green1", cex = 0.8)

