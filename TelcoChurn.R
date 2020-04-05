#Setup Packages
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("caret")) install.packages("caret")
if (!require("car")) install.packages("car")
if (!require("MASS")) install.packages("MASS")
if (!require("randomForest")) install.packages("randomForest")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("pROC")) install.packages("pROC")

library(tidyverse)
library(caret)
library(car)
library(MASS)
library(randomForest)
library(rpart.plot)
library(pROC)

################################
# Analysis
################################

#Load the dataset
cust_data = read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
#Remove rows with NA values
cust_data = na.omit(cust_data)

summary(cust_data)

ggplot(cust_data, aes(fill = Contract, x = Churn)) + 
  geom_bar(stat = "count") + 
  ggtitle("Churn by Contract Type") + 
  theme(plot.title = element_text(hjust = 0.5))

ggplot(cust_data, aes(x = tenure, y = TotalCharges, color = Churn)) +
  geom_point() + 
  ggtitle("Total Charges vs. Tenure with Churn Details") + 
  theme(plot.title = element_text(hjust = 0.5))

# ggplot(cust_data, aes(x = tenure, y = MonthlyCharges, color = Churn)) + 
#   geom_point()

# ggplot(data = cust_data, aes(tenure)) + 
#   geom_histogram()
breaks_t = seq(0,75,5)
cust_data$tenure_bin = cut(cust_data$tenure, breaks_t)

tenure_bin_table = table(cust_data$tenure_bin, cust_data$Churn)
knitr::kable(tenure_bin_table)

ggplot(cust_data, aes(tenure_bin, fill = Churn)) + 
  geom_bar() +
  ggtitle("Customer Count by Tenure with Churn Details") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5))

# ggplot(data = cust_data, aes(MonthlyCharges)) + 
#   geom_histogram()
breaks_mc = seq(10, 120, 10)
cust_data$MonthlyCharges_Bin = cut(cust_data$MonthlyCharges, breaks_mc)

mc_bin_table = table(cust_data$MonthlyCharges_Bin, cust_data$Churn)
knitr::kable(mc_bin_table)

ggplot(cust_data, aes(MonthlyCharges_Bin, fill = Churn)) + 
  geom_bar() + 
  ggtitle("Customer Count by Monthly Charges with Churn Details") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(hjust = 0.5))

################################
# Prediction Methods
################################
# columns 2, 4-5, 7-18 contain categorical variables that should be converted
# colnames(cust_data)

dv = dummyVars(~ ., 
               data = cust_data[,c(2,4:5,7:18)], 
               fullRank = TRUE)
enc_cust = predict(dv, newdata = cust_data)

#reconstruct the dataset - include the encoded variables now
cust_data_enc = cbind(cust_data[,-c(2,4:5,7:18)], enc_cust)

#remove columns not necessary for prediction
remove_cols = c("customerID", "tenure_bin", "MonthlyCharges_Bin")
cust_data_new = cust_data_enc[, -which(colnames(cust_data_enc) %in% remove_cols)]

#create correlation matrix
cust_cor = cor(cust_data_new[,-5])

#create correlation plot
cor_plot = levelplot(cust_cor)
cor_plot

#remove highly correlated variables
cust_data_mod = subset(cust_data_new, 
                       select = -c(`OnlineSecurity.No internet service`,
                                   `OnlineBackup.No internet service`, `TechSupport.No internet service`,
                                   `StreamingTV.No internet service`, `StreamingMovies.No internet service`,
                                   `DeviceProtection.No internet service`, `MultipleLines.No phone service`))

# levelplot(cor(cust_data_mod[,-4]), aspect = "iso", scales=list(x=list(rot=90)))

#Build training and test sets
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = cust_data_mod$Churn, times = 1, p = 0.2, list = FALSE)

cust_train = cust_data_mod[-test_index,]
cust_test = cust_data_mod[test_index,]

#Logistic Regression Model

#Model 1 - use all variables
churn_log_model1 = glm(Churn ~ ., data = cust_train, family = binomial)
# coef(summary(churn_log_model1)) #show model coefficients summary
knitr::kable(coef(summary(churn_log_model1))[1:10,]) #show model coefficients summary

#Check for collinearity 
vif_model1 = vif(churn_log_model1)
knitr::kable(vif_model1[vif_model1 > 5]) #show only collinear variables

# Check correlation plot for tenure, TotalCharges, MonthlyCharges
cust_train_cor1 = cor(cust_train[2:4,2:4])
cust_train_corplot1 = levelplot(cust_train_cor1, aspect = "iso", scales=list(x=list(rot=90)))
cust_train_corplot1 #tenure, MonthlyCharges, and TotalCharges are highly correlated

#Model 2 - remove MonthlyCharges and tenure variables.
#Create new cust_train set with variables removed
cust_train2 = subset(cust_train, 
                    select = -c(MonthlyCharges, TotalCharges))

churn_log_model2 = glm(Churn ~ ., data = cust_train2, family = binomial)
#summary(churn_log_model2)
p_values_model2 = coef(summary(churn_log_model2))[,4]
knitr::kable(p_values_model2[p_values_model2 > 0.05])

vif_model2 = vif(churn_log_model2)
vif_model2[vif_model2 > 5] #there is no collinearity

#Model 3 - remove variables that are not statistically significant
cust_train3 = subset(cust_train2,
                     select = -c(gender.Male, Partner.Yes, 
                                 OnlineBackup.Yes, DeviceProtection.Yes))

churn_log_model3 = glm(Churn ~ ., data = cust_train3, family = binomial)
knitr::kable(coef(summary(churn_log_model3)))

#Setup Train and Test Sets
cust_train_f = cust_train3
cust_test_f = cust_test[,colnames(cust_train_f)]

#Predictions on Training Set
glm_churn_probs_train = predict(churn_log_model3, data = cust_train_f, 
                               type = "response")
#Find the optimal cutoff level for deciding Yes/No
cutoffs = seq(0.2, 0.8, 0.01)

optimal_cutoff_func = function(i) {
  glm_churn_pred_train = rep("No", length(glm_churn_probs_train))
  glm_churn_pred_train[glm_churn_probs_train > i] = "Yes"
  accuracy = mean(glm_churn_pred_train ==  cust_train_f$Churn)
  return(accuracy)
}

cutoff_accs = sapply(cutoffs, optimal_cutoff_func)
opt_cutoff = cutoffs[which.max(cutoff_accs)]

glm_churn_pred_train = rep("No", length(glm_churn_probs_train))
glm_churn_pred_train[glm_churn_probs_train > opt_cutoff] = "Yes"

table(glm_churn_pred_train, cust_train_f$Churn)
mean(glm_churn_pred_train ==  cust_train_f$Churn) #80.39111% accuracy

# #Predictions on Test Set
# glm_churn_probs_test = predict(churn_log_model3, cust_test_f, 
#                                 type = "response")
# glm_churn_pred_test = rep("No", length(glm_churn_probs_test))
# glm_churn_pred_test[glm_churn_probs_test > opt_cutoff] = "Yes"
# 
# table(glm_churn_pred_test, cust_test_f$Churn)
# mean(glm_churn_pred_test ==  cust_test_f$Churn) #81.59204% accuracy

# #LDA
# 
# #Build LDA model
# churn_lda_model = lda(Churn ~ ., data = cust_train_f)
# churn_lda_model
# 
# #Predict on training set
# lda_churn_pred = predict(churn_lda_model, data = cust_train_f)
# lda_churn_pred_class = lda_churn_pred$class
# 
# table(lda_churn_pred_class, cust_train_f$Churn)
# mean(lda_churn_pred_class == cust_train_f$Churn)

# Decision Tree
#Format column names (remove spaces)
colnames(cust_train_f) <- make.names(colnames(cust_train_f))
colnames(cust_test_f) <- make.names(colnames(cust_test_f))

#Build Decision Tree on Training Set
churn_train_rpart <- train(Churn ~ ., data = cust_train_f,
                           tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 50)),
                           method = "rpart")
confusionMatrix(churn_train_rpart)

#Plots
plot(churn_train_rpart)
rpart.plot(churn_train_rpart$finalModel)

# Random Forest
churn_train_randForest <- randomForest(Churn ~ ., data = cust_train_f,
                                       ntrees = 1000)
churn_train_randForest$confusion

################################
# Final Results
################################

#Apply models on test set
#Logistic Regression
cust_train_f = cust_train3
cust_test_f = cust_test[,colnames(cust_train_f)]

glm_churn_probs_test = predict(churn_log_model3, cust_test_f,
                                type = "response")
glm_churn_pred_test = rep("No", length(glm_churn_probs_test))
glm_churn_pred_test[glm_churn_probs_test > opt_cutoff] = "Yes"

glm_tbl = table(glm_churn_pred_test, cust_test_f$Churn)

glm_metrics = c(mean(glm_churn_pred_test ==  cust_test_f$Churn), #81.59204% accuracy
recall(glm_tbl), #91.48% recall
precision(glm_tbl)) #84.68% precision

#Decision Trees
#Format column names (remove spaces)
colnames(cust_train_f) <- make.names(colnames(cust_train_f))
colnames(cust_test_f) <- make.names(colnames(cust_test_f))

pred_churn_test_rpart = predict(churn_train_rpart, cust_test_f)
rpart_tbl = table(pred_churn_test_rpart, cust_test_f$Churn)

rpart_metrics = c(mean(pred_churn_test_rpart ==  cust_test_f$Churn), #78.32267% accuracy
recall(rpart_tbl), #88.577% recall
precision(rpart_tbl)) #83.03% precision

#Random Forest
pred_churn_test_rf = predict(churn_train_randForest, cust_test_f)
rf_tbl = table(pred_churn_test_rf, cust_test_f$Churn)

rf_metrics = c(mean(pred_churn_test_rf ==  cust_test_f$Churn), #80.31% accuracy
recall(rf_tbl), #90.13% recall
precision(rf_tbl)) #84.18% precision

#Result Summary
full_metrics = data.frame(glm_metrics, rpart_metrics, rf_metrics)
rownames(full_metrics) = c("accuracy", "recall", "precision")
knitr::kable(full_metrics)

#ROC comparison
# glm_churn_pred_test1 = ifelse(glm_churn_pred_test == "Yes", 1, 0)
# pred_churn_test_rpart1 = ifelse(pred_churn_test_rpart == "Yes", 1, 0)
# pred_churn_test_rf1 = ifelse(pred_churn_test_rf == "Yes", 1, 0)
# 
# plot(roc(response = cust_test_f$Churn, predictor = glm_churn_pred_test1))
# plot(roc(response = cust_test_f$Churn, predictor = pred_churn_test_rpart1))
# plot(roc(response = cust_test_f$Churn, predictor = pred_churn_test_rf1))

