# Loading the required libaries
library(caret)

# Setting the random seed
set.seed(42)

# Loading the parkinsons data set
data <- read.csv("processed_data.csv")

# Check structure of data set
str(data)

# Check for missing values, sum = 0 indicates no missing value
sum(is.na(data))

# Splitting data set based on class. 80% train and 20% test
index <- createDataPartition(data$Class, p = 0.80, list = FALSE)

trainSet <- data[index,]
testSet <- data[-index,]

# Check testSet to ensure balanced no. of samples from vowel "a" and "o"
summary(testSet)

# Define the training control for multiple models
fitControl_rf <- trainControl(method ="cv", number = 5, savePredictions = 'final', classProbs = T, search = "grid")
fitControl <- trainControl(method ="cv", number = 5, savePredictions = 'final', classProbs = T)

# Defining predictors and outcome
predictors <- c("Jitter_ppq5", "Shimmer_local", "NTH", "HTN", "Median.pitch", "Standard.deviation", 
                "Minimum.pitch", "Maximum.pitch", "Number.of.pulses", "Standard.deviation.of.period",
                "Fraction.of.locally.unvoiced.frames", "Number.of.voice.breaks", "Degree.of.voice.breaks")
outcomeName <- 'Class'

# Training the random forest model
x <- data[,2:14]
metric <- "Accuracy"
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry = c(1:10))
model_rf <- train(trainSet[,predictors], trainSet[,outcomeName], metric = metric, method = "rf", tuneGrid = tunegrid, trControl = fitControl_rf)

# Predicting rf model using test set
testSet$pred_rf <- predict(object = model_rf, testSet[,predictors])

# Checking accuracy of random forest model
confusionMatrix(testSet$Class, testSet$pred_rf, positive = 'P')

# Training the kNN model
model_knn <- train(trainSet[,predictors], trainSet[,outcomeName], method = "knn", preProcess = c("center", "scale"), trControl = fitControl, tuneLength = 10)

# Predicting knn model using test set
testSet$pred_knn <- predict(object = model_knn, testSet[,predictors])

# Check accuracy of knn model
confusionMatrix(testSet$Class, testSet$pred_knn, positive = 'P')

# Training the SVM model
#gridSVM <- expand.grid(C = c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 5))
model_svmrbf <- train(trainSet[,predictors], trainSet[,outcomeName], method = "svmRadial", trControl = fitControl, preProcess = c("center", "scale"), tuneLength = 10)

# Predicting svm model using test set
testSet$pred_svm <- predict(object = model_svmrbf, testSet[,predictors])

# Check accuracy of SVM model
confusionMatrix(testSet$Class, testSet$pred_svm, positive = 'P')

# Saving the models (to use when we want to save the model)
#save(model_rf, file = "parkinson_modelrf.rda")
#save(model_knn, file = "parkinson_modelknn.rda")
#save(model_svmrbf, file = "parkinson_modelsvmrbf.rda")
  
# Ensembling by majority vote
testSet$pred_majority <- as.factor(ifelse(testSet$pred_rf == 'P' & testSet$pred_knn == 'P', 'P',
                                          ifelse(testSet$pred_rf == 'P' & testSet$pred_svm == 'P', 'P',
                                                 ifelse(testSet$pred_knn == 'P' & testSet$pred_svm == 'P', 'P', 'H'))))

# Check accuracy of ensemble
confusionMatrix(testSet$Class, testSet$pred_majority, positive = 'P')

# To obtain 95% confidence interval for error
CM <- confusionMatrix(testSet$Class, testSet$pred_majority, positive = 'P')
main_stats <- CM$overall
accuracy <- main_stats['Accuracy']
str(accuracy)
error <- 1 - accuracy
n <- nrow(testSet)
# For calculation of error confidence interval, equation is as follows:
# error +/- const * sqrt((error * (1 - error)) / n)
# where const = 1.96 for 95% confidence and n is the no. of instances in test set
const <- 1.96
error_dev <- const * sqrt((error * (1 - error)) / n)
max_error = error + error_dev
min_error = error - error_dev
sprintf("The error translates to %.4f +/- %.4f at 95%% confidence interval.",error, error_dev)

