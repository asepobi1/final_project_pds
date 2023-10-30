install.packages("tidyverse")
install.packages("caret")
install.packages("cluster")
install.packages("rpart")
install.packages("randomForest")

# 1. Eksplorasi Data dan Analisis Klastering
# Load required libraries
library(tidyverse)
library(cluster)

# Load the dataset
# Load the dataset
data <- read.csv("C:/Users/Lenovo/Downloads/AREA/College/Term 5/PDS/Final Project - Dataset/dataset/train.csv")
View(data)
# Perform exploratory data analysis
summary(data)
head(data)
# Visualize data as needed (histograms, scatterplots, etc.)

# Perform clustering analysis (example with k-means)
set.seed(123)  # Set a seed for reproducibility
kmeans_result <- kmeans(data, centers = 3)  # You can choose the number of clusters

# Discuss and interpret the clustering results
table(kmeans_result$cluster)  # Count of data points in each cluster

# 2. Membangun Model Regresi Linear
# Load required libraries
library(caret)

# Build a linear regression model
lm_model <- lm(HousePrice ~ ., data = data)  # Replace "HousePrice" with the actual target variable name

# Explore different models and perform model selection
# You can explore other regression models and evaluate their performance using cross-validation or other techniques.

# Evaluate the prediction performance of the model
summary(lm_model)  # Check the summary of the linear regression model

# Discuss and interpret the results
# Interpret the coefficients, check model assumptions, and assess the model's performance.

# 3. Membangun Model Klasifikasi
# Transform house prices into distinct classes (cheap and expensive)
threshold <- 200000  # Adjust the threshold as needed
data$PriceClass <- ifelse(data$HousePrice <= threshold, "Cheap", "Expensive")

# Load required libraries
library(rpart)
library(randomForest)

# Build a tree-based classification model (e.g., Decision Tree)
tree_model <- rpart(PriceClass ~ ., data = data)

# Perform and discuss methods to increase the model accuracy
# You can try different algorithms (e.g., Random Forest) and tune hyperparameters to improve accuracy.

# Perform k-fold cross-validation
library(caret)
set.seed(123)
cv <- train(PriceClass ~ ., data = data, method = "rf", trControl = trainControl(method = "cv", number = 5))

# Evaluate the classification models
confusionMatrix(cv)
# Choose appropriate evaluation metrics for classification, such as accuracy, precision, recall, F1-score, etc.

# Discuss and interpret the results
# Interpret the model's performance and discuss the implications of classification results.

