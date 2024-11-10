
# MLP model using Grid Search with Boston Housing Data

#install.packages("brulee")




library(brulee)
library (tidymodels)
library (tidyverse)

tidymodels_prefer()

# Load an inspect data

data("BostonHousing", package = "mlbench")

summary(BostonHousing)


# Set seed for reproducibility
set.seed(96)

# Create training and test split 
dataSplit <- initial_split (BostonHousing, prop=0.8)
trainBost <- training (dataSplit)
testBost <- testing (dataSplit)

# CV set 
cvBost <- vfold_cv(trainBost, v = 5)

cvBost

glimpse (trainBost)

# Model with training hyper-parameters
modelMLP <- mlp(
  hidden_units = tune(), 
  dropout = tune(), 
  epochs = tune(),
  activation = "relu", 
  engine="brulee", 
  mode="regression")

modelMLP

# Params 
params <- extract_parameter_set_dials(modelMLP)

params

# Define recipe
recipeMPL <- recipe (medv ~ ., data=trainBost) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_mutate(chas = as.integer(chas))

# Setup the workflow 
workflowMP <- workflow() %>%
  add_model(modelMLP) %>%
  add_recipe(recipeMPL)

# Setup Grid search using random values for the tuning parameters
gridMLP <- grid_random(
  params, #Parameters to tune
  size = 20 # Random combinations
)

glimpse(gridMLP)



# Tune MLP 
tuneMLP <- tune_grid(
  workflowMP,
  resamples = cvBost,
  grid = gridMLP
)

as.data.frame(tuneMLP$.metrics) %>% filter (.metric=='rmse') %>% arrange(.estimate) %>% select (hidden_units, dropout, epochs, .metric, .estimate)




# Get the best parameters from the tuning
paramsMLPBest <- select_best(tuneMLP, metric="rmse")
#View the best parameters
paramsMLPBest

# Add the parameters to the workflow
wfMLPBest <- finalize_workflow(workflowMP, paramsMLPBest)

wfMLPBest

# Fit the final model and use it to predict medv on the training data 
mlpFinal<- fit(wfMLPBest, data = trainBost)
predictMLP <- predict(mlpFinal, new_data = testBost) 

#Apply final model to the test dataset and generate some metrics
predictMLP %>% mutate(actual=testBost$medv) %>% metrics(truth = actual, estimate = .pred)






















