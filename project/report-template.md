# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Devika Lakshmanan

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
If the predicted count was negative then the prediction was rejected by kaggle for this submission dataset. Hence, all rows for which negative count was predicted, the predicted count was replaced with 0 value. 

### What was the top ranked model that performed?
For the initial model, the fit's presets parameter was set to 'best_quality' as per the project instructions to be trained within 10 mins. Also the hyperparameters was left to be the 'default' setting. These settings focused on maximizing the prediction's accuracy and also allowed for bagging and stacking since auto_stack is 'True' with above settings.

It was noticed that in the above scenario, WeightedEnsemble_L3 turned out to be the top ranking model.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
datetime column was not providing enough data to discover patterns in the bike rent dataset.
It was important to analyse the set based on trends like day of the month, day of the week, hour of the day, month etc.

Hence, datetime column was changed from object datatype to DateTime64 datatype

### How much better did your model preform after adding additional features and why do you think that is?
The model performed drastically better after the addition of new features. The public score in kaggle improved from 1.84484 to 0.6505. Since the submissions are evaluated on Root Mean Squared Logarithmic Error (RMSLE), this reduction represents a significant reduction in prediction error.

By engineering new features like day, hour , month etc. we made the dataset more informative and the model was able to uncover new patterns to increase the predictive performance.

When I ran the feature importance method against the predictor tuned with new features it was noticed that hour of the day which was extracted from datetime column was listed as the most important feature which would have significantly improved the accuracy of the predictions.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
I attempted 4 rounds of hyperparameter tuning as follows:
    - Attempt 1: 
        Changes: Set parameter 'hyperparameter' = light and removed parameter presets = 'best_quality'.
        Observed: Accuracy of models reduced a bit but Autogluon was able to train 2 more models as compared with    previous setting where 'hyperparameter' = default and presets = 'best_quality'
        Result: Score changed from 0.6505 to 0.51559
    - Attempt 2:
        Changes: Added auto_stack=True while maintaining the other settings for Hyperparameter optimization 1
        Observed: Accuracy of models increased with bagging and stacking enabled but Autogluon was able to train lesser number of models. Only 6 models were trained.
        Result: Score changed from 0.51559 to 0.46962
    - Attempt 3:
        Changes: Removed auto_stack=True and added a separate validation set using tuning_data while maintaining the other settings for Hyperparameter optimization 2
        Observed: Accuracy of models reduced when bagging and stacking was removed but Autogluon was again able to train 10 models. auto_stack when set to True in Attempt 2 was able to hold out data for validation and build ensemble models. So, that option was better than explicitly specifying tuning_data and disbabling bagging/stacking.
        Result: Score changed from 0.46962 to 0.52032
    - Attempt 4:
        Changes: Set hyperparameters and hyperparameter_tune_kwargs to dictionary values for 4 models i.e. LightGBM, CatBoost, XGBoost, neural network implemented in Pytorch and using the other settings as per HPO1
        Observed: This setting trained 8 models however the best models performance was less than the best model achieved with other Hyperparameter tuning settings.
        Result: Score changed from 0.52032 to 0.53885
        
### If you were given more time with this dataset, where do you think you would spend more time?
I would have spent more time in understanding the training data in more depth and in engineering more features. 
I think there is a lot more scope for feature engineering which could result in models with improved performance.
Also, I would have tried to find hyperparameter optimization parameters for individual models that could yield better performance.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

|model|hpo1|hpo2|hpo3|hpo4|score|
|--|--|--|--|--|
|initial|1.84484|1.84484|1.84484|1.84484|1.84484|
|add_features|0.65050|0.65050|0.65050|0.65050|0.65050|
|hpo|0.51559|0.46962|0.52032|0.53885|0.46962|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary

AutoGluon is a powerful AutoML library which helps to train various models as per the use case.
For the Bike Sharing Demand problem with a tabular dataset, the TabularPredictor provided by AutoGluon library trained several powerful models based on the provided configurations.

The models trained with default configuration helped build a baseline model.
With a bit of feature engineering, AutoGluon was able to train better models with more predictive accuracy. So, it is important to understand the data in depth and engineer features that can potentially boost the predictive performance.

With all 3 Hyperparameter optimizations, score further improved. It was noticed that some tradeoffs in accuracy were balanced with additional models that were trained in the given time limit of 10 mins. So, it is important to try various presets and not just stick to 'best_quality' as the processing time and disk space constraint in a given environment might limit the number of models trained with 'best_quality' preset.

Also, model ensembling with stacking/bagging improves model performance. If more training time and disk space is available then these options must be explored.

Finally tried hyperparameter optimization for top 3 models achieved in the first round of hyperparameter optimization to see if a better model can be arrived at. The output of HPO1 was chosen for optimization since the top models output by HPO2 (extremely randomized trees, random forest) were all non configurable as per the documention in [AutoGluon TabularPredictor documentation](https://auto.gluon.ai/stable/api/autogluon.predictor.html#autogluon.tabular.TabularPredictor.fit)
The hyperparameter optimization in this attempt didn't yield in a better model. It performed the worst among all rounds of Hyperparameter optimization.



