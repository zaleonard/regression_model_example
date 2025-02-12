***Regression modeling blueprint***

This repo contains a blueprint that I have put together for running regression modeling type tasks.  It takes an excel file as an input for now, and relies on the user for determining which parameters to select for model training

***How It's Made***

Tech used: Python.  A requirements.txt is included for relevant modules.  Create environment using requirements.txt
The script is broken up into a few different sections:
1. data cleaning - the lines of code are short, since data cleaning is very subjective of the dataset
2. Exploratory data analysis - A short example of stripplots and boxplots are used to compare parameters, a find_outliers function is defined to quickly find outliers in datasets.  Additionally, a function to plot columns side-by-side may be helpful for further dataset context
3. Dataset normalization if applicable.  Three scaler methods are used in this case, but more can be added if necessary
4. model training and learning_curve graph generation - for more information about the dataset and where the variance and bias are located within a model.  The models chosen for this comparison are linear regression, adaboost, xgboos, decision tree, random forest, KNN, and SVM
5. Hyperparameter tuning - comment out as necessary if too computationally expensive
6. Model comparison using cross-val-score metrics
7. Model prediction graphs to aid in model selection
8. Final model selection, feature importance, and saving the model

***Optimizations***

- there are some redundencies in the code
- Develop more OOP version of this code
- Allow for easier plug-and-play of model selection
