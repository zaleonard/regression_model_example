# %%
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    explained_variance_score,
    max_error,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV, 
    learning_curve, 
    cross_val_score
)


#upload file
df = pd.read_excel(
    r'...'
)

# drop Acid Linker_B (%), NHS eq, EDC eq, EDA eq, Ext Linker Difference_FGI-Amine (%)
print(df.columns)
print(df.info())

# drop columns of interest based on above
df = df.drop(
    labels=[
        "...",
        "...",
        "...",
    ],
    axis=1,
)

#apply other necessary cleaning here
numerical_cols = [
    "...",
    "...",
]

#EDA code, example below
for column in df[numerical_cols].columns:
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=df[numerical_cols], y=column, color='blue', jitter=True)
    sns.boxplot(data=df[numerical_cols], y=column, color='orange', whis=1.5)
    plt.title(f'Individual Value Plot with Boxplot Overlay for {column}')
    plt.ylabel(column)
    plt.show()

def find_outliers(df, threshold=3):
    """finds outliers in a dataframe

    Args:
        df (dataframe object): dataframe of float columns
        threshold (int, optional): threshold for outliers. Defaults to 3.

    Returns:
        list: returns the rows of the dataframe with outliers
    """
    outlier_indices = []
    
    # Calculate Z-scores for each column
    z_scores = np.abs((df - df.mean()) / df.std())
    
    # Identify outliers for each column
    for column in df.columns:
        outliers = z_scores[column] > threshold
        outlier_indices.extend(df.index[outliers].tolist())
    
    # Remove duplicates
    outlier_indices = list(set(outlier_indices))
    
    return outlier_indices


# Remove outliers from the DataFrame
outlier_indices = find_outliers(df[numerical_cols])
df_cleaned = df.drop(outlier_indices)

print("Original DataFrame:")
print(df)
print("\nIndices of outliers:")
print(outlier_indices)
print("\nDataFrame after removing outliers:")
print(df_cleaned)

def plot_columns_side_by_side(df1, df2):
    """plots datafarmes with the same columns side by side

    Args:
        df1 (dataframe object): dataframe, pre data-cleaning for example
        df2 (dataframe object): dataframe, post data-cleaning
    """
    for column in df1.columns:
        if column in df2.columns:
            fig, axes = plt.subplots(1, 2, figsize=(20, 6))
            
            # Plot for df1
            sns.stripplot(data=df1, y=column, color='blue', jitter=True, ax=axes[0])
            sns.boxplot(data=df1, y=column, color='orange', whis=1.5, ax=axes[0])
            axes[0].set_title(f'{column} in {df1}')
            axes[0].set_ylabel(column)
            
            # Plot for df2
            sns.stripplot(data=df2, y=column, color='blue', jitter=True, ax=axes[1])
            sns.boxplot(data=df2, y=column, color='orange', whis=1.5, ax=axes[1])
            axes[1].set_title(f'{column} in {df2}')
            axes[1].set_ylabel(column)
            
            # Show plot
            plt.show()

#plot columns
plot_columns_side_by_side(df, df_cleaned)
    
#data scaling
categorical_cols = [
    "...",
    "..."
]
df_floats = df[numerical_cols]


# modeling
# determining the best scaling method (of those shown)
scaler = preprocessing.MinMaxScaler()
minmax_scaled_df_floats = scaler.fit_transform(df_floats)
minmax_scaled_df_floats = pd.DataFrame(
    minmax_scaled_df_floats, columns=df_floats.columns
)

scaler = preprocessing.StandardScaler()
scaled_df_floats = scaler.fit_transform(df_floats)
scaled_df_floats = pd.DataFrame(scaled_df_floats, columns=df_floats.columns)

scaler = preprocessing.RobustScaler()
robust_scaled_df_floats = scaler.fit_transform(df_floats)
robust_scaled_df_floats = pd.DataFrame(
    robust_scaled_df_floats, columns=df_floats.columns
)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(15, 5))

for col in df_floats:
    ax1.set_title("Before Scaling")
    sns.kdeplot(df_floats[col], ax=ax1)
for col in df_floats:
    ax2.set_title("After Min-Max Scaling")
    sns.kdeplot(minmax_scaled_df_floats[col], ax=ax2)
for col in df_floats:
    ax3.set_title("After Standard Scaling")
    sns.kdeplot(scaled_df_floats[col], ax=ax3)
for col in df_floats:
    ax4.set_title("After Robust Scaling")
    sns.kdeplot(robust_scaled_df_floats[col], ax=ax4)
plt.show()

#Choose your scaler, standardscaler chosen here
scaler = preprocessing.StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df_encoded = pd.get_dummies(df, columns=categorical_cols)
print(df_encoded.columns)

# train the models
X = df_encoded.drop(["target_parameter"], axis=1) #change to target parameter
y = df_encoded["target_parameter"] #change to target parameter

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.3)

# generating learning curve and kfold validation to see how good the data set is
# linear regression
train_sizes = [10, 20, 40, 60]

train_sizes_lr, train_scores_lr, validation_scores_lr = learning_curve(
    estimator=LinearRegression(),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
)
train_scores_mean_lr = -train_scores_lr.mean(axis=1)
validation_scores_mean_lr = -validation_scores_lr.mean(axis=1)

# Xgboost
train_sizes_xgb, train_scores_xgb, validation_scores_xgb = learning_curve(
    estimator=GradientBoostingRegressor(),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
)
train_scores_mean_xgb = -train_scores_xgb.mean(axis=1)
validation_scores_mean_xgb = -validation_scores_xgb.mean(axis=1)

# Adaboost
train_sizes_ada, train_scores_ada, validation_scores_ada = learning_curve(
    estimator=AdaBoostRegressor(),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
)
train_scores_mean_ada = -train_scores_ada.mean(axis=1)
validation_scores_mean_ada = -validation_scores_ada.mean(axis=1)

# DecisionTree
train_sizes_dt, train_scores_dt, validation_scores_dt = learning_curve(
    estimator=DecisionTreeRegressor(),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
)
train_scores_mean_dt = -train_scores_dt.mean(axis=1)
validation_scores_mean_dt = -validation_scores_dt.mean(axis=1)

# randomforest
train_sizes_rf, train_scores_rf, validation_scores_rf = learning_curve(
    estimator=RandomForestRegressor(),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
)
train_scores_mean_rf = -train_scores_rf.mean(axis=1)
validation_scores_mean_rf = -validation_scores_rf.mean(axis=1)

# KNN
train_sizes_knn, train_scores_knn, validation_scores_knn = learning_curve(
    estimator=KNeighborsRegressor(),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
)
train_scores_mean_knn = -train_scores_knn.mean(axis=1)
validation_scores_mean_knn = -validation_scores_knn.mean(axis=1)

# SVM
train_sizes_svm, train_scores_svm, validation_scores_svm = learning_curve(
    estimator=SVR(),
    X=X,
    y=y,
    train_sizes=train_sizes,
    cv=5,
    scoring="neg_mean_squared_error",
    shuffle=True,
)
train_scores_mean_svm = -train_scores_svm.mean(axis=1)
validation_scores_mean_svm = -validation_scores_svm.mean(axis=1)

#plot learning curves
fig, ax = plt.subplots(3, 3, figsize=(15, 13))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

ax[0, 0].plot(train_sizes_lr, train_scores_mean_lr, label="Training error")
ax[0, 0].plot(train_sizes_lr, validation_scores_mean_lr, label="Validation error")
ax[0, 0].set_xlabel(f"Training set size", fontsize=14)
ax[0, 0].set_ylabel("MSE", fontsize=14)
ax[0, 0].set_title("Linear Regression learning curve")
ax[0, 0].legend()

ax[2, 0].plot(train_sizes_xgb, train_scores_mean_xgb, label="Training error")
ax[2, 0].plot(train_sizes_xgb, validation_scores_mean_xgb, label="Validation error")
ax[2, 0].set_xlabel(f"Training set size", fontsize=14)
ax[2, 0].set_ylabel("MSE", fontsize=14)
ax[2, 0].set_title("XgBoost learning curve")
ax[2, 0].legend()

ax[0, 1].plot(train_sizes_ada, train_scores_mean_ada, label="Training error")
ax[0, 1].plot(train_sizes_ada, validation_scores_mean_ada, label="Validation error")
ax[0, 1].set_xlabel(f"Training set size", fontsize=14)
ax[0, 1].set_ylabel("MSE", fontsize=14)
ax[0, 1].set_title("AdaBoost learning curve")
ax[0, 1].legend()

ax[0, 2].plot(train_sizes_dt, train_scores_mean_dt, label="Training error")
ax[0, 2].plot(train_sizes_dt, validation_scores_mean_dt, label="Validation error")
ax[0, 2].set_xlabel(f"Training set size", fontsize=14)
ax[0, 2].set_ylabel("MSE", fontsize=14)
ax[0, 2].set_title("Decision Tree learning curve")
ax[0, 2].legend()

ax[1, 0].plot(train_sizes_rf, train_scores_mean_rf, label="Training error")
ax[1, 0].plot(train_sizes_rf, validation_scores_mean_rf, label="Validation error")
ax[1, 0].set_xlabel(f"Training set size", fontsize=14)
ax[1, 0].set_ylabel("MSE", fontsize=14)
ax[1, 0].set_title("Random Forest learning curve")
ax[1, 0].legend()

ax[1, 1].plot(train_sizes_knn, train_scores_mean_knn, label="Training error")
ax[1, 1].plot(train_sizes_knn, validation_scores_mean_knn, label="Validation error")
ax[1, 1].set_xlabel(f"Training set size", fontsize=14)
ax[1, 1].set_ylabel("MSE", fontsize=14)
ax[1, 1].set_title("KNN learning curve")
ax[1, 1].legend()

ax[1, 2].plot(train_sizes_svm, train_scores_mean_svm, label="Training error")
ax[1, 2].plot(train_sizes_svm, validation_scores_mean_svm, label="Validation error")
ax[1, 2].set_xlabel(f"Training set size", fontsize=14)
ax[1, 2].set_ylabel("MSE", fontsize=14)
ax[1, 2].set_title("SVM learning curve")
ax[1, 2].legend()

plt.show()

# hyperparameter tuning
# xgboost
tuned_parameters = [
    {
        "max_depth": [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200],
        "n_estimators": [10, 25, 50, 100, 200, 300, 400, 500],
    }
]
RMSE_xgb = ["root_mean_squared_error(y_test, y_pred2)"]
for value in RMSE_xgb:
    xgbregr = GridSearchCV(GradientBoostingRegressor(), tuned_parameters, cv=4)
    xgbregr.fit(X_train, y_train)
    y_true, y_pred2 = y_test, xgbregr.predict(X_test)

# adaboost
tuned_parameters = [
    {"learning_rate": [0.1, 1, 2, 3, 4, 5], "n_estimators": [100, 200, 300, 400, 500]}
]
RMSE_ada = ["root_mean_squared_error(y_test, y_pred3)"]
for value in RMSE_ada:
    adaregr = GridSearchCV(AdaBoostRegressor(), tuned_parameters, cv=4)
    adaregr.fit(X_train, y_train)
    y_true, y_pred3 = y_test, adaregr.predict(X_test)

# decisiontree
tuned_parameters = [{"max_depth": [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 200]}]
RMSE_dt = ["root_mean_squared_error(y_test, y_pred4)"]
for value in RMSE_dt:
    regressor_dt = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=4)
    regressor_dt.fit(X_train, y_train)
    y_true, y_pred4 = y_test, regressor_dt.predict(X_test)

# random forests
tuned_parameters = [
    {
        "max_depth": [5, 10, 15, 20, 50, 70],
        "n_estimators": [10, 25, 50, 100, 150, 200, 250],
    }
]
RMSE_rf = ["root_mean_squared_error(y_test, y_pred5)"]
for value in RMSE_rf:
    regr_rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=4)
    regr_rf.fit(X_train, y_train)
    y_true, y_pred5 = y_test, regr_rf.predict(X_test)


# KNN
scaler = preprocessing.RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

tuned_parameters = [{"n_neighbors": [1, 2, 3, 4, 5, 10, 15, 20], "p": [1, 2]}]
RMSE_knn = ["root_mean_squared_error(y_test,y_pred6)"]
for i in RMSE_knn:
    model = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=4)
    model.fit(X_train_scaled, y_train)
    y_true, y_pred6 = y_test, model.predict(X_test_scaled)

# SVM
tuned_parameters = [
    {
        "kernel": ["linear", "rbf", "poly"],
        "C": [1, 2, 3, 5, 6, 7, 10],
        "gamma": [0.0001, 0.001, 0.01, 0.1, 1],
    }
]
RMSE_svm = ["root_mean_squared_error(y_test, y_pred7)"]
for value in RMSE_svm:
    svr_regr = GridSearchCV(SVR(), tuned_parameters, cv=4)
    svr_regr.fit(X_train_scaled, y_train)
    y_true, y_pred7 = y_test, svr_regr.predict(X_test_scaled)

print("The best hyper-parameters for Xgboost are: ", xgbregr.best_params_)
print("The best hyper-parameters for AdaBoost are: ", adaregr.best_params_)
print("The optimum max-depth for Decision Tree is: ", regressor_dt.best_params_)
print("The best hyper-parameters for Random Forests are: ", regr_rf.best_params_)
print("The best hyper-parameters for KNN are: ", model.best_params_)
print("The best hyper-parameters for SVR are: ", svr_regr.best_params_)

# predictions based on best hyper parameters
# linear regression
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# xgboost
xgb = GradientBoostingRegressor(
    n_estimators=xgbregr.best_params_["n_estimators"],
    max_depth=xgbregr.best_params_["max_depth"],
    random_state=0,
)
xgb.fit(X_train, y_train)

# Adaboost
adaregr = AdaBoostRegressor(
    learning_rate=adaregr.best_params_["learning_rate"],
    n_estimators=adaregr.best_params_["n_estimators"],
)
adaregr.fit(X_train, y_train)

# Decision tree
regressor_dt = DecisionTreeRegressor(
    random_state=0, max_depth=regressor_dt.best_params_["max_depth"]
)
regressor_dt.fit(X_train, y_train)

# random forests
regr_rf = RandomForestRegressor(
    max_depth=regr_rf.best_params_["max_depth"],
    random_state=0,
    n_estimators=regr_rf.best_params_["n_estimators"],
)
regr_rf.fit(X_train, y_train)

# KNN
neigh = KNeighborsRegressor(
    n_neighbors=model.best_params_["n_neighbors"],
    metric="minkowski",
    p=model.best_params_["p"],
)
neigh.fit(X_train_scaled, y_train)

# SVR
svr_regr = SVR(
    gamma=0.0001, kernel=svr_regr.best_params_["kernel"], C=svr_regr.best_params_["C"]
)
svr_regr.fit(X_train_scaled, y_train)

# %%
# cross-val-score
# default scorere for each
# linear regression - R^2
# xgboost is RMSE
# adaboost is R^2
# decision tree is MSE
# random forest is R^2
# KNN is R^2
# SVR is R^2

# predictions and model comparisons
y_pred1 = regressor.predict(X_test_scaled)  # linear regression
y_pred2 = xgb.predict(X_test)  # xgboost
y_pred3 = adaregr.predict(X_test)  # Adaboost
y_pred4 = regressor_dt.predict(X_test)  # decision tree
y_pred5 = regr_rf.predict(X_test)  # Random Forests
y_pred6 = neigh.predict(X_test_scaled)  # KNN
y_pred7 = svr_regr.predict(X_test_scaled)  # SVR

RMSE_lr = root_mean_squared_error(y_test, y_pred1)
RMSE_xgb = root_mean_squared_error(y_test, y_pred2)
RMSE_ada = root_mean_squared_error(y_test, y_pred3)
RMSE_dt = root_mean_squared_error(y_test, y_pred4)
RMSE_rf = root_mean_squared_error(y_test, y_pred5)
RMSE_knn = root_mean_squared_error(y_test, y_pred6)
RMSE_svr = root_mean_squared_error(y_test, y_pred7)

print(
    "Compare this score(R^2 value of kfold x-val) to the R^2 scores below to determine kfolds vs split_train cross_val methods"
)

linear_cvs = (
    round(cross_val_score(regressor, X, y, cv=5).mean(), 2),
    "+/-",
    round(cross_val_score(regressor, X, y, cv=5).std(), 2),
)
xgb_cvs = (
    round(cross_val_score(xgb, X, y, cv=5, scoring="r2").mean(), 2),
    "+/-",
    round(cross_val_score(xgb, X, y, cv=5, scoring="r2").std(), 2),
)
ada_cvs = (
    round(cross_val_score(adaregr, X, y, cv=5).mean(), 2),
    "+/-",
    round(cross_val_score(adaregr, X, y, cv=5).std(), 2),
)
dt_cvs = (
    round(cross_val_score(regressor_dt, X, y, cv=5, scoring="r2").mean(), 2),
    "+/-",
    round(cross_val_score(regressor_dt, X, y, cv=5, scoring="r2").std(), 2),
)
rf_cvs = (
    round(cross_val_score(regr_rf, X, y, cv=5).mean(), 2),
    "+/-",
    round(cross_val_score(regr_rf, X, y, cv=5).std(), 2),
)
knn_cvs = (
    round(cross_val_score(neigh, X, y, cv=5).mean(), 2),
    "+/-",
    round(cross_val_score(neigh, X, y, cv=5).std(), 2),
)
svr_cvs = (
    round(cross_val_score(svr_regr, X, y, cv=5).mean(), 2),
    "+/-",
    round(cross_val_score(svr_regr, X, y, cv=5).std(), 2),
)

linear_r2 = round((r2_score(y_test, y_pred1)), 3)
xgb_r2 = round((r2_score(y_test, y_pred2)), 3)
ada_r2 = round((r2_score(y_test, y_pred3)), 3)
dt_r2 = round((r2_score(y_test, y_pred4)), 3)
rf_r2 = round((r2_score(y_test, y_pred5)), 3)
knn_r2 = round((r2_score(y_test, y_pred6)), 3)
svr_r2 = round((r2_score(y_test, y_pred7)), 3)

linear_adj = round(
    (1 - (1 - r2_score(y_test, y_pred1)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3
)
xgb_adj = round(
    (1 - (1 - r2_score(y_test, y_pred2)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3
)
ada_adj = round(
    (1 - (1 - r2_score(y_test, y_pred3)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3
)
dt_adj = round(
    (1 - (1 - r2_score(y_test, y_pred4)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3
)
rf_adj = round(
    (1 - (1 - r2_score(y_test, y_pred5)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3
)
knn_adj = round(
    (1 - (1 - r2_score(y_test, y_pred6)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3
)
svr_adj = round(
    (1 - (1 - r2_score(y_test, y_pred7)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)), 3
)

linear_rmse = round(RMSE_lr, 2)
xgb_rmse = round(RMSE_xgb, 2)
ada_rmse = round(RMSE_ada, 2)
dt_rmse = round(RMSE_dt, 2)
rf_rmse = round(RMSE_rf, 2)
knn_rmse = round(RMSE_knn, 2)
svr_rmse = round(RMSE_svr, 2)

# Define columns and index names
columns = [
    "linear regression",
    "xgboost",
    "adaboost",
    "decision tree",
    "random forest",
    "KNN",
    "SVR",
]
index = ["cross-val-score (r2)", "r2", "r2_adj", "RMSE"]

# Create the DataFrame
df = pd.DataFrame(columns=columns, index=index)


def add_values_to_df(df, model_name, values):
    """
    Adds values to the DataFrame for a specific model.

    Parameters:
    df (pd.DataFrame): The DataFrame to update.
    model_name (str): The column name corresponding to the model.
    values (dict): A dictionary with index names as keys and the corresponding values to add.

    Example:
    values = {
        'cross-val-score (r2)': 0.85,
        'r2': 0.87,
        'r2_adj': 0.86,
        'RMSE': 0.25
    }
    """
    for index_name, value in values.items():
        if index_name in df.index and model_name in df.columns:
            df.at[index_name, model_name] = value
    return df


# Example usage
linear_values = {
    "cross-val-score (r2)": linear_cvs,
    "r2": linear_r2,
    "r2_adj": linear_adj,
    "RMSE": linear_rmse,
}
xgb_values = {
    "cross-val-score (r2)": xgb_cvs,
    "r2": xgb_r2,
    "r2_adj": xgb_adj,
    "RMSE": xgb_rmse,
}
ada_values = {
    "cross-val-score (r2)": ada_cvs,
    "r2": ada_r2,
    "r2_adj": ada_adj,
    "RMSE": ada_rmse,
}
dt_values = {
    "cross-val-score (r2)": dt_cvs,
    "r2": dt_r2,
    "r2_adj": dt_adj,
    "RMSE": dt_rmse,
}
rf_values = {
    "cross-val-score (r2)": rf_cvs,
    "r2": rf_r2,
    "r2_adj": rf_adj,
    "RMSE": rf_rmse,
}
knn_values = {
    "cross-val-score (r2)": knn_cvs,
    "r2": knn_r2,
    "r2_adj": knn_adj,
    "RMSE": knn_rmse,
}
svr_values = {
    "cross-val-score (r2)": svr_cvs,
    "r2": svr_r2,
    "r2_adj": svr_adj,
    "RMSE": svr_rmse,
}

df = add_values_to_df(df, "linear regression", linear_values)
df = add_values_to_df(df, "xgboost", xgb_values)
df = add_values_to_df(df, "adaboost", ada_values)
df = add_values_to_df(df, "decision tree", dt_values)
df = add_values_to_df(df, "random forest", rf_values)
df = add_values_to_df(df, "KNN", knn_values)
df = add_values_to_df(df, "SVR", svr_values)

#display df for table comparison
print(df)

# graphs
fig, ax = plt.subplots(3, 3, figsize=(15, 13))
plt.subplots_adjust(hspace=0.4, wspace=0.4)
y_col = "Amine Ext Linker (%)"
ax[0, 0].scatter(y_test, y_pred1)
ax[0, 0].set_xlabel(f"Actual {y_col}")
ax[0, 0].set_ylabel(f"Predicted {y_col}")
ax[0, 0].set_title("Linear Regression")

ax[2, 0].scatter(y_test, y_pred2)
ax[2, 0].set_xlabel(f"Actual {y_col}")
ax[2, 0].set_ylabel(f"Predicted {y_col}")
ax[2, 0].set_title("Xgb")

ax[0, 1].scatter(y_test, y_pred3)
ax[0, 1].set_xlabel(f"Actual {y_col}")
ax[0, 1].set_ylabel(f"Predicted {y_col}")
ax[0, 1].set_title("AdaBoost")

ax[0, 2].scatter(y_test, y_pred4)
ax[0, 2].set_xlabel(f"Actual {y_col}")
ax[0, 2].set_ylabel(f"Predicted {y_col}")
ax[0, 2].set_title("Decision Tree")

ax[1, 0].scatter(y_test, y_pred5)
ax[1, 0].set_xlabel(f"Actual {y_col}")
ax[1, 0].set_ylabel(f"Predicted {y_col}")
ax[1, 0].set_title("Random Forest")

ax[1, 1].scatter(y_test, y_pred6)
ax[1, 1].set_xlabel(f"Actual {y_col}")
ax[1, 1].set_ylabel(f"Predicted {y_col}")
ax[1, 1].set_title("KNN")

ax[1, 2].scatter(y_test, y_pred7)
ax[1, 2].set_xlabel(f"Actual {y_col}")
ax[1, 2].set_ylabel(f"Predicted {y_col}")
ax[1, 2].set_title("SVM")

# choose final model and graphs
################################
# linear regression = y_pred1, regressor, linear_rmse
# XgBoost regressor = y_pred2, xgb, xgb_rmse
# AdaBoost regressor = y_pred3, adaregr, adaregr_rmse
# Decision Tree regressor = y_pred4, regressor_DT, dt_rmse
# Random Forest regressor = y_pred5, regr_rf, rf_rmse
# KNN regressor = y_pred6, neigh, knn_rmse
# SVM regressor = y_pred7, svr_regr0, svr_rmse
################################

# final model - choose based on previous code
final_model = y_pred1
model_name = "Linear Regressor"
MSE_model = linear_rmse
feat_importances = pd.Series(xgb.feature_importances_, index=X.columns)

print(
    "Mean absolute error:  ", round(mean_absolute_error(y_test, final_model), 3)
)  # measure of how far the predictions were from the actual output
print(
    "Root Mean Squared Error:  ", round(root_mean_squared_error(y_test, final_model), 3)
)  # average of the squared of the difference
print(
    "Explained variance score:  ",
    round(explained_variance_score(y_test, final_model), 3),
)  # explanation of how well the model predicts (similar to R^2 but does not offset due to goodness of fit)
print(f"The R^2 value for {model_name} is :", round((r2_score(y_test, final_model)), 3))
print("The maximum error is:   ", round(max_error(y_test, final_model), 3))


plt.scatter(y_test, final_model)
plt.xlabel(f"Actual {y_col}")
plt.ylabel(f"Predicted {y_col}")
plt.title(f"{model_name}")
plt.show()

feat_importances_sorted = feat_importances.sort_values(ascending=True)
feat_importances_sorted.plot(kind="barh")
plt.title("Most important features")
plt.show()

# save chosen model
# Regressor
# Xgb
# Adaregr
# Regressor_dt
# Regr_rf
# Neigh
# Svr_regr


with open('linear_regressor.pkl', 'wb') as file:
    pickle.dump(regressor, file)
