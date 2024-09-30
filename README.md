# House-Price-Analysis
Predicts prices of a house and analyses this price based on several features of the house in multiple ways.

## Project Overview
This repository contains a machine learning project for predicting house prices using various models such as Decision Trees, Random Forest and XGBoost. The goal is to provide accurate price predictions based on a variety of features related to houses, and to compare the performance of these models with and without hyperparameter tuning. Additionally, this project incorporates feature importance analysis, clustering, and model explainability techniques like graphs and plots to improve interpretability.

## Features and Highlights

**Exploratory Data Analysis (EDA)** : Initial data exploration to understand the distribution of variables and correlations between them.<br/><br/>
**Feature Engineering** : Creation of new features and transformation of existing ones to improve model performance.<br/><br/>
**Model Building** : Implementation of Decision Tree, Random Forest models and XGBoost.<br/><br/>
**Performance Comparison** : Evaluation of model performance before and after hyperparameter tuning.<br/><br/>
**Model Explainability** : Use of SHAP values and clustering techniques to explain model predictions.<br/><br/>
**Interactive Visualizations** : Advanced visualizations to interpret feature importance and model performance.<br/>

## Data

The dataset used in this project contains various features about houses, including location, number of rooms, size, age, and more. Each feature is used to predict the price of the house.

**Features** :<br/>

1) Income of people in that area<br/>
2) Age of the house<br/>
3) Number of rooms<br/>
4) Number of bedrooms<br/>
5) Location (categorized or encoded)<br/>
6) Population in that area<br/>


## Models and Techniques Used

1) **Decision Tree** : A basic model where hyperparameter tuning is applied to improve performance.
2) **Random Forest** : An ensemble model combining multiple decision trees. It’s evaluated with and without tuning to compare improvements in accuracy and stability.
3) **XGBoost** : Similar to Random Forest model but with extra accuracy increasing parameters like regularization and gradient boosting which allows accuracy to increase after every subsequent technique used on it.
4) **Feature Importance Analysis** : Plotting graphs showing how much important is one variable for the prediction of the house price. Also new features are added combining already present features.
5) **Clustering** : Grouping similar data points to enhance model performance and identify patterns in house pricing.
6) **Seaborn plots** : More advanced plots containing more than just simple features like density estimates, heatmaps which help in overall analysis of the model and the house price itself.


### Importing Libraries

```ruby
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,cross_val_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
```

### Loading Data

```ruby
house_info = pd.read_csv('/Users/rakshitrane/Downloads/USA_Housing.csv')
```

### Plots

```ruby
sns.pairplot(house_info)
numeric_df = house_info.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True)
```
Creating pairplots between every two combinations from the features available. Then after collecting only the column which has numeric columns are selected to show the heatmap between the features.

### Creating new variables

```ruby
house_info['Price_per_sqft'] = house_info['Price'] / house_info['Area Population']
house_info['Avg Area Income * Avg Area Number of Rooms'] = house_info['Avg. Area Income'] * house_info['Avg. Area Number of Rooms']
```

### Creating scatter plot from linear regression 

```ruby
X = house_info[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]

y = house_info['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

a = LinearRegression()
a.fit(X_train,y_train)

predictions = a.predict(X_test)
plt.scatter(y_test,predictions)
```
In this we simply fix the y and x axis and train the model through linear regression and then we take all the predicted values from the model and create a scatter plot.

## Evaluation Metrics

The models are evaluated using various metrics to ensure accuracy and interpretability:

1) Mean Absolute Error (MAE)
2) Root Mean Squared Error (RMSE)
3) R-squared Score
4) Feature Importance using SHAP values

### Performance of Random Forest and Decision Tree

```ruby
# Use the best parameters to initialize the Decision Tree model
best_dt_model = DecisionTreeRegressor(random_state=42, 
                                       max_depth=dt_grid_search.best_params_['max_depth'],
                                       min_samples_split=dt_grid_search.best_params_['min_samples_split'],
                                       min_samples_leaf=dt_grid_search.best_params_['min_samples_leaf'])

# Train the optimized Decision Tree model
best_dt_model.fit(X_train, y_train)

# Predict on the test set
y_pred = best_dt_model.predict(X_test)

# Calculate MSE and R² for the optimized Decision Tree
mse_tree = mean_squared_error(y_test, y_pred)
r2_tree = r2_score(y_test, y_pred)

print(f"Optimized Decision Tree - Mean Squared Error: {mse_tree}")
print(f"Optimized Decision Tree - R² Score: {r2_tree}")

# Initialize Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)

# Train Random Forest model
random_forest_model.fit(X_train, y_train)

# Predict on the test set
y_pred_forest = random_forest_model.predict(X_test)

# Calculate MSE and R² for Random Forest
mse_forest = mean_squared_error(y_test, y_pred_forest)
r2_forest = r2_score(y_test, y_pred_forest)

print(f"Optimized Random Forest - Mean Squared Error: {mse_forest}")
print(f"Optimized Random Forest - R² Score: {r2_forest}")
```
We simply perform hyperparameter tuning and decide the best parameters for each model. Grid Search is a brute-force technique where a set of possible hyperparameter values are predefined, and the model is trained and evaluated for every combination. It ensures all combinations are considered and so whichever parameters work the best is stored. There are multiple ways of doing so, RandomSearch is one technique but it may miss out the best combination. We define the number of iterations to be made. Then we use those parameters and trian the model. We print out the MSE and R^2 scores and see that Random Forest is doing better.

### XGBoost

```ruby
# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# Initialize XGBRegressor
xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Use RandomizedSearchCV for hyperparameter tuning
xgb_random_search = RandomizedSearchCV(estimator=xgboost_model, param_distributions=param_grid,
                                       n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

xgb_random_search.fit(X_train, y_train)

# Get the best parameters
print(xgb_random_search.best_params_)

# Evaluate the tuned model
y_pred_xgb_tuned = xgb_random_search.best_estimator_.predict(X_test)
mse_xgb_tuned = mean_squared_error(y_test, y_pred_xgb_tuned)
r2_xgb_tuned = r2_score(y_test, y_pred_xgb_tuned)

print(f"Optimized XGBoost - Mean Squared Error: {mse_xgb_tuned}")
print(f"Optimized XGBoost - R² Score: {r2_xgb_tuned}")
```
As explained in the previous section, we do the same thing with XGBoost and notice XGBoost has better MSE and R^2 score.

### Feature Importance

```ruby
# Feature Importance Visualization, shows which features' importance is the most while predicting house prices
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)

importances = rf_model.feature_importances_
features = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title('Feature Importance')
plt.show()
```
We first train a RandomForest model. A decision tree makes predictions by splitting the dataset into smaller subsets based on feature values. Each split or subset is chosen so as to best reduce the error. The model creates multiple decision trees by training on different bootstrapped (similar to sampling with replacement) subsets of the data. At each split within each decision tree, the algorithm selects a feature and a threshold value that gives the least error. The importance of a feature is based on how much it improves the model's accuracy at each split. The code after this just outputs certain easy to code graphs for more visualization like clustering graph, and betwene house price and 2 features. Also cumulative distribution function is also outputted.

## Results

**Initial Performance** : The Decision Tree model performed with a baseline accuracy, while the Random Forest showed better performance due to its ensemble nature.

**Tuned Models** : Hyperparameter tuning significantly improved the accuracy of both models, with Random Forest consistently outperforming Decision Tree.

**XGBoost with hyperparameter tuning** : Hyperparameter tuning of XGBoost increased the R^2 score of the model even more outperforming Decision Tree and Random Forest.

**Visualization** : The error distribution and how far off the predicted values are from the true values can be seen. Also through the feature importance graph, we see that the avg. income of people in that area is the most important feature in predicting the house prices in this case. We can also see heatmap and cumulative distribution of probability as a function of house prices.





