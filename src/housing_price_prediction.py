import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Load the dataset
data = pd.read_csv("data/Housing.csv")

# Encode categorical features
data = pd.get_dummies(data, columns=['mainroad', 'guestroom', 'basement',
                                     'hotwaterheating', 'airconditioning',
                                     'prefarea', 'furnishingstatus'], drop_first=True)

# Define features and target. X is data except price, and Y is price.
X = data.drop("price", axis=1)
y = data["price"]

# Split the data. 80% training and 20% testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for model.
# Scaler standardizes the data so that all features are in similar range (helps model to learn faster)
# Polynomial features adds more complex features by squaring and combining original features. 
# Ridge regression is the type of model used. Slightly more complex than linear regression. Prevents model from going too extreme with any one feature.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', Ridge())
])

# Define hyperparameters to finetune the model. 
param_grid = {
    'poly__degree': [1, 2, 3],
    'model__alpha': [0.1, 1.0, 10.0]
}

# Perform grid search to try different combinations of degree and aplha (regularization of model)
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Make predictions with the best model
predictions = best_model.predict(X_test)

# Evaluate the model performance
mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)
print("MAE:", mae)
print("RMSE:", rmse)

# Visualize results
fig, ax = plt.subplots()
ax.scatter(y_test, predictions)
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted Prices")
ax.set_title("Actual vs Predicted Housing Prices")

# Format the plot (diagram x,y units)
def millions(x, pos):
    'The two args are the value and tick position'
    return f'{x*1e-6:,.1f}M'

formatter = ticker.FuncFormatter(millions)
ax.xaxis.set_major_formatter(formatter)
ax.yaxis.set_major_formatter(formatter)

plt.show()
