import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ast import literal_eval
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
# Read csv file
df = pd.read_csv('input1.csv')

# Parse string list
for col in df.columns[2:-1]:
    df[col] = df[col].apply(lambda x: literal_eval(x) if pd.notnull(x) else [0, 0, 0, 0, 0])

# Expand characteristic column
feature_cols = df.columns[2:-1]
features = df[feature_cols].apply(lambda x: np.concatenate(x.values).tolist(), axis=1)

# Create a new DataFrame to save the expanded features.
features_df = pd.DataFrame(features.tolist(), index=df.index)

# Add the target row to the new DataFrame.
features_df['target1'] = df['target1']
features_df.insert(0, 'group1', df['group1'])
features_df.insert(1, 'group2', df['group2'])

# Print the processed DataFrame
print(features_df.head())

# Separate features and labels
X = features_df.drop(columns=['target1']).values
y1 = features_df['target1'].values

# Carry out standardization treatment
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data set
X_train, X_test, y1_train, y1_test = train_test_split(X_scaled, y1, test_size=0.2, random_state=173)

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(20, 10),],
    'batch_size': [16,],
    'alpha': [0.1],
    'learning_rate': ['constant',],
    'max_iter': [1500,],
    'learning_rate_init': [0.001,]
}

# Define parameter grid
nn_reg = MLPRegressor(random_state=1)

# instantiation GridSearchCV
grid_search = GridSearchCV(nn_reg, param_grid, cv=4, scoring='r2', n_jobs=12)

# Perform a grid search
grid_search.fit(X_train, y1_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Evaluate the best model
y_pred_train = best_model.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y1_train, y_pred_train))
r2_1_train = r2_score(y1_train, y_pred_train)
y_pred_test = best_model.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y1_test, y_pred_test))
r2 = r2_score(y1_test, y_pred_test)
print(y1_test)
print(y1_train)
print(y_pred_test)
print(y_pred_train)
print("Best parameters:", best_params)
print("Train R2 score:", r2_1_train)
print("Train RMSE:", rmse_train)
print("Test RMSE:", rmse_test)
print("Test R2 score:", r2)

# Draw a scatter plot between the predicted value and the true value.
plt.scatter(y1_test, y_pred_test, color='blue', label='Test Data')
plt.scatter(y1_train, y_pred_train, color='red', label='Train Data')
plt.plot([-4.5, -3.2], [-4.5, -3.2], color='black', linewidth=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs. Predicted values (R2 score: {:.2f})'.format(r2))
plt.legend()
plt.show()

# Draw a loss function diagram
plt.plot(best_model.loss_curve_)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()

# SHAP value analysis
explainer = shap.KernelExplainer(best_model.predict, X_train[:100])
shap_values = explainer.shap_values(X_test[:100])

# Draw a SHAP value
shap.summary_plot(shap_values, X_test[:100], plot_type="bar")
shap.summary_plot(shap_values, X_test[:100])

