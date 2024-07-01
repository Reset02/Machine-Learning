import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor


def function(X,mu,s):
    phi = np.exp(-((X - mu)**2)/(2*s**2))
    return phi

##################把兩個資料整合後拆分
# Load the exercise and calories datasets
exercise_df = pd.read_csv('exercise.csv')
calories_df = pd.read_csv('calories.csv')

# Merge the two datasets based on the "User_ID" column
merged_df = pd.merge(exercise_df, calories_df, on='User_ID')
merged_df['Gender'] =  merged_df['Gender'].map({'male':1, 'female': 0})
# Shuffle the merged dataset randomly
shuffled_df = merged_df.sample(frac=1, random_state=42)

# Split the shuffled dataset into training, validation, and testing sets
train_size = int(0.7 * len(shuffled_df))
val_size = int(0.1 * len(shuffled_df))
test_size = len(shuffled_df) - train_size - val_size

train_data = shuffled_df[:train_size]
val_data = shuffled_df[train_size:train_size+val_size]
test_data = shuffled_df[train_size+val_size:]

# Reset the index of each dataset
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# Print the shapes of the resulting dataframes
print("Training set shape:", train_data.shape)
print("Validation set shape:", val_data.shape)
print("Testing set shape:", test_data.shape)


##################第一題
def MLR(train_data, val_data, test_data):
    # Extract the features and target variable from the training data
    X_train = train_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
    y_train = train_data['Calories'].values
    

    # Add a column of ones to the feature matrix for the intercept term
    X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)

    # Extract the features and target variable from the validation data
    X_val = val_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
    y_val = val_data['Calories'].values

    # Add a column of ones to the feature matrix for the intercept term
    X_val = np.concatenate([np.ones((X_val.shape[0], 1)), X_val], axis=1)

    # Compute the maximum likelihood estimate of the weights
    w_mle = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    # Compute the predicted values using the maximum likelihood estimate
    y_pred_mle = X_val @ w_mle

    # Compute the mean squared error for each prediction
    mse_mle = np.mean((y_val - y_pred_mle)**2)


    # Extract the features and target variable from the testing data
    X_test = test_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
    y_test = test_data['Calories'].values

    # Add a column of ones to the feature matrix for the intercept term
    X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)

    # Compute the predicted values using the maximum likelihood estimate
    y_pred_mle_test = X_test @ w_mle

    # Compute the mean squared error for the test set for each method
    mse_mle_test = np.mean((y_test - y_pred_mle_test)**2)

    # Choose the best model based on the validation error

    return mse_mle_test, w_mle, y_pred_mle_test, mse_mle

mse_mle_test, w_mle, y_pred_mle_test, mse_mle = MLR(train_data, val_data, test_data)
print("Mean Squared Error (MLE):", mse_mle_test)

##################第二題
def BLR(train_data, val_data, test_data):
    # Extract the features and target variable from the training data
    X_train = train_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
    y_train = train_data['Calories'].values
    # Extract the features and target variable from the validation data
    X_val = val_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
    y_val = val_data['Calories'].values

    # Compute the hyperparameters for the prior distribution over w
    alpha = 1.0
    beta = 1.0

    # Compute the posterior distribution over w
    S_0 = alpha * np.eye(X_train.shape[1])
    m_0 = np.zeros(X_train.shape[1])
    S_n = np.linalg.inv(np.linalg.inv(S_0) + beta * X_train.T @ X_train)
    m_n = S_n @ (np.linalg.inv(S_0) @ m_0 + beta * X_train.T @ y_train)

    # Compute the predicted values using the estimated parameters
    y_pred_val = X_val @ m_n

    # Compute the mean squared error for each prediction on validation set
    mse_val = np.mean((y_val - y_pred_val)**2)

    # Extract the features and target variable from the testing data
    X_test = test_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
    y_test = test_data['Calories'].values

    # Compute the predicted values using the estimated parameters
    y_pred_test = X_test @ m_n

    # Compute the mean squared error for each prediction on testing set
    mse_test = np.mean((y_test - y_pred_test)**2)

    return m_n, mse_val, mse_test, y_pred_test
m_n, mse_val, mse_test, y_pred_test = BLR(train_data, val_data, test_data)
print("Mean Squared Error (Bayesian Linear Regression):", mse_test)

##################第四題
X_train = train_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
y_train = train_data['Calories'].values
X_test = test_data[['Gender','Duration', 'Heart_Rate', 'Body_Temp', 'Weight', 'Height', 'Age']].values
y_test = test_data['Calories'].values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Gradient Boosting Regression model
gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = gb_reg.predict(X_test)

# Compute the mean squared error
mse = np.mean((y_test - y_pred) ** 2)

print("Best MSE(Gradient Boosting Regression):", mse)



##################第三題畫圖
# Extract the features and target variable from the data
X = test_data["Duration"].values.reshape(-1, 1)
y = test_data["Calories"].values
# X = merged_df["Duration"].values.reshape(-1, 1)
# y = merged_df["Calories"].values

# Maximum Likelihood Linear Regression
w_ml = np.linalg.inv(X.T @ X) @ X.T @ y
y_ml = X @ w_ml

# Define the prior hyperparameters
alpha = 1
beta = 1
sigma = 1

# Define the design matrix
phi = np.hstack((np.ones_like(X), X))

# Calculate the posterior covariance matrix
S_inv = alpha * np.eye(2) + beta * phi.T @ phi / sigma**2
S = np.linalg.inv(S_inv)

# Calculate the posterior mean of the coefficients
m = beta * S @ phi.T @ y / sigma**2

# Calculate the predicted values and standard deviation
y_pred = phi @ m
std_pred = np.sqrt(sigma**2 + np.sum(phi @ S * phi, axis=1))

# Plot the actual data and the best-fit line
plt.scatter(X, y,s=10, c='b',alpha=0.7, label="Observations")
plt.plot(X, X @ w_ml, color='r', linestyle="--", label="Best-fit(MLR)")
plt.plot(X, y_pred, color='black', label="Best-fit(Bayesian)")
plt.xlabel("Duration(min)")
plt.ylabel("Calories Burnt")
plt.legend()
plt.show()


