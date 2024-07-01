import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

wine_data = pd.read_csv('wine.csv')
# 1.
# Split data into training and test sets
test_data = pd.concat([wine_data[wine_data['target'] == i].sample(n=20)
                       for i in range(3)])
train_data = wine_data.drop(test_data.index)

# Save to CSV files
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

# 2.
train_data = pd.read_csv('train.csv')

# Calculate the prior probabilities
num_instances = len(train_data)
prior_probs = [len(train_data[train_data['target'] == i]) / num_instances for i in range(3)]

# Calculate the mean and standard deviation of each feature for each class
means = train_data.groupby('target').mean().values[:, :]
stds = train_data.groupby('target').std().values[:, :]

# Define the likelihood function for each feature for each class
likelihoods = np.zeros((3, 13), dtype=object)
for i in range(3):
    for j in range(13):
        likelihoods[i, j] = norm(loc = means[i, j], scale = stds[i, j]).pdf

# Load the test data
test_data = pd.read_csv('test.csv')

# Calculate the posterior probabilities for each instance in the test data
num_correct = 0
for i, row in test_data.iterrows():
    true_label = row[0]
    probs = np.zeros(3)
    for j in range(3):
        prob = np.log(prior_probs[j])
        for k in range(1, 14):
            likelihood = likelihoods[j, k-1]
            prob += np.log(likelihood(row[k]))
        probs[j] = prob
    predicted_label = np.argmax(probs)
    if predicted_label == true_label:
        num_correct += 1

# Calculate the accuracy rate
accuracy = num_correct / len(test_data)
print('Accuracy rate:', accuracy)

# 3. plot the visualized result of testing data 
# Load the test data
test_data = pd.read_csv('test.csv')

# Perform PCA on the test data
pca = PCA(n_components = 3)
transformed = pca.fit_transform(test_data.values[:, 1:])

# # Plot the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green']
labels = ['Type 0', 'Type 1', 'Type 2']
for i in range(3):
    indices = np.where(test_data['target'] == i)[0]
    ax.scatter(transformed[indices, 0], transformed[indices, 1], transformed[indices, 2], color = colors[i], label = labels[i])
    
plt.legend()
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('PCA Visualization of Wine Data')
plt.show()