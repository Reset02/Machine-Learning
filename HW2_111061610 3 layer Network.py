import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import Data_test
import Data_train
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.svm import SVC

NUM_COMPONENTS = 2
pca = PCA(n_components = NUM_COMPONENTS)

# example of dataset create
class Dataset(Dataset):

    # data loading
    def __init__(self, data_save_name, data):

        self.data = data
        if self.data == 'train' :
            self.data_save_name = os.path.join(data_save_name, 'Data_train')
        else:
            self.data_save_name = os.path.join(data_save_name, 'Data_test')

        self.data_label = ['Carambula', 'Lychee', 'Pear']

        if self.data == 'train':
            self.data_num = 490
        else:
            self.data_num = 166

        self.images = []
        self.labels = []

        for label, type_name in enumerate(self.data_label):
            for i in range(self.data_num):
                name = os.path.join(self.data_save_name, type_name, '{}_{}_{}.png'.format(type_name, self.data, i))
                # 使用 plt.read讀取圖片
                image = np.array(plt.imread(name), dtype=np.float32)
                #儲存圖片和標籤 x and y
                self.images.append(image)
                self.labels.append(label)

        self.images = np.array(self.images)
        self.PCA_images = self.PCA_features()

    def PCA_features(self):
        #將每個圖像的像素值展開為一維數組
        self.images_reshape = self.images.reshape(self.images.shape[0], -1)
        if self.data == 'train':
            return pca.fit_transform(self.images_reshape)
        else:
            return pca.transform(self.images_reshape)
    # Dataloader 需要實現__len__和__getitem__方法。其中__len__方法返回Dataset中的樣本數量，__getitem__方法根據索引(index)返回指定的樣本。    
    def __getitem__(self, index):
        return self.PCA_images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)

# Create datasets.
# 注意注意 Dataset(r'C:\Users\lulu3\Desktop\HW2_111061610', 'train') 這是我的資料路徑
# 助教跑程式時，請改成助教資料存放路徑 FruitDataset('資料存放路徑', 'train') ，如果在C巢 要加 r, 變成 FruitDataset(r'資料存放路徑', 'train')
train_data = Dataset(r'C:\Users\lulu3\Desktop\HW2_111061610', 'train') 
test_data = Dataset(r'C:\Users\lulu3\Desktop\HW2_111061610', 'test')

# Create data loaders.
BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = True)

# 3個inputs和3個outputs，一個隱藏層有512個神經元
device = "cpu"
NUM_CLASSES = 3

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1 = 512, hidden_size2 = 256, output_size = NUM_CLASSES):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        self.weights1 = np.random.randn(self.input_size, self.hidden_size1)
        self.bias1 = np.zeros((1, self.hidden_size1))
        
        self.weights2 = np.random.randn(self.hidden_size1, self.hidden_size2)
        self.bias2 = np.zeros((1, self.hidden_size2))
        
        self.weights3 = np.random.randn(self.hidden_size2, self.output_size)
        self.bias3 = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.weights1) + self.bias1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        self.a3 = self.softmax(self.z3)
        return self.a3
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        # calculate gradients for output layer
        delta3 = self.a3 - y
        d_weights3 = (1 / m) * np.dot(self.a2.T, delta3)
        d_bias3 = (1 / m) * np.sum(delta3, axis=0, keepdims=True)
        
        # calculate gradients for hidden layer 2
        delta2 = np.dot(delta3, self.weights3.T) * self.a2 * (1 - self.a2)
        d_weights2 = (1 / m) * np.dot(self.a1.T, delta2)
        d_bias2 = (1 / m) * np.sum(delta2, axis=0, keepdims=True)
        
        # calculate gradients for hidden layer 1
        delta1 = np.dot(delta2, self.weights2.T) * self.a1 * (1 - self.a1)
        d_weights1 = (1 / m) * np.dot(X.T, delta1)
        d_bias1 = (1 / m) * np.sum(delta1, axis=0)
        
        # update weights and biases
        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1
        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2
        self.weights3 -= learning_rate * d_weights3
        self.bias3 -= learning_rate * d_bias3

model = NeuralNetwork(NUM_COMPONENTS)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    return loss

def train(dataloader, model):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    epoch_loss = []
    correct = []

    for inputs, true_outputs in tqdm(dataloader):
        # 先把資料變成numpy型式
        inputs, true_outputs = inputs.numpy(), true_outputs.numpy()

        # Compute prediction error
        pred = model.forward(inputs)
        loss = cross_entropy_loss(pred, np.eye(3)[true_outputs])
        # Backpropagation
        model.backward(inputs, np.eye(3)[true_outputs], 1e-3)
        
        epoch_loss.append(loss)
        epoch_loss_all = np.sum(epoch_loss)
        pred = pred.argmax(axis=1)
        correct.append(np.equal(pred, true_outputs).sum())
        correct_all = np.sum(correct)

    avg_epoch_loss = epoch_loss_all / num_batches
    avg_acc = correct_all / size

    return avg_epoch_loss, avg_acc

def test(dataloader, model):
    num_batches = len(dataloader)
    size = len(dataloader.dataset)
    epoch_loss = []
    correct = []

    for inputs, true_outputs in tqdm(dataloader):
        # 先把資料變成numpy型式
        inputs, true_outputs = inputs.numpy(), true_outputs.numpy()

        pred = model.forward(inputs)
        epoch_loss.append(cross_entropy_loss(pred, np.eye(3)[true_outputs]))
        epoch_loss_all = np.sum(epoch_loss)
        pred = pred.argmax(axis=1)
        correct.append(np.equal(pred, true_outputs).sum())
        correct_all = np.sum(correct)

    avg_epoch_loss = epoch_loss_all / num_batches
    avg_acc = correct_all / size

    return avg_epoch_loss, avg_acc

def plot_decision_regions(x, y, model):
    cmap = ListedColormap(['r', 'g', 'b'])
    # Create a meshgrid of points to classify
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    X_grid = np.c_[xx1.ravel(), xx2.ravel()]

    # Create an SVC model and fit it to the data
    svc = SVC(gamma='auto')
    svc.fit(x, y)

    # Predict the labels for all points on the grid
    pred = svc.predict(X_grid)

    # Plot the decision regions and data points
    Z = pred.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap = cmap)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap = cmap, edgecolor='black')
    plt.title('Two layer SVC decision regions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

# Call this function after training the model
x_test, y_test = next(iter(test_dataloader))
# 先把資料變成numpy型式
x_test, y_test = x_test.numpy(), y_test.numpy()
plot_decision_regions(x_test, y_test, model)


epochs = 20
train_loss_all = []
for epoch in range(epochs):
    train_loss, train_acc = train(train_dataloader, model)
    test_loss, test_acc = test(test_dataloader, model)
    print(f"Epoch {epoch + 1:2d}: Loss = {train_loss:.4f} Acc = {train_acc:.2f} Test_Loss = {test_loss:.4f} Test_Acc = {test_acc:.2f}")
    train_loss_all.append(train_loss)
print("Done!")

plt.plot(train_loss_all)
plt.title("Three layer Network Training Loss ")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
