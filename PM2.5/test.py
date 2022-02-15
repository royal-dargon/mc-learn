import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston


class Adagrad():
    def __init__(self):
        self.data_x = data_x
        self.data_y = data_y
        self.feature = data_features

    def data_clearning(self):
        rows = self.data_x.shape[0]
        data_mean = np.mean(self.data_x, axis=0)
        data_std = np.std(self.data_x, axis=0)
        self.x_std = (self.data_x - data_mean) / data_std
        I = np.zeros(rows) + 1.
        self.x_b_std = np.insert(self.x_std, 0, I, axis=1)
        return self.x_b_std

    def dLoss(self, theta):
        return self.x_b_std.T.dot(self.x_b_std.dot(theta) - self.data_y) * 2 / self.x_b_std.shape[0]

    def Loss(self, theta):
        return (self.x_b_std.dot(theta) - self.data_y).T.dot(self.x_b_std.dot(theta) - self.data_y) / \
               self.x_b_std.shape[0]

    def gradient_descent(self):
        theta = np.zeros(self.x_b_std.shape[1])
        eta = 0.1
        epsilon = 1e-4
        loss = []
        dL_history = []
        while True:
            dL = self.dLoss(theta)
            dL_history.append(dL.tolist())
            loss.append(np.sum(np.abs(self.Loss(theta))))
            last_theta = theta
            theta = theta - eta / np.sqrt(np.sum(np.array(dL_history) ** 2, axis=0)) * dL
            if np.abs(np.sum(np.abs(self.Loss(theta)) - np.sum(np.abs(self.Loss(last_theta))))) < epsilon:
                break
        return theta, loss

    def plot_loss(self, loss):
        plt.style.use("ggplot")
        plt.title("Adagrad_Loss")
        plt.xlabel("times")
        plt.plot(loss)

    def predict(self, theta, x_std):
        return x_std.dot(theta.T)

    def start(self):
        self.data_clearning()
        theta, loss = self.gradient_descent()
        self.plot_loss(loss)
        predict = self.predict(theta, self.x_b_std)
        return theta, loss, predict


if __name__ == '__main__':
    boston = load_boston()
    data_x = boston.data
    data_y = boston.target
    data_features = boston.feature_names
    GD = Adagrad()
    theta, loss, predict = GD.start()
    print("参数为：\n", theta)
    print("loss为：\n", loss[-1])
    print("target：\n", data_y[:5])
    print("target_predict：\n", predict[:5])