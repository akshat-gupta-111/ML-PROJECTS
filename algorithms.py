import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LinearRegression:
    def __init__(self):
        self.weight = None
        self.intercept = None

    def compute_cost(self, X, y, w, b):
        X = np.array(X)
        y = np.array(y)
        m = X.shape[0]
        n = len(w)
        cost = 0.0
        for i in range(m):
            f_wb_i = np.dot(X[i], w) + b
            cost += (f_wb_i - y[i]) ** 2
        cost = cost / (2 * m)
        return cost

    def compute_gradient(self, X, y, w, b):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        dj_dw_j = np.zeros((n,))
        dj_db = 0.0

        for i in range(m):
            f_wb = np.dot(X[i], w) + b
            for j in range(n):
                dj_dw_j[j] += (f_wb - y[i]) * X[i][j]
            dj_db += f_wb - y[i]
        dj_dw_j /= m
        dj_db /= m
        return dj_dw_j, dj_db

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters):
        X = np.array(X)
        y = np.array(y)
        w = np.array(w_in, copy=True)
        b = b_in
        j_history = []

        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradient(X, y, w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            if i < 100000:
                j_history.append(self.compute_cost(X, y, w, b))
        return w, b, j_history

    def train(self, X, y, alpha=0.01, epoch=1000):
        X = np.array(X)
        y = np.array(y)
        w_in = np.zeros(X.shape[1])
        b_in = 0.0
        self.weight, self.intercept, _ = self.gradient_descent(X, y, w_in, b_in, alpha, epoch)

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weight) + self.intercept

    def plot_cost_curve(self, X, y, alpha=0.01, epoch=1000):
        X = np.array(X)
        y = np.array(y)
        w_in = np.zeros(X.shape[1])
        b_in = 0.0
        _, _, j_history = self.gradient_descent(X, y, w_in, b_in, alpha, epoch)
        plt.plot(j_history)
        plt.title("Cost vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()

class LogisticRegression:
    def __init__(self):
        self.weight = None
        self.intercept = None

    def compute_cost_logistic(self, X, y, w, b):
        X = np.array(X)
        y = np.array(y)
        m = X.shape[0]
        cost = 0
        for i in range(m):
            z = np.dot(X[i], w) + b
            f_wb = sigmoid(z)
            loss = -y[i] * (np.log(f_wb)) - ((1 - y[i]) * np.log(1 - f_wb))
            cost += loss
        return cost / m

    def compute_gradient_logistic(self, X, y, w, b):
        X = np.array(X)
        y = np.array(y)
        m, n = X.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.0

        for i in range(m):
            err = (sigmoid(np.dot(X[i], w) + b) - y[i])
            for j in range(n):
                dj_dw[j] += err * X[i, j]
            dj_db += err
        dj_dw /= m
        dj_db /= m
        return dj_dw, dj_db

    def gradient_descent(self, X, y, w_in, b_in, alpha, num_iters):
        X = np.array(X)
        y = np.array(y)
        w = np.array(w_in, copy=True)
        b = b_in
        j_history = []

        for i in range(num_iters):
            dj_dw, dj_db = self.compute_gradient_logistic(X, y, w, b)
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            if i < 100000:
                j_history.append(self.compute_cost_logistic(X, y, w, b))
        return w, b, j_history

    def train(self, X, y, alpha=0.01, epoch=1000):
        X = np.array(X)
        y = np.array(y)
        w_in = np.zeros(X.shape[1])
        b_in = 0.0
        self.weight, self.intercept, _ = self.gradient_descent(X, y, w_in, b_in, alpha, epoch)

    def predict(self, X):
        X = np.array(X)
        return sigmoid(np.dot(X, self.weight) + self.intercept)

    def plot_cost_curve(self, X, y, alpha=0.01, epoch=1000):
        X = np.array(X)
        y = np.array(y)
        w_in = np.zeros(X.shape[1])
        b_in = 0.0
        _, _, j_history = self.gradient_descent(X, y, w_in, b_in, alpha, epoch)
        plt.plot(j_history)
        plt.title("Cost vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.show()