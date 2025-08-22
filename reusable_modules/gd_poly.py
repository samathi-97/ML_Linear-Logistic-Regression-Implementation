import numpy as np

def compute_cost(X, Y, beta):
    m = len(Y)
    y_hat = X.dot(beta)
    cost = (1 / (2 * m)) * np.sum((y_hat - Y) ** 2)
    return cost

def gradient_descent(X, Y, beta, lr, n_iter):
    m = len(Y)
    cost_histroy = []

    for i in range(n_iter):
        y_hat = X.dot(beta)
        gradients = (1 / m) * X.T.dot(y_hat - Y)

        beta = beta - lr * gradients

        cost = compute_cost(X, Y, beta)
        cost_histroy.append(cost)

        if i % 100 == 0:
            print(f"Iteration : {i}: Cost : {cost:.4f}")

    return cost_histroy, beta