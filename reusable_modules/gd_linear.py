import numpy as np

def compute_cost(X, y, beta_0, beta_1):
    """
    Cost Function (Mean Squared Error)

    -  J = ( 1/2m) Σ(ŷᵢ - yᵢ)²
    """
    m = len(y)
    y_hat = beta_0 + beta_1 * X
    cost = (1/(2*m)) * np.sum((y_hat - y) ** 2)
    return cost

def compute_gradients(X, y, beta_0, beta_1):
    """
    Gradients (Partial Derivatives)

        - ∂J/∂β₀ = ( /m) Σ(ŷᵢ - yᵢ)
        - ∂J/∂β₁ = ( /m) Σ(ŷᵢ - yᵢ)xᵢ

    """
    m = len(y)
    y_hat = beta_0 + beta_1 * X
    grad_beta_0 = (1/m) * np.sum(y_hat - y)
    grad_beta_1 = (1/m) * np.sum((y_hat - y) * X)
    return grad_beta_0, grad_beta_1


def gradient_descent(X, y, learning_rate, num_itr):
    """
    Gradient Descent Algorithm

    - β := β - α∇J

            - β₀ := β₀ - α . ∂J/∂β₀
            - β₁ := β₁ - α . ∂J/∂β₁

    """
    beta_0 = np.random.randn()
    beta_1 = np.random.randn()

    print(f"beta values:\n  β₀ = {beta_0}\n  β₁ = {beta_1}")
    cost_history = []

    for i in range(num_itr):
        grad_beta_0, grad_beta_1 = compute_gradients(X, y, beta_0, beta_1)
        beta_0 =  beta_0 - learning_rate * grad_beta_0
        beta_1 = beta_1 - learning_rate * grad_beta_1

        cost = compute_cost(X, y, beta_0, beta_1)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return beta_0, beta_1, cost_history