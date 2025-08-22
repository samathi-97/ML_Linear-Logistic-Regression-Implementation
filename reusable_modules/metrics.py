import numpy as np

def MSE(y, y_hat):
   """
    Mean Squared Error (MSE)
    e_i = y_i - ŷ_i
    MSE = (1/m) * Σ (e_i ** 2)
   """
   e_i = y - y_hat
   return np.mean(e_i ** 2)

def MAE(y, y_hat):
    """
    Mean Absolute Error (MAE)
    e_i = y_i - ŷ_i
    MAE = (1/m) * Σ |e_i|
    """
    e_i = y - y_hat
    return np.mean(np.abs(e_i))

def RMSE(y, y_hat):
    """
    - Root Mean Squared Error (RMSE)
    - e_i = y_i - ŷ_i
    - RMSE = sqrt( (1/m) * Σ (e_i ** 2) )
         = sqrt(MSE)
    """
    e_i = y - y_hat
    return np.sqrt(np.mean(e_i ** 2))

def R2_Score(y, y_hat):
    """
    - Coefficient of Determination (R²)

    - e_i = y_i - ŷ_i
    - ȳ   = mean(y)
    - std_i = y_i - ȳ

    - R² = 1 - ( Σ (e_i ** 2) / Σ (std_i ** 2) )
    """
    y_mean = np.mean(y)
    e_i = y - y_hat
    std_i = y - y_mean

    return 1 - (np.sum(e_i ** 2) / np.sum(std_i ** 2))
