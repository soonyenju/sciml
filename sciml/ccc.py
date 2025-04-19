# https://rowannicholls.github.io/python/statistics/agreement/correlation_coefficients.html#lins-concordance-correlation-coefficient-ccc
# Lin LIK (1989). “A concordance correlation coefficient to evaluate reproducibility”. Biometrics. 45 (1):255-268.
import numpy as np
import pandas as pd

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Remove NaNs
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator


# y_true = [3, -0.5, 2, 7, np.NaN]
# y_pred = [2.5, 0.0, 2, 8, 3]
# ccc = concordance_correlation_coefficient(y_true, y_pred)
# print(ccc)