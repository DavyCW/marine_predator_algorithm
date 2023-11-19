import numpy as np

def levy(n, m, beta):
    num = np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2)  # Used for Numerator
    den = np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)  # Used for Denominator

    sigma_u = (num / den)**(1 / beta)  # Standard deviation

    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, 1, (n, m))

    z = u / np.abs(v)**(1 / beta)

    return z
