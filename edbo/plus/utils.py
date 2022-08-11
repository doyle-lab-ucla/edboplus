
import numpy as np


class EDBOStandardScaler:
    """
    Custom standard scaler for EDBO.
    """
    def __init__(self):
        pass

    def fit(self, x):
        self.mu  = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x-[self.mu])/[self.std]

    def inverse_transform(self, x):
        return x * [self.std] + [self.mu]

    def inverse_transform_var(self, x):
        return x * [self.std]
