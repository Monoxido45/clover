# simulation class to perform basic tests
import numpy as np
from scipy import stats

class simulation:
    def __init__(self, dim = 20, coef = 0.3):
        self.dim = dim
        self.coef = coef
    
    def change_dim(self, new_dim):
        self.dim = new_dim
        return self
        
    def homoscedastic(self, n, random_seed = 1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low = -1.5, high = 1.5, size = (n, self.dim))
        y = np.random.normal(self.coef*X[:, 0], scale = 1, size = n)
        self.X, self.y = X, y
        self.kind = "homoscedastic"
    
    def bimodal(self, n, random_seed = 1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low = -1.5, high = 1.5, size = (n, self.dim))
        f_x = (X[:, 0] - 1)**(2)*(X[:, 0] + 1)
        g_x = 2*int(X[:, 0] >= -0.5)*np.sqrt(X[:,0] + 0.5)
        sigma_x = np.sqrt(0.25 + np.abs(X[:0]))
        y = (0.5*np.random.normal(f_x - g_x, scale = sigma_x, size = n) + 
             0.5*np.random.normal(f_x + g_x, scale = sigma_x, size = n))
        self.X, self.y = X, y
        self.kind = "bimodal"
    
    def heteroscedastic(self, n, random_seed = 1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low = -1.5, high = 1.5, size = (n, self.dim))
        y = np.random.normal(self.coef*X[:, 0], scale = np.sqrt(1 + self.coef*np.abs(X[:, 0])), size = n)
        self.X, self.y = X, y
        self.kind = "heteroscedastic"
    
    def homoscedastic_quantiles(self, X_grid, sig):
        q = [sig/2, 1 - sig/2]
        lower = stats.norm.ppf(np.repeat(q[0], X_grid.shape[0]), loc = self.coef*X_grid, scale = 1)
        upper = stats.norm.ppf(np.repeat(q[1], X_grid.shape[0]), loc = self.coef*X_grid, scale = 1)
        interval = np.vstack((lower, upper)).T 
        return interval
    
    def homoscedastic_r(self, X_grid, B = 1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = np.random.normal(self.coef*X_grid[i], scale = 1, size = B)
        return y_mat
    
    def heteroscedastic_quantiles(self, X_grid, sig):
        q = [sig/2, 1 - sig/2]
        lower = stats.norm.ppf(np.repeat(q[0], X_grid.shape[0]), loc = self.coef*X_grid, 
                               scale = np.sqrt(1 + self.coef*np.abs(X_grid)))
        upper = stats.norm.ppf(np.repeat(q[1], X_grid.shape[0]), loc = self.coef*X_grid, 
                               scale = np.sqrt(1 + self.coef*np.abs(X_grid)))
        interval = np.vstack((lower, upper)).T 
        return interval
    
    def heteroscedastic_r(self, X_grid, B = 1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = np.random.normal(self.coef*X_grid[i],
                                           scale = np.sqrt(1 + self.coef*np.abs(X_grid[i])),
                                           size = B)
        return y_mat

    def predict(self, X_pred, significance = 0.05):
        quantile_kind = getattr(self, self.kind + "_quantiles")
        return quantile_kind(X_pred[:, 0], sig = significance)
    
    def fit(self, X, y, significance = 0.05):
        return self.predict(X, significance)

