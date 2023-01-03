import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split


def split(X, y, test_size=0.4, calibrate=True, random_seed=1250):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed
    )
    if calibrate:
        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train, y_train, test_size=0.25, random_state=random_seed
        )
        return {
            "X_train": X_train,
            "X_calib": X_calib,
            "X_test": X_test,
            "y_train": y_train,
            "y_calib": y_calib,
            "y_test": y_test,
        }
    else:
        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }


class simulation:
    def __init__(self, dim=20, coef=0.3, hetero_value=1, asym_value=0.6, t_degree=4):
        self.dim = dim
        self.coef = coef
        self.hetero_value = hetero_value
        self.asym_value = asym_value
        self.t_degree = t_degree

    def change_dim(self, new_dim):
        self.dim = new_dim
        return self

    def homoscedastic(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        y = np.random.normal(self.coef * X[:, 0], scale=1, size=n)
        self.X, self.y = X, y
        self.kind = "homoscedastic"

    def t_residuals(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        y = (self.coef * X[:, 0]) + np.random.standard_t(self.t_degree, size=n)
        self.X, self.y = X, y
        self.kind = "t_residuals"

    def bimodal(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        f_x = (X[:, 0] - 1) ** (2) * (X[:, 0] + 1)
        g_x = 2 * np.sqrt((X[:, 0] >= -0.5) * (X[:, 0] + 0.5))
        sigma_x = np.sqrt(0.25 + np.abs(X[:, 0]))
        y = 0.5 * np.random.normal(
            f_x - g_x, scale=sigma_x, size=n
        ) + 0.5 * np.random.normal(f_x + g_x, scale=sigma_x, size=n)
        self.X, self.y = X, y
        self.kind = "bimodal"

    def asymmetric(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        y = (self.coef * X[:, 0]) + np.random.gamma(
            1 + (self.asym_value * np.abs(X[:, 0])),
            1 + (self.asym_value * np.abs(X[:, 0])),
            size=n,
        )
        self.X, self.y = X, y
        self.kind = "asymmetric"

    def set_heterosc_coef(self, value):
        self.hetero_value = value

    def set_asym_coef(self, value):
        self.asym_value = value

    def set_t_degree(self, value):
        self.t_degree = value

    def heteroscedastic(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        y = np.random.normal(
            self.coef * X[:, 0],
            scale=np.sqrt(self.hetero_value + self.coef * np.abs(X[:, 0])),
            size=n,
        )
        self.X, self.y = X, y
        self.kind = "heteroscedastic"

    def heteroscedastic_latent(self, n, random_seed=1250):
        np.random.seed(random_seed)
        # generating the latent variables first then the uniforms
        X = np.column_stack(
            (
                np.random.binomial(1, p=0.2, size=n),
                np.random.uniform(low=-5, high=5, size=(n, self.dim + 1)),
            )
        )

        eps1 = np.random.normal(0, scale=1, size=n)
        eps2 = np.random.normal(0, scale=0.1, size=n)

        # generating responses according to lantet variables
        y = (
            3 * eps2
            + (
                (X[:, 0] == 1)
                * ((-0.2 * eps1 * (X[:, 1] - 5)) - (X[:, 1] * (X[:, 1] > 0)))
            )
            + (
                (X[:, 0] == 0)
                * ((0.2 * eps1 * (X[:, 1] + 5)) + (X[:, 1] * (X[:, 1] > 0)))
            )
        )
        self.X, self.y = X[:, 1:], y
        self.kind = "heteroscedastic_latent"

    def non_cor_heteroscedastic(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        y = np.random.normal(
            1, scale=np.sqrt(self.hetero_value + self.coef * np.abs(X[:, 0])), size=n
        )
        self.X, self.y = X, y
        self.kind = "non_cor_heteroscedastic"

    def homoscedastic_quantiles(self, X_grid, sig):
        q = [sig / 2, 1 - sig / 2]
        lower = stats.norm.ppf(
            np.repeat(q[0], X_grid.shape[0]), loc=self.coef * X_grid, scale=1
        )
        upper = stats.norm.ppf(
            np.repeat(q[1], X_grid.shape[0]), loc=self.coef * X_grid, scale=1
        )
        interval = np.vstack((lower, upper)).T
        return interval

    def homoscedastic_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = np.random.normal(self.coef * X_grid[i], scale=1, size=B)
        return y_mat

    def heteroscedastic_quantiles(self, X_grid, sig):
        q = [sig / 2, 1 - sig / 2]
        lower = stats.norm.ppf(
            np.repeat(q[0], X_grid.shape[0]),
            loc=self.coef * X_grid,
            scale=np.sqrt(1 + self.coef * np.abs(X_grid)),
        )
        upper = stats.norm.ppf(
            np.repeat(q[1], X_grid.shape[0]),
            loc=self.coef * X_grid,
            scale=np.sqrt(1 + self.coef * np.abs(X_grid)),
        )
        interval = np.vstack((lower, upper)).T
        return interval

    def heteroscedastic_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = np.random.normal(
                self.coef * X_grid[i],
                scale=np.sqrt(self.hetero_value + self.coef * np.abs(X_grid[i])),
                size=B,
            )
        return y_mat

    def heteroscedastic_latent_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            # indicator variables
            # for X1
            x_large = (X_grid[i, 1] > 0) + 0
            # for X0
            x_0, x_1 = (X_grid[i, 0] == 0) + 0, (X_grid[i, 0] == 1) + 0

            y_mat[i, :] = np.random.normal(
                x_large * ((x_0 - x_1) * X_grid[i, 1]),
                scale=np.sqrt(
                    (9 * (0.01)) + (0.04 * ((X_grid[i, 1] + ((x_0 - x_1) * 5)) ** 2))
                ),
                size=B,
            )
        return y_mat

    def non_cor_heteroscedastic_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = np.random.normal(
                1,
                scale=np.sqrt(self.hetero_value + self.coef * np.abs(X_grid[i])),
                size=B,
            )
        return y_mat

    def asymmetric_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = (self.coef * X_grid[i]) + np.random.gamma(
                1 + (self.asym_value * np.abs(X_grid[i])),
                1 + (self.asym_value * np.abs(X_grid[i])),
                size=B,
            )
        return y_mat

    def t_residuals_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = (self.coef * X_grid[i]) + np.random.standard_t(
                self.t_degree, size=B
            )
        return y_mat

    def predict(self, X_pred, significance=0.05):
        quantile_kind = getattr(self, self.kind + "_quantiles")
        return quantile_kind(X_pred[:, 0], sig=significance)

    def fit(self, X, y, significance=0.05):
        return self.predict(X, significance)
