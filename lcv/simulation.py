import numpy as np
from scipy import stats
from scipy.special import erfinv, beta


class simulation:
    def __init__(
        self,
        dim=20,
        coef=0.3,
        noise=True,
        signif_vars=1,
        hetero_value=1,
        asym_value=0.6,
        t_degree=4,
    ):
        self.dim = dim
        self.coef = coef
        self.noise = noise
        self.vars = signif_vars
        self.hetero_value = hetero_value
        self.asym_value = asym_value
        self.t_degree = t_degree

    def change_dim(self, new_dim):
        self.dim = new_dim
        return self

    def homoscedastic(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        if self.noise:
            y = np.random.normal(self.coef * X[:, 0], scale=1, size=n)
        else:
            y = np.random.normal(
                self.coef * np.mean(X[:, np.arange(0, self.vars)], axis=1),
                scale=1,
                size=n,
            )
        self.X, self.y = X, y
        self.kind = "homoscedastic"

    def t_residuals(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=-1.5, high=1.5, size=(n, self.dim))
        if self.noise:
            y = (self.coef * X[:, 0]) + np.random.standard_t(self.t_degree, size=n)
        else:
            y = (
                self.coef * np.mean(X[:, np.arange(0, self.vars)], axis=1)
            ) + np.random.standard_t(self.t_degree, size=n)
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
        if self.noise:
            y = (self.coef * X[:, 0]) + np.random.gamma(
                1 + (self.asym_value * np.abs(X[:, 0])),
                1 + (self.asym_value * np.abs(X[:, 0])),
                size=n,
            )
        else:
            y = (
                self.coef * np.mean(X[:, np.arange(0, self.vars)], axis=1)
            ) + np.random.gamma(
                1
                + (
                    self.asym_value
                    * np.abs(np.mean(X[:, np.arange(0, self.vars)], axis=1))
                ),
                1
                + (
                    self.asym_value
                    * np.abs(np.mean(X[:, np.arange(0, self.vars)], axis=1))
                ),
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
        if self.noise:
            y = np.random.normal(
                self.coef * X[:, 0],
                scale=np.sqrt(self.hetero_value + self.coef * np.abs(X[:, 0])),
                size=n,
            )
        else:
            y = np.random.normal(
                self.coef * np.mean(X[:, np.arange(0, self.vars)], axis=1),
                scale=np.sqrt(
                    self.hetero_value
                    + self.coef * np.abs(np.mean(X[:, np.arange(0, self.vars)], axis=1))
                ),
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
        if self.noise:
            y = np.random.normal(
                1,
                scale=np.sqrt(self.hetero_value + self.coef * np.abs(X[:, 0])),
                size=n,
            )
        else:
            y = np.random.normal(
                1,
                scale=np.sqrt(
                    self.hetero_value
                    + self.coef * np.abs(np.mean(X[:, np.arange(0, self.vars)], axis=1))
                ),
                size=n,
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
            if self.noise:
                y_mat[i, :] = np.random.normal(self.coef * X_grid[i], scale=1, size=B)
            else:
                y_mat[i, :] = np.random.normal(
                    self.coef * np.mean(X_grid[i, np.arange(0, self.vars)]),
                    scale=1,
                    size=B,
                )
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
            if self.noise:
                y_mat[i, :] = np.random.normal(
                    self.coef * X_grid[i],
                    scale=np.sqrt(self.hetero_value + self.coef * np.abs(X_grid[i])),
                    size=B,
                )
            else:
                y_mat[i, :] = np.random.normal(
                    self.coef * np.mean(X_grid[i, np.arange(0, self.vars)]),
                    scale=np.sqrt(
                        self.hetero_value
                        + self.coef
                        * np.abs(np.mean(X_grid[i, np.arange(0, self.vars)]))
                    ),
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
            if self.noise:
                y_mat[i, :] = np.random.normal(
                    1,
                    scale=np.sqrt(self.hetero_value + self.coef * np.abs(X_grid[i])),
                    size=B,
                )
            else:
                y_mat[i, :] = np.random.normal(
                    1,
                    scale=np.sqrt(
                        self.hetero_value
                        + self.coef
                        * np.abs(np.mean(X_grid[i, np.arange(0, self.vars)]))
                    ),
                    size=B,
                )
        return y_mat

    def asymmetric_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            if self.noise:
                y_mat[i, :] = (self.coef * X_grid[i]) + np.random.gamma(
                    1 + (self.asym_value * np.abs(X_grid[i])),
                    1 + (self.asym_value * np.abs(X_grid[i])),
                    size=B,
                )
            else:
                y_mat[i, :] = (
                    self.coef * np.mean(X_grid[i, np.arange(0, self.vars)])
                ) + np.random.gamma(
                    1
                    + (
                        self.asym_value
                        * np.abs(np.mean(X_grid[i, np.arange(0, self.vars)]))
                    ),
                    1
                    + (
                        self.asym_value
                        * np.abs(np.mean(X_grid[i, np.arange(0, self.vars)]))
                    ),
                    size=B,
                )
        return y_mat

    def t_residuals_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            if self.noise:
                y_mat[i, :] = (self.coef * X_grid[i]) + np.random.standard_t(
                    self.t_degree, size=B
                )
            else:
                y_mat[i, :] = (
                    self.coef * np.mean(X_grid[i, np.arange(0, self.vars)])
                ) + np.random.standard_t(self.t_degree, size=B)
        return y_mat

    def predict(self, X_pred, significance=0.05):
        quantile_kind = getattr(self, self.kind + "_quantiles")
        return quantile_kind(X_pred[:, 0], sig=significance)

    def fit(self, X, y, significance=0.05):
        return self.predict(X, significance)


class toy_simulation:
    def __init__(
        self, coef=0.3, hetero_value=1, asym_value=0.6, alpha=1, beta=1, xlim=1, rate=1
    ):
        self.coef = coef
        self.hetero_value = hetero_value
        self.asym_value = asym_value
        self.xlim = xlim
        self.alpha = alpha
        self.beta = beta
        self.rate = rate

    def bimodal(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=0, high=self.xlim, size=(n, 1))
        y = (X[:, 0] ** 2) + (
            0.5
            * np.random.normal(
                -X[:, 0], np.sqrt((self.hetero_value ** 2) - (X[:, 0] ** 2)), size=n
            )
            + 0.5
            * np.random.normal(
                X[:, 0], np.sqrt((self.hetero_value ** 2) - (X[:, 0] ** 2)), size=n
            )
        )

        self.X, self.y = X, y
        self.kind = "bimodal"

    def bimodal_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            y_mat[i, :] = np.random.normal(
                X_grid[i] ** 2,
                scale=np.sqrt((0.5) * ((self.hetero_value ** 2) - (X_grid[i] ** 2))),
                size=B,
            )
        return y_mat

    def bimodal_oracle(self, X_grid, B=1000, sig=0.1):
        band = np.zeros((X_grid.shape[0], 2))
        for i in range(X_grid.shape[0]):
            sample = np.random.normal(
                X_grid[i] ** 2,
                scale=np.sqrt((0.5) * ((self.hetero_value ** 2) - (X_grid[i] ** 2))),
                size=B,
            )
            band[i, 1], band[i, 0] = (
                np.quantile(sample, (1 - (sig / 2))),
                np.quantile(sample, (sig / 2)),
            )

        return band

    def bimodal_laplace(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=0, high=self.xlim, size=(n, 1))
        y = (X[:, 0] ** 2) + (
            0.5
            * np.random.laplace(
                -X[:, 0],
                np.sqrt(((self.hetero_value ** 2) - (X[:, 0] ** 2)) / 2),
                size=n,
            )
            + 0.5
            * np.random.laplace(
                X[:, 0],
                np.sqrt(((self.hetero_value ** 2) - (X[:, 0] ** 2)) / 2),
                size=n,
            )
        )

        self.X, self.y = X, y
        self.kind = "bimodal_laplace"

    def bimodal_laplace_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        for i in range(X_grid.shape[0]):
            b_2 = ((self.hetero_value ** 2) - (X_grid[i] ** 2)) / 2
            y_mat[i, :] = np.random.laplace(
                X_grid[i] ** 2,
                scale=np.sqrt((0.5 * b_2) + (0.0625 * (b_2 ** 2))),
                size=B,
            )
        return y_mat

    def bimodal_laplace_oracle(self, X_grid, B=1000, sig=0.1):
        band = np.zeros((X_grid.shape[0], 2))
        for i in range(X_grid.shape[0]):
            b_2 = ((self.hetero_value ** 2) - (X_grid[i] ** 2)) / 2
            sample = np.random.laplace(
                X_grid[i] ** 2,
                scale=np.sqrt((0.5 * b_2) + (0.0625 * b_2 ** 2)),
                size=B,
            )
            band[i, 1], band[i, 0] = (
                np.quantile(sample, (1 - (sig / 2))),
                np.quantile(sample, (sig / 2)),
            )

        return band

    def splitted(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=0, high=self.xlim, size=(n, 1))
        mad_norm = np.sqrt(self.hetero_value * 2) * erfinv(1 / 2)

        y = (
            X[:, 0] ** 2
            + ((X[:, 0] <= self.xlim / 2) * np.random.laplace(0, mad_norm, size=n))
            + (
                (X[:, 0] > self.xlim / 2)
                * np.random.normal(0, np.sqrt(self.hetero_value), size=n)
            )
        )

        self.X, self.y = X, y
        self.kind = "splitted"

    def splitted_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        mad_norm = np.sqrt(self.hetero_value * 2) * erfinv(1 / 2)

        for i in range(X_grid.shape[0]):
            if X_grid[i] <= (self.xlim / 2):
                y_mat[i, :] = np.random.laplace(X_grid[i] ** 2, scale=mad_norm, size=B,)
            else:
                y_mat[i, :] = np.random.normal(
                    X_grid[i] ** 2, scale=np.sqrt(self.hetero_value), size=B,
                )
        return y_mat

    def splitted_oracle(self, X_grid, sig=0.1, B=1000):
        mad_norm = np.sqrt(self.hetero_value * 2) * erfinv(1 / 2)
        band = np.zeros((X_grid.shape[0], 2))

        for i in range(X_grid.shape[0]):
            if X_grid[i] <= self.xlim / 2:
                sample = np.random.laplace(X_grid[i] ** 2, scale=mad_norm, size=B,)
            else:
                sample = np.random.normal(
                    X_grid[i] ** 2, scale=np.sqrt(self.hetero_value), size=B,
                )
            band[i, 1], band[i, 0] = (
                np.quantile(sample, (1 - (sig / 2))),
                np.quantile(sample, (sig / 2)),
            )
        return band

    def splitted_beta(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=0, high=self.xlim, size=(n, 1))
        mad_beta = (2 * (self.alpha ** self.alpha) * (self.beta ** self.beta)) / (
            beta(self.alpha, self.beta)
            * ((self.alpha + self.beta) ** (self.alpha + self.beta + 1))
        )

        y = (
            X[:, 0] ** 2
            + ((X[:, 0] <= self.xlim / 2) * np.random.laplace(0, mad_beta, size=n))
            + (
                (X[:, 0] > self.xlim / 2)
                * (
                    np.random.beta(self.alpha, self.beta, size=n)
                    - (self.alpha / (self.alpha + self.beta))
                )
            )
        )

        self.X, self.y = X, y
        self.kind = "splitted_beta"

    def splitted_beta_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        mad_beta = (2 * (self.alpha ** self.alpha) * (self.beta ** self.beta)) / (
            beta(self.alpha, self.beta)
            * ((self.alpha + self.beta) ** (self.alpha + self.beta + 1))
        )

        for i in range(X_grid.shape[0]):
            if X_grid[i] <= (self.xlim / 2):
                y_mat[i, :] = np.random.laplace(X_grid[i] ** 2, scale=mad_beta, size=B)
            else:
                y_mat[i, :] = X_grid[i] ** 2 + (
                    np.random.beta(self.alpha, self.beta, size=B,)
                    - (self.alpha / (self.alpha + self.beta))
                )
        return y_mat

    def splitted_beta_oracle(self, X_grid, sig=0.1, B=1000):
        mad_beta = (2 * (self.alpha ** self.alpha) * (self.beta ** self.beta)) / (
            beta(self.alpha, self.beta)
            * ((self.alpha + self.beta) ** (self.alpha + self.beta + 1))
        )
        band = np.zeros((X_grid.shape[0], 2))

        for i in range(X_grid.shape[0]):
            if X_grid[i] <= self.xlim / 2:
                sample = np.random.laplace(X_grid[i] ** 2, scale=mad_beta, size=B,)
            else:
                sample = X_grid[i] ** 2 + (
                    np.random.beta(self.alpha, self.beta, size=B)
                    - (self.alpha / (self.alpha + self.beta))
                )
            band[i, 1], band[i, 0] = (
                np.quantile(sample, (1 - (sig / 2))),
                np.quantile(sample, (sig / 2)),
            )
        return band

    def splitted_exp(self, n, random_seed=1250):
        np.random.seed(random_seed)
        X = np.random.uniform(low=0, high=self.xlim, size=(n, 1))
        mad_exp = 2 / (np.exp(1)*self.rate)
        y = (
            X[:, 0] ** 2
            + ((X[:, 0] <= self.xlim / 2) * np.random.laplace(0, mad_exp, size=n))
            + (
                (X[:, 0] > self.xlim / 2)
                * (np.random.exponential(1 / self.rate, size=n) - (1 / self.rate))
            )
        )

        self.X, self.y = X, y
        self.kind = "splitted_exp"

    def splitted_exp_r(self, X_grid, B=1000):
        y_mat = np.zeros((X_grid.shape[0], B))
        mad_exp = 2 / (np.exp(1)*self.rate)

        for i in range(X_grid.shape[0]):
            if X_grid[i] <= (self.xlim / 2):
                y_mat[i, :] = np.random.laplace(X_grid[i] ** 2, scale=mad_exp, size=B)
            else:
                y_mat[i, :] = X_grid[i] ** 2 + (
                    np.random.exponential(1 / self.rate, size=B) - (1 / self.rate)
                )
        return y_mat

    def splitted_exp_oracle(self, X_grid, sig=0.1, B=1000):
        mad_beta = 2 / (np.exp(1)*self.rate)
        band = np.zeros((X_grid.shape[0], 2))

        for i in range(X_grid.shape[0]):
            if X_grid[i] <= self.xlim / 2:
                sample = np.random.laplace(X_grid[i] ** 2, scale=mad_beta, size=B,)
            else:
                sample = X_grid[i] ** 2 + (
                    np.random.exponential(1 / self.rate, size=B) - (1 / self.rate)
                )

            band[i, 1], band[i, 0] = (
                np.quantile(sample, (1 - (sig / 2))),
                np.quantile(sample, (sig / 2)),
            )
        return band
