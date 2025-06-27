import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import pinvh
from sklearn.linear_model import LinearRegression

class FisherPHATEAnalyzer:
    def __init__(self, phate_operator, gene_expression):
        self.phate_op = phate_operator
        self.gene_expression = gene_expression
        self.phate_coords = None
        self.metrics = None

    def compute_fisher_metric(self):
        """
        Compute the Fisher Information metric at each PHATE embedding point.
        """
        P_t = self.phate_op._calculate_potential(t="auto")
        assert self.gene_expression.shape[0] == P_t.shape[0], (
            f"Mismatch: gene_expression has {self.gene_expression.shape[0]} rows but PHATE was trained on {P_t.shape[0]}"
        )

        self.phate_coords = self.phate_op.fit_transform(self.gene_expression)
        n, g = self.gene_expression.shape

        metrics = [np.zeros((2, 2)) for _ in range(n)]

        for i in range(n):
            w = P_t[i]
            mu = w @ self.gene_expression

            G = (self.gene_expression - mu).T * w
            Sigma = G @ (self.gene_expression - mu)
            Sigma_inv = pinvh(Sigma + 1e-8 * np.eye(g))

            reg = LinearRegression(fit_intercept=True)
            reg.fit(self.phate_coords, self.gene_expression, sample_weight=w)
            J = reg.coef_.T

            metrics[i] = J @ Sigma_inv @ J.T

        self.metrics = metrics

    def plot_fisher_quiver(self, step=5, scale=60):
        """
        Plot the Fisher metric field as quiver arrows over the PHATE embedding.
        """
        if self.phate_coords is None or self.metrics is None:
            raise RuntimeError("Must call compute_fisher_metric() before plotting.")

        idx = np.arange(0, len(self.phate_coords), step)
        coords_sampled = self.phate_coords[idx]
        metrics_sampled = [self.metrics[i] for i in idx]

        U, V = [], []
        for G in metrics_sampled:
            vals, vecs = np.linalg.eigh(G)
            direction = vecs[:, 1]  # principal eigenvector
            magnitude = np.sqrt(vals[1]) if vals[1] > 0 else 0
            U.append(direction[0] * magnitude)
            V.append(direction[1] * magnitude)

        U = np.array(U)
        V = np.array(V)

        plt.figure(figsize=(9, 9))
        plt.quiver(coords_sampled[:, 0], coords_sampled[:, 1],
                   U, V, color='black', angles='xy', scale_units='xy', scale=scale)
        plt.scatter(self.phate_coords[:, 0], self.phate_coords[:, 1], s=2, alpha=0.3, color='steelblue')
        plt.title('Fisher Vector Field (Quiver Style)')
        plt.xlabel('PHATE1')
        plt.ylabel('PHATE2')
        plt.grid(True)
        plt.axis('equal')
        plt.show()
