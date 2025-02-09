import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.covariance = None

    def fit(self, features: np.array):
        """

        features: (n_samples, n_features)
        """

        """
        Step 1: standardization
        """

        """
        Step 2: covariance matrix calculation
        A covariance matrix is computed to understand the relationships between features. For
        n features, this matrix is of size n x n, and each entry Cov(xi, xj) represents the covariance between features
        xi and xj.
        """

        m = len(features[0])
        self.covariance = (1/(m-1))*np.matmul(np.transpose(features), features)

        """
        Interpretation of the covariance matrix
        Diagonal elements Cii: variances of individual features
        Off-diagonal elements Cij: covariances between pairs of features
        - Positive covariance indicates that the two features increase together
        - Negative covariance indicates an inverse relationship
        - Zero covariance implies no linear relationship
        """


        """
        Step 3: eigenvectors and eigenvalues
        """

        eigen_vals, eigen_vecs = np.linalg.eig(self.covariance)

        # Step 4: Choosing principal components

        # Step 5: Transforming the data


# https://towardsdatascience.com/a-step-by-step-implementation-of-principal-component-analysis-5520cc6cd598

features = np.array([[1, 2], [3, 4], [5, 6]])
f_mean = np.mean(features, axis=0)
features_norm = features - f_mean
m = features.shape[0]
cov = (1/(m-1))*np.matmul(np.transpose(features_norm), features_norm)
eigen_vals, eigen_vecs = np.linalg.eig(cov)
print(eigen_vecs)

indices = np.arange(0, len(eigen_vals), 1)
indices = ([x for _,x in sorted(zip(eigen_vals, indices))])[::-1]
eig_val = eigen_vals[indices]
eig_vec = eigen_vecs[:,indices]
print("Sorted Eigen vectors ", eig_vec)
print("Sorted Eigen values ", eig_val, "\n")
