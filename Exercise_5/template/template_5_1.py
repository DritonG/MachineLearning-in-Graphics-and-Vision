import numpy as np
import utils

OUT_PATH = 'out/pca'

def main():
    """The main function of this script."""
    # -----------------------------
    # TODO: ADAPT FOR EXERCISE 5.1e
    # -----------------------------
    # HINT: you can used numpys advanced indexing feature to
    # select an appropriate subset of x_train / x_test
    x_train, y_train = utils.load_fashion_mnist('train')
    x_test, y_test = utils.load_fashion_mnist('test')

    mean, s, V = compute_pca(x_train.numpy())

    analyze_variance(s)
    create_reconstructions(mean, V, x_test.numpy())
    create_samples(mean, V, s)


def compute_pca(x):
    """Compute PCA of x and plot mean as well as first 5 principal
    components.

    Args:
        x (np.array): NxK numpy array containing N data points of dimensionality
            K as input to PCA.

    Returns:
        mu (np.array): mean of x
        s (np.array): singular values of the covariance matrix of x
        V (np.array): KxK array containing all K principal components of x as
            row vectors
    """
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5.1a)
    # 1. Compute the mean vector of x
    # 2. Compute the covariance matrix of x
    # 3. Use `np.linalg.svd` to compute the singular values s
    #    and the matrix of principal components V
    #
    # HINT 1: You can use matrix products to calculate
    #   the covariance matrix efficiently
    # HINT 2: Don't forget to normalize the covariance
    #   matrix by the number of samples N
    # -------------------------------------
    # print("DataSet: ", x)
    mu = np.mean(x)
    # substracting the mean from the original dataset to center around 0
    centered_X = x.T - mu
    # calculating the covariance matrix C = AA^t/(n-1) (Slide 12, PCA)
    cov = centered_X @ centered_X.T / (len(x)-1)
    # svd --> principal components "V" in descending order with respect to their eigenvalues
    u, s, V = np.linalg.svd(cov)
    utils.plot_principal_components(V[:5, :], OUT_PATH)
    #print(s)
    return mu, s, V


def analyze_variance(s):
    """Analyze the variance of the singular values of the PCA

    Args:
        s (np.array): singular values
    """
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5.1b)
    # -------------------------------------
    # 1. Normalize the vector of singluar values by its sum
    # 2. Apply np.cumsum to the normalized vector of the
    #    singular values to obtain the commulative distribution
    #    of singular values
    # 3. Determine how many entries you need to achieve
    #    50%, 90%, 95% and 99% of the variance and print the result
    # 4. Plot the commulative distribution distribution of the
    #    singular values
    #
    # HINT: you can use `plot_cummulative_distribution` to plot
    # the commulative distribution of singular values
    s_norm = s/sum(s)
    s_cumsum = np.cumsum(s_norm)
    print("Entries to achieve 50%:", np.where(s_cumsum >= 50/100)[0][0])
    print("Entries to achieve 90%:", np.where(s_cumsum >= 90/100)[0][0])
    print("Entries to achieve 95%:", np.where(s_cumsum >= 95/100)[0][0])
    print("Entries to achieve 99%:", np.where(s_cumsum >= 99/100)[0][0])

    utils.plot_cummulative_distribution(s_cumsum, OUT_PATH)

def create_reconstructions(mean, V, x, ncomp=5, nplots=5):
    """Apply PCA to test data, print mean squared  error and plot first
    `nplots` reconstructions.

    Args:
        mean (np.array): mean of PCA
        V (np.array): array containing principal components as row vectors
        x (np.array): test data
        ncomp (int, optional): number of principal components to use
        nplots (int, optional): number of principal components to plot
    """
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5.1c)
    # -------------------------------------
    # 1. Normalize the test data x by the mean vector
    # 2. Multiply the result from the right with V.T to project
    #    the normalized vector onto the principal components
    # 3. Set all coefficients to 0 execpt for the first ncomp
    #    components
    # 4. Project the coefficients back to image space by
    #    multiplying from the right with V  and add back the mean
    x_norm = x - mean
    x_proj = x_norm @ V.T
    x_coef = x_proj
    x_coef[:, ncomp:] = 0
    x_recon = x_coef @ V + mean
    # Mean Squared Error
    MSE = np.square(np.subtract(x, x_recon)).mean()
    print("Mean Squared Error: ", MSE)

    utils.plot_reconstructions(x[:nplots, :], x_recon[:nplots, :], OUT_PATH)
    # Compression ratio = uncompressed size / compressed size
    comp_ratio = x.shape[1]/ncomp
    print("Compression Ratio", comp_ratio)

def create_samples(mean, V, s, nsamples=5):
    """Use PCA to sample synthetic data points and plot them

    Args:
        mean (np.array): mean of PCA
        V (np.array): array containing principal components
        nsamples (int, optional): number of samples to draw
    """
    # -------------------------------------
    # TODO: ENTER CODE HERE (EXERCISE 5.1d)
    # 1. sample a normally distributed random
    #    vector of size s.size
    # 2. multiply  this vector with np.sqrt(s)
    # 3. multiply this vector from the left with V.T
    #    to project it back to image space and add
    #    the mean vector
    # 4. append sample to x_rnd
    #
    # -------------------------------------
    # List of samples
    x_rnd = []
    # Loop over the number of samples to draw
    for i in range(nsamples):
        rand_vec = np.random.normal(loc=mean, size=s.size)  # 1.
        rand_vec_proj = rand_vec * np.sqrt(s)               # 2.
        x_sample = V.T @ rand_vec_proj + mean               # 3.
        x_rnd.append(x_sample)                              # 4.
    utils.plot_samples(x_rnd, OUT_PATH)


if __name__ == '__main__':
    main()
