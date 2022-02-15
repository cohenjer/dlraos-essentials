import numpy as np
import math
from scipy.interpolate import BSpline


def count_support_onesparse(input, ref):
    """
    Computes the percentage of same elements in two lists or 1d numpy arrays.
    """
    # deal with 1d row-array
    if isinstance(ref, np.ndarray):
        ref = ref.flatten()
    if isinstance(input, np.ndarray):
        input = input.flatten()
    count = 0
    for i in ref:
        if i in input:
            count+=1
    return count/len(ref)*100

def gen_mix(dims, snr=20, distr='Gaussian'):
    '''
    Generates simulated dataset for experiments according to the one-sparse mixed sparse coding model.

    Parameters
    ----------
    dims : list of length 4
        [m, n, d, r]

    snr : integer
        signal to noise ratio, controls noise level

    distr : string
        Default is 'Gaussian', but 'Uniform' also works. 'Decreasing' is Gaussian D,B and Uniform X with artificially decreasing weights for X.

    Returns
    -------
    Y : nd numpy array
        noised data

    Ytrue : nd numpy array
        noiseless data

    D : nd numpy array
        dictionary normalized columnswise in l2 norm

    B : nd numpy array
        mixing matrix

    X : nd numpy array
        true unknown sparse coefficients

    S : 1d numpy array
        support of X

    sig : float
        noise variance used in practice

    '''
    m, n, d, r = dims

    k=1

    if distr == 'Gaussian':
        D = np.random.randn(n, d)
        B = np.random.randn(m, r)
    elif distr == 'Uniform':
        D = np.random.rand(n, d)
        B = np.random.rand(m, r)
    else:
        print('Distribution not supported')

    for i in range(d):
        D[:, i] = D[:, i]/np.linalg.norm(D[:, i])

    # X k-sparse columnwise generation
    X = np.zeros([d, r])
    S = []
    for i in range(r):
        pos = np.random.permutation(d)[0:k]
        if distr == 'Uniform':
            X[pos, i] = np.random.rand(k)
        elif distr == 'Gaussian':
            X[pos, i] = np.random.randn(k)
        else:
            print('Distribution not supported')
        S.append(pos)

    # Formatting to np 1d array
    S = np.transpose(np.array(S)).flatten()

    # Noise and SNR
    Ytrue = D@X@B.T
    noise = np.random.rand(n, m)

    spower = np.linalg.norm(Ytrue, 'fro')**2
    npower = np.linalg.norm(noise, 'fro')**2
    old_snr = np.log10(spower/npower)
    sig = 10**((old_snr - snr/10)/2)
    noise = sig*noise  # scaling to get SNR right

    Y = Ytrue + noise

    return Y, Ytrue, D, B, X, S, sig