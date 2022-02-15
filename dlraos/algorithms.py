import numpy as np
from numpy.linalg import lstsq, solve
import warnings
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

import tensorly as tl
from tensorly.tenalg.proximal import hals_nnls
import copy
from scipy.optimize import linear_sum_assignment

def osmf_mpals(Y, rank, D, n_iter_max=100, init='random', tol=1e-8,
            verbose=0, return_errors=False,
            cvg_criterion='abs_rec_error', nonnegative=False,simplex=False, optimal_assignment=True):
    """One sparse Dictionary-based Matrix Factorization via Matching Pursuit Alternating (nonnegative) Least Squares

        minimize 1/2 \|Y - D@X@B.T\|_F^2 wrt X, B s.t. \|X_i\|_0\leq 1

        First estimate  A:= D@X without the dictionary constraint,
        Second estimate X from A
        Third estimate  B from D@X

    The subroutine `estimate A then X` can be optimal if noise level is low and D and B are well conditionned.

    Parameters
    ----------
    Y : numpy array
        input data matrix
    rank : int
        Number of components.
    D : ndarray
        Dictionary for the first mode. Should be normalized with l2 norm columnwise.
    simplex : boolean, default False
        decides if the columns of B are on the simplex or not.
        Not implemented yet.
    nonnegative : boolean, default False
        set to True is A and B should be nonnegative.
    optimal_assignment : boolean, default True
        choose if the same atom in D can be selected twice.
    n_iter_max : int, default 100
        Maximum number of outer iteration
    init : {'random', [DX, B]]}, optional
        Type of factor matrix initialization. If a list of factors is passed, it is directly used for initalization. See `initialize_factors`.
    tol : float, default: 1e-8
        Relative reconstruction error tolerance. The algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for the outer iterations, works if `tol` is not None. If 'rec_error', iterations stop at current iteration if :math:`(previous rec_error - current rec_error) < tol`. If 'abs_rec_error', iterations terminate when :math:`|previous rec_error - current rec_error| < tol`.

    Returns
    -------
    Decomposed Matrix : list of numpy arrays
        Estimated DMF factors [DX, B]

    X : numpy array
        The sparse coefficients of the first factor

    S : numpy array
        Support of X

    errors : list
        A list of reconstruction errors (sqrt and normalized) at each iteration.
    """

    # only need to initialize B
    if init=='random':
        B = np.random.randn(Y.shape[1],rank)
        X = np.random.randn(D.shape[1],rank)
        DX = D@X
    else:
        X = np.copy(init[0])
        DX = D@X
        B = np.copy(init[1])

    Ynorm = np.linalg.norm(Y)
    rec_errors = [np.linalg.norm(Y - DX@B.T)/Ynorm]

    # Precomputations
    DtY = D.T@Y
    DtD = D.T@D

    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)


        # A estimate
        BtB = B.T@B
        YB = Y@B
        if nonnegative:
            DX = hals_nnls(YB.T,BtB,V=DX.T, exact=False)[0].T
        else:
            DX = np.linalg.solve(BtB,YB.T).T

        # X estimate
        X = np.zeros(X.shape)
        cost = D.T@DX
        if optimal_assignment:
            # use scipy optimize
            Sx,_ = linear_sum_assignment(-cost)
        else:
            # take the maximum columnwise
            Sx = np.argmax(cost,axis=0)
        for j in range(rank):
            X[Sx[j],j]=1
        DX = D[:,Sx] # same than D@X

        # B least squares update 
        DXtDX = DtD[Sx,:][:,Sx] #X.T@DtD@X
        DXY = DtY[Sx,:]
        if iteration==0:
            Bt=np.copy(B.T)
        DXtDX_reg = DXtDX+(1e-16)*np.max(DXtDX.flatten())*np.eye(DXtDX.shape[0])

        #if simplex:
        #    #test todo
        #    Bt = hals_nnls(DXY,DXtDX,V=Bt, exact=False,tol=1e-4)[0]
        #else:
        if nonnegative:
            Bt = hals_nnls(DXY,DXtDX,V=Bt, exact=False, tol=1e-4, n_iter_max=10)[0]
        else:
            # watch out for poor conditionning
            # what about a few steps of gradient descent? To avoid falling into local minimum too fast?
            # Maybe even stochastic gradient
            Bt = np.linalg.solve(DXtDX_reg, DXY)
        B = Bt.T


        # Calculate the current unnormalized error if we need it
        if (tol or return_errors):
            err = np.linalg.norm(Y - DX@B.T)/Ynorm

        rec_errors.append(err)

        if verbose:
            print("MPALS iteration {}, reconstruction error: {}, atoms list: {}".format(iteration, err, Sx))

        if iteration >= 1:
            rec_error_decrease = rec_errors[-2] - rec_errors[-1]

            if cvg_criterion == 'abs_rec_error':
                stop_flag = abs(rec_error_decrease) < tol
            elif cvg_criterion == 'rec_error':
                stop_flag = rec_error_decrease < tol
            else:
                raise TypeError("Unknown convergence criterion")

            if stop_flag:
                if verbose:
                    print("MPALS converged after {} iterations".format(iteration))
                break
    
    return [DX, B], X, Sx, rec_errors


def osmf_pen_als(Y, rank, D, n_iter_max=100, init='random', tol=1e-8, lamb=1,
            verbose=0, return_errors=False,
            cvg_criterion='abs_rec_error', nonnegative=False,simplex=False, optimal_assignment=True):
    """One sparse Dictionary-based Matrix Factorization via Matching Pursuit Alternating (nonnegative) Least Squares

        minimize 1/2 \|Y - A@B.T\|_F^2 + \lambda \|A - DX\|_F^2 wrt X,A, B s.t. \|X_i\|_0\leq 1

        Uses alternating (nonnegative) least squares

    Parameters
    ----------
    Y : numpy array
        input data matrix
    rank : int
        Number of components.
    D : ndarray
        Dictionary for the first mode. Should be normalized with l2 norm columnwise.
    simplex : boolean, default False
        decides if the columns of B are on the simplex or not.
        Not implemented yet.
    lamb : float
        Regularization parameter.
    nonnegative : boolean, default False
        set to True is A and B should be nonnegative.
    optimal_assignment : boolean, default True
        choose if the same atom in D can be selected twice.
    n_iter_max : int, default 100
        Maximum number of outer iteration
    init : {'random', [DX, B]]}, optional
        Type of factor matrix initialization. If a list of factors is passed, it is directly used for initalization. See `initialize_factors`.
    tol : float, default: 1e-8
        Relative reconstruction error tolerance. The algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol`.
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for the outer iterations, works if `tol` is not None. If 'rec_error', iterations stop at current iteration if :math:`(previous rec_error - current rec_error) < tol`. If 'abs_rec_error', iterations terminate when :math:`|previous rec_error - current rec_error| < tol`.

    Returns
    -------
    Decomposed Matrix : list of numpy arrays
        Estimated DMF factors [DX, B]

    X : numpy array
        The sparse coefficients of the first factor

    S : numpy array
        Support of X

    rec_errors : list
         sqrt of data fitting error (relative error) along iterations

    loss : list
        Data fitting error + regularization along iterations

    pen : list
        Penalization error along iterations
    """

    # only need to initialize B
    if init=='random':
        B = np.random.randn(Y.shape[1],rank)
        X = np.random.randn(D.shape[1],rank)
    else:
        X = np.copy(init[0])
        B = np.copy(init[1])
    DX = D@X
    A = DX.copy()

    Ynorm = np.linalg.norm(Y)
    loss = [np.linalg.norm(Y - A@B.T)**2 + lamb*np.linalg.norm(A-DX)**2]
    rec_errors = [np.linalg.norm(Y - A@B.T)/Ynorm]
    pen = [np.linalg.norm(A-DX)/np.linalg.norm(A)]


    for iteration in range(n_iter_max):

        if verbose > 1:
            print("Starting iteration", iteration + 1)


        # A estimate
        B_larger = np.concatenate((B.T, np.sqrt(lamb)*np.eye(rank)), axis=1)
        Y_larger = np.concatenate((Y, np.sqrt(lamb)*DX ), axis=1)
        # todo: update with sum formula
        BBt = B_larger@B_larger.T
        YB = Y_larger@B_larger.T
        if nonnegative:
            A = hals_nnls(YB.T,BBt,V=A.T, exact=False, tol=1e-4, n_iter_max=10)[0].T
        else:
            A = np.linalg.solve(BBt,YB.T).T

        # X estimate
        X = np.zeros(X.shape)
        cost = D.T@A
        if optimal_assignment:
            # use scipy optimize
            Sx,_ = linear_sum_assignment(-cost)
        else:
            # take the maximum columnwise
            Sx = np.argmax(cost,axis=0)
        for j in range(rank):
            X[Sx[j],j]=1
        DX = D[:,Sx] # same than D@X

        # B least squares update 
        if iteration==0:
            Bt=np.copy(B.T)
        AtY = A.T@Y
        AtA = A.T@A

        if simplex:
            #test todo
            Bt = hals_nnls(AtY,AtA,V=Bt, exact=False, n_iter_max=10, tol=1e-4)[0]
        else:
            if nonnegative:
                Bt = hals_nnls(AtY,AtA,V=Bt, exact=False, n_iter_max=10, tol=1e-4)[0]
            else:
                # watch out for poor conditionning
                Bt = np.linalg.solve(AtA, AtY)
        B = Bt.T


        # Calculate the current unnormalized error if we need it
        if (tol or return_errors):
            err = np.linalg.norm(Y - A@B.T)/Ynorm

        rec_errors.append(err)
        loss.append(np.linalg.norm(Y - A@B.T)**2 + lamb*np.linalg.norm(A-DX)**2)
        pen.append(np.linalg.norm(A-DX)/np.linalg.norm(A))

        if verbose:
            print("Penalized ALS iteration {}, reconstruction error: {}, atoms list: {}".format(iteration, err, Sx))

        if iteration >= 1:
            rec_error_decrease = rec_errors[-2] - rec_errors[-1]

            if cvg_criterion == 'abs_rec_error':
                stop_flag = abs(rec_error_decrease) < tol
            elif cvg_criterion == 'rec_error':
                stop_flag = rec_error_decrease < tol
            else:
                raise TypeError("Unknown convergence criterion")

            if stop_flag:
                if verbose:
                    print("MPALS converged after {} iterations".format(iteration))
                break


    return [A, B], X, Sx, rec_errors, loss, pen
