from dlraos.algorithms import osmf_mpals, osmf_pen_als
from dlraos.utils import count_support_onesparse, gen_mix
import tensorly as tl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from itertools import count, permutations
import plotly.express as px
import copy

# Seeding
np.random.seed(seed=0)

# Generation
k = 1
r = 6 #6,2
n = 50 #10
m = 50 #20
d = 100 #50
SNR = 20  # dB # provides the noise level in dB
distr = 'Uniform'
tol = 0
verbose = 0
nonnegative = True
n_iter_max = 20
N = 30

store_pd = pd.DataFrame(columns=["value", "error type", "algorithm"])

for i in range(N):

    # Data gen
    Y, Ytrue, D, B, X, S, sig = gen_mix([m, n, d, r], snr=SNR,  distr=distr)

    # Initialize with true factors for sanity check
    #init = [X,B]

    # Fix random init
    X0 = np.random.rand(d,r)
    B0 = np.random.rand(m,r)
    init = [X0,B0]
    
    # MPALS/TrickOMP. A is exactly in D. 
    cp_tensor_est1, X1, S1, err1 = osmf_mpals(Y, r, D, init=copy.deepcopy(init),  return_errors=True, verbose=verbose, n_iter_max=n_iter_max, nonnegative=nonnegative, tol=tol, optimal_assignment=True)

    # penalized DLRA, ALS algorithm. A is not exactly in D, penalization is Frobenius norm.
    eps = 1e-1
    cp_tensor_est3, X3, S3, err3, loss, pen = osmf_pen_als(Y, r, D, init=copy.deepcopy(init), lamb=eps, return_errors=True, verbose=verbose, n_iter_max=n_iter_max, nonnegative=nonnegative, tol=tol, optimal_assignment=True)

    val_1 = count_support_onesparse(S1,S)
    val_3 = count_support_onesparse(S3,S)

    print('error solution', np.linalg.norm(Y - D@X@B.T, 'fro')/np.linalg.norm(Y, 'fro'))
    print('final error MPALS method, random init', err1[-1], val_1)
    print('final error penALS method, random init', err3[-1], val_3)


    # Storing in Pandas Dataframe
    dic = {
    'experiment name': 2*['toy experiment'],
    'k': 2*[k],
    'r': 2*[r],
    'd': 2*[d],
    'n': 2*[n],
    'm': 2*[m],
    # don't hesitate to put more hyperparameters
    "Relative reconstruction error": [err1[-1], err3[-1]],
    'Support error': [val_1,val_3],
    'algorithm':['mpals', 'penals'] 
    }
    data = pd.DataFrame(dic)
    store_pd = store_pd.append(data, ignore_index=True)

# calling boxplot from plotly
fig = px.box(store_pd, x="algorithm", y="Relative reconstruction error", points='all', color="algorithm", title="Relative reconstruction error")

# Figure layout options
fig.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False
)
fig.show()
fig2 = px.box(store_pd, x="algorithm", y="Support error", points='all', color="algorithm",
labels={
    'value': 'Support Recovery (%)',
    'algorithm':''
}, title="Support recovery")

# Figure 2 layout options
fig2.update_layout(
    font_family="HelveticaBold",
    font_size=15,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(zeroline=False, gridcolor='rgb(233,233,233)'),
    paper_bgcolor="white",#'rgb(233,233,233)',
    plot_bgcolor="white",#'rgb(233,233,233)',
    showlegend=False
)
fig2.show()

# Todo update path for your configuration
#year = 2021
#month = 10
#day = 20
#path = '../..'
#stor_name = '{}-{}-{}'.format(year,month,day)
#store_pd.to_pickle('{}/data/XP_synth/{}_DMF'.format(path,stor_name))
#fig.write_image('{}/data/XP_synth/{}_DMF_plot1.pdf'.format(path,stor_name))
#fig2.write_image('{}/data/XP_synth/{}_DMF_plot2.pdf'.format(path,stor_name))
# Frontiers export
#fig.write_image('{}/data/XP_synth/{}_DMF_plot1.jpg'.format(path,stor_name))
#fig2.write_image('{}/data/XP_synth/{}_DMF_plot2.jpg'.format(path,stor_name))
#
# to load data
#store_pd = pd.read_pickle('{}/data/XP_synth/{}_DMF'.format(path,stor_name))
