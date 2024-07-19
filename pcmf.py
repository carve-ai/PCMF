#---
# Pathwise Clustered Matrix Factorization (PCMF), Locally Linear (LL) PCMF, and Pathwise Clustered Canonical Correlation Analysis (P3CA) #
#---

# The code implementations of these algorithms are currently proprietary (version 0.001) (see license file). 
# It will soon be released and licensed for academic use only (please read the license file).
# Citation: Buch, Amanda M., Conor Liston, and Logan Grosenick. (2024) Simple and Scalable Algorithms for Cluster-Aware Precision Medicine. 
#           Proceedings of The 27th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 238:136-144

# All Rights Reserved. Copyright (2022-present) Amanda M. Buch, Conor Liston, & Logan Grosenick

#---

###------- IMPORT FUNCTIONS -------###
import numpy as np
from sklearn.utils.extmath import randomized_svd
from itertools import combinations 
import cvxpy as cp
import time
from numba import jit, prange
import matplotlib.pyplot as plt

# to use first run 'python setup.py build_ext --inplace'
# from admm_utils import prox as cprox

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI_score, adjusted_rand_score as ARI_score, rand_score as rand_score
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score, mean_squared_error # cohen_kappa_score, hinge_loss, coverage_error, consensus_score

from scipy.sparse import csr_matrix
from sksparse.cholmod import cholesky, cholesky_AAt
from scipy.spatial.distance import pdist, squareform

###------- Clustering functions -------###

def two_cluster_data(m=[50,50], means=[0,0], n_X=200, sigma=0.075, density=1.0, seed=1, plot=True, intercept=False, gen_seeds=True, seeds='NaN', scale_data=False):
    '''
    Generates two clusters in n_X dimensions with m[0],m[1] observations per class.  
    '''
    # Get clustered data
    X_clusters, u_true, v_true, _ = generate_cluster_PMD_data(m, n_X, sigma, density, 2, means=means, gen_seeds=gen_seeds, seeds=seeds) 
    X_c = np.vstack(X_clusters)
    true_clusters = np.repeat([0,1],m)
    #
    if scale_data:
        scaler = StandardScaler()
        scaler.fit(X_c)
        X_c = scaler.transform(X_c)
    #
    if plot:
        plt.figure(figsize=(6,6))
        plt.scatter(X_clusters[0][:,0],X_clusters[0][:,1], c='darkblue')
        plt.scatter(X_clusters[1][:,0],X_clusters[1][:,1], c='darkorange')
        plt.axis("off")
    #        
    if intercept:
        X_c = np.hstack((X_c,np.ones((X_c.shape[0],1))))
    #
    return X_c, true_clusters

def fit_spectral(X, true_clusters, n_clusters):
    '''Spectral clustering'''
    from sklearn.cluster import SpectralClustering
    #
    data_in = X
    #
    spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=20, affinity="nearest_neighbors").fit(data_in)
    labels = spectral_clustering.labels_
    #
    # Calculate scores
    nmi, ari, ri, mse = calculate_scores(labels, true_clusters)
    # Calculate accuracy
    conf_mat_ord = confusion_matrix_ordered(labels,true_clusters)
    acc = np.sum(np.diag(conf_mat_ord))/np.sum(conf_mat_ord)
    #
    return labels, ari, nmi, acc

###------- PCMF functions -------###

def pcmf_ADMM(X, penalty_list, problem_rank=1, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=2.0, print_progress=True, parallel=False, output_file='NaN', numba=False):
    '''
    Pathwise Clustered Matrix Factorization (PCMF)
    Fits fully constrained PCMF problem iterating between convex clustering and an SVD of rank 'problem_rank' using ADMM updates.
    '''
    # Initialize                                                                                                     
    if weights is False:
        weights = get_weights(X,gauss_coef=0.0)
    else:
        weights = get_weights(X,gauss_coef=gauss_coef, neighbors=neighbors)
    D, chol_factor = chol_D(X.shape[0], X.shape[1], rho, reg_scale=1.0+rho)
    G = Z1 = D*X
    A = Z2 = X.copy()
    U, s, Vh = SVD(A, problem_rank)

    A_list = []
    U_list = []
    s_list = []
    V_list = []
    #
    try:
        # Iterate over penalty grid fitting problem for each value
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            num_obs = X.shape[0]
            num_var = X.shape[1]
            if print_progress == True:
                if parallel == True:
                    file1 = open(output_file, "a")
                    file1.write(str([i+1])+" {:.5e}".format(penalty))
                    file1.write("...")
                    file1.close()
                else:                                                        
                    print("[",i+1,"]","{:.5e}".format(penalty), end="")
                    print("...", end="")

            for it in range(admm_iters):
                if numba is False:
                    A = chol_factor(X + rho*D.T*(G - Z1) + rho*(np.dot(U*s,Vh) - Z2))
                    G = cprox(D*A+Z1, penalty, rho, weights)
                else:
                    A = chol_factor(X + rho*D.T*(G - Z1) + rho*(np.dot(U*s,Vh) - Z2))
                    G = prox_numba_arr(np.zeros_like(G), D*A+Z1, penalty, rho, weights)
                #
                U, s, Vh = SVD(A + Z2, problem_rank)
                #
                Z1 = Z1 + rho*(D*A - G)
                Z2 = Z2 + rho*(A - np.dot(U*s,Vh))
            #
            A_list.append(A)
            V_list.append(Vh)
            s_list.append(s)
            U_list.append(U)
            #
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
    return A_list, U_list, s_list, V_list


def pcmf_PALS(X, penalty_list, rho=1.0, admm_iters = 5, verb=False, weights=False, neighbors=None, gauss_coef=0.5,  print_progress=True, parallel=False, output_file='NaN', numba=False):
    '''
    Locally Linear PCMF
    Relaxation of full PCMF problem to iterate between clustering on u and clustering on V. 
    '''
    # Initialize                                                                                                     
    if weights is False:
        weights = get_weights(X, gauss_coef=0.0)
    else:
        weights = get_weights(X, gauss_coef=gauss_coef, neighbors=neighbors)
    D, chol_factor = chol_D(X.shape[0], X.shape[1], rho)
    V = X.copy()
    for i in range(V.shape[0]):
        V[i,:] = np.mean(X, axis=0)
    W = Z = W2 = Z2 = D*X
    d = np.ones(X.shape[0])
    V_list = []
    u_list = []
    s_list = []
    # First run initial on v_init                                                                                    
    Xu_tildes = []
    for i in range(X.shape[0]):
        Xu_tildes.append(np.dot(X[i,:], V[i,:]))
    u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb)
    u.shape = (len(u),1)
    #
    try:
        # Iterate over penalty grid fitting problem for each value
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            num_obs = X.shape[0]
            num_var = X.shape[1]
            if print_progress == True:
                if parallel == True:
                    file1 = open(output_file, "a")
                    file1.write(str([i+1])+" {:.5e}".format(penalty))
                    file1.write("...")
                    file1.close()
                else:
                    print("[",i+1,"]","{:.5e}".format(penalty), end="")
                    print("...", end="")
            # Update V                                                                                                   
            Xv_tildes = []
            for i in range(X.shape[0]):
                Xv_tildes.append(u[i]*X[i,:])
            Xv = np.asarray(Xv_tildes)
            for it in range(admm_iters):
                if numba is False:
                    V = chol_factor(Xv + rho*D.T*(W - Z))
                    W = cprox(D*V+Z, penalty, rho, weights)
                else:
                    V = chol_factor(nb_add(Xv, rho*D.T*nb_subtract(W,Z)))
                    W = prox_numba_arr(np.zeros_like(W), D*V+Z, penalty, rho, weights)
                #
                Z = Z + D*V - W
            #
            V = l2_ball_proj(V)
            V_list.append(V)
            # Update u                                                                                                   
            Xu_tildes = []
            for i in range(X.shape[0]):
                Xu_tildes.append(np.dot(X[i,:], V[i,:]))
            Xu = np.asarray(Xu_tildes)
            try:
                u = clusterpath_PCMF_subproblem_u(Xu_tildes, 1, penalty, verb)
            except:
                print("PCMF subproblem is not defined for single cluster u, using PMD subproblem.")
                u = clusterpath_PMD_subproblem_u(Xu_tildes, 1, verb)
            u.shape = (len(u),1)
            s = u*Xv
            u_list.append(u)
            s_list.append(d)
            #
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        #
    return V_list, u_list, s_list


def p3ca(X, Y, penalty_list, rho=1.0, admm_iters = 2, cca_iters = 3, 
                 weights=False, neighbors=None, gauss_coef=0.5, verb=True, non_negative=False, parallel=False, output_file='', numba=True):
    '''
    Pathwise Clustered Canonical Correlation Analysis (P3CA)
    Main function to run ADMM-version of P3CA, runs along penalty path penalty_list
    Parameters:
        X
        Y
        penalty_list
        rho
        admm_iters = number of admm updates for each u and each v
        cca_iters = number of u / v updates
        weights
        neighbors
        verb: whether to print output of iteration and penalty at each outer iteration.
    Returns:
    '''
    # Initialize U as X and V as Y
    Dx, chol_factor_X = chol_D(X.shape[0], X.shape[1], rho)
    Wx = Zx = Dx*X
    Dy, chol_factor_Y = chol_D(Y.shape[0], Y.shape[1], rho)
    Wy = Zy = Dy*Y
    #
    U_list = []
    V_list = []
    penalty = penalty_list[0]
    # Initial U update
    if penalty_list[0] > penalty_list[-1]:
        V_initial = np.tile(np.mean(Y,axis=0),Y.shape[0]).reshape(Y.shape)
    else:
        V_initial = Y.copy()
    Xu_tildes = []
    for i in range(X.shape[0]):
        Xu_tildes.append(np.dot(np.outer(X[i,:].T,Y[i,:]),V_initial[i,:]))
    Xu = np.asarray(Xu_tildes)
    U, Wx, Zx = admm_CCA_update(Xu, chol_factor_X, Dx, Wx, Zx, rho, penalty, X, weights=weights, neighbors=neighbors, numba=numba, gauss_coef=gauss_coef)
    #
    # Initial V update
    if penalty_list[0] > penalty_list[-1]:
        U_initial = np.tile(np.mean(X,axis=0),X.shape[0]).reshape(X.shape)
    else:
        U_initial = X.copy()
    Yv_tildes = []
    for i in range(Y.shape[0]):
        Yv_tildes.append(np.dot(np.outer(Y[i,:].T,X[i,:]),U_initial[i,:]))
    Yv = np.asarray(Yv_tildes)
    V, Wy, Zy = admm_CCA_update(Yv, chol_factor_Y, Dy, Wy, Zy, rho, penalty, Y, weights=weights, neighbors=neighbors, numba=numba, gauss_coef=gauss_coef)
    #
    try:
        for i in range(len(penalty_list)):
            penalty = penalty_list[i]
            print_progress = verb
            if print_progress == True:
                if parallel == True:
                    file1 = open(output_file, "a")
                    file1.write(str([i+1])+" {:.5e}".format(penalty))
                    file1.write("...")
                    file1.close()
                else:
                    print("[",i+1,"]","{:.5e}".format(penalty), end="")
                    print("...", end="")

            for it in range(cca_iters):
                # U update
                for it in range(admm_iters):
                    Xu_tildes = []
                    for i in range(X.shape[0]):
                        Xu_tildes.append(np.dot(np.outer(X[i,:].T,Y[i,:]),V[i,:]))
                    #
                    Xu = np.asarray(Xu_tildes)
                    U, Wx, Zx = admm_CCA_update(Xu, chol_factor_X, Dx, Wx, Zx, rho, penalty, X, weights=weights, neighbors=neighbors, gauss_coef=gauss_coef)
                    if non_negative:
                        U[U<0] = 0
                    #
                # V update
                for it in range(admm_iters):
                    Yv_tildes = []
                    for i in range(X.shape[0]):
                        Yv_tildes.append(np.dot(np.outer(Y[i,:].T,X[i,:]),U[i,:]))
                    Yv = np.asarray(Yv_tildes)

                    V, Wy, Zy = admm_CCA_update(Yv, chol_factor_Y, Dy, Wy, Zy, rho, penalty, Y, weights=weights, neighbors=neighbors, gauss_coef=gauss_coef)
                    if non_negative:
                        V[V<0] = 0
                    #
            U_list.append(U)
            V_list.append(V)
            #
    except KeyboardInterrupt:
        print("KeyboardInterrupt has been caught.")
        #
    return U_list, V_list
