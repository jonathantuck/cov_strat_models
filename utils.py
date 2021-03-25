import numpy as np
import cvxpy as cp 
import strat_models
from scipy import sparse
from scipy.sparse.linalg import factorized

def joint_cov_prox(Y, nu, theta, t):
    """
    Proximal operator for joint covariance estimation
    """
    if Y is None:
        return nu

    n, nk = Y[0].shape
    Yemp = Y[0]@Y[0].T
    
    s, Q = np.linalg.eigh(nu/(t*nk)-Yemp/nk)
    w = ((t*nk)*s + np.sqrt(((t*nk)*s)**2 + 4*(t*nk)))/2
    return Q @ np.diag(w) @ Q.T

class covariance_max_likelihood_loss(strat_models.Loss):
    """
    f(theta) = Trace(theta @ Y) - logdet(theta)
    """
    def __init__(self):
        super().__init__()
        self.isDistribution = True

    def evaluate(self, theta, data):
        assert "Y" in data
        return np.trace(theta @ data["Y"]) - np.linalg.slogdet(theta)[1]

    def setup(self, data, G):
        Y = data["Y"]
        Z = data["Z"]

        K = len(G.nodes())

        shape = (data["n"], data["n"])
        theta_shape = (K,) + shape

        #preprocess data
        for y, z in zip(Y, Z):
            vertex = G._node[z]
            if "Y" in vertex:
                vertex["Y"] += [y]
            else:
                vertex["Y"] = [y]

        Y_data = []
        for i, node in enumerate(G.nodes()):
            vertex = G._node[node]
            if 'Y' in vertex:
                Y = vertex['Y']
                Y_data += [Y]
                del vertex['Y']
            else:
                Y_data += [None]

        cache = {"Y": Y_data, "n":data["n"], "theta_shape":theta_shape, "shape":shape, "K":K}
        return cache

    def prox(self, t, nu, warm_start, pool, cache):
        """
        Proximal operator for joint covariance estimation
        """
        res = pool.starmap(joint_cov_prox, zip(cache["Y"], nu, warm_start, t*np.ones(cache["K"])))
        return np.array(res)

    def logprob(self, data, G):
        
        logprobs = []
        
        for y,z in zip(data["Y"], data["Z"]):
            n, nk = y.shape
            
            Y = (y@y.T)/nk
            
            if (np.zeros((n,n)) == Y).all():
#                 logprobs += [0]
                continue            
            
            theta = G._node[z]["theta"]
            logprobs += [np.linalg.slogdet(theta)[1] - np.trace(Y@theta)]

        return logprobs

    def sample(self, data, G):
        Z = turn_into_iterable(data["Z"])
        sigmas = [np.linalg.inv(G._node[z]["theta"]) for z in Z]

        n = sigmas[0].shape[0]
        return [np.random.multivariate_normal(np.zeros(n), sigma) for sigma in sigmas]
    
class trace_offdiagL1Norm(strat_models.Regularizer):
    """
    r(theta) = lambd_0 * Tr(theta) + lambd_1 * || theta ||_{off diagonal, 1}
    """
    def __init__(self, lambd=(1,1)):
#         super().__init__(lambd)
        self.lambd = lambd
    
    def evaluate(self, theta):
        od_idx = np.where(~np.eye(theta.shape[0],dtype=bool))
        
        return self.lambd[0]*np.trace(theta) + self.lambd[1]*np.norm(theta[od_idx], 1)
    
    def prox(self, t, nu, warm_start, pool):
        if self.lambd == (0,0):
            return nu
        
        K = nu.shape[0]
        n = nu.shape[1]
        
        diag_idx = np.where(np.eye(n,dtype=bool))
        od_idx = np.where(~np.eye(n,dtype=bool))
        
        T = np.zeros((K, n, n))

        for k in range(K):
            T[k][diag_idx] = nu[k][diag_idx] - self.lambd[0]*t
            T[k][od_idx] = np.maximum(nu[k][od_idx] - self.lambd[1]*t, 0) - np.maximum(-nu[k][od_idx] - self.lambd[1]*t, 0)
        
        return T

def backtest(ws):
    RETURNS = np.array(df_heldout[sectors])/100
    value = 1
    vals = []
    risk = []
    lev = []
    W = []

    for date in range(1,df_heldout.shape[0]):
        vix = int(df_heldout.iloc[date]["VIX_quantile_yesterday"])
        vol = int(df_heldout.iloc[date]["5_day_trail_vol_yesterday"])

        w = ws[vix, vol]

        value *= (1+RETURNS[date, :])@w
        vals += [value]
        lev += [np.linalg.norm(w,1)]
        W += [w.reshape(-1,1)]

        risk += [np.sqrt(w@covs[vix,vol]@w)]
        
    ann_ret, ann_risk = annualized_return_risk(vals)
    
    return ann_ret, ann_risk

def annualized_return_risk(vals):
    """
    Compute annualized return and risk of portfolio value vector.
    """
    P = 252
    v = np.array(vals)
    vt1 = v[1:]
    vt = v[:-1]
    rets = (vt1-vt)/vt
    
    ann_return = np.mean(rets)*P
    ann_risk = np.std(rets)*np.sqrt(P)
    
    return ann_return, ann_risk

def get_wts_cm_cc(gamma, mu, cov, nodes):
    """
    Get portfolio weights for problem 
    with common mean, and common covariance.
    """
    ws = dict()
    w = cp.Variable(9)
    obj_common = gamma*cp.quad_form(w, cov) - w@mu
    cons_common = [sum(w)==1, cp.norm1(w) <= 2]
    prob_common = cp.Problem(cp.Minimize(obj_common), cons_common)
    prob_common.solve(verbose=False, solver="MOSEK")
    
    for (vix, vol) in nodes:
        ws[vix, vol] = w.value
            
    return ws

def get_wts_cm_sc(gamma, mu, covs):
    """
    Get portfolio weights for problem
    with common mean, stratified covariance
    """
    ws = dict()
    for (vix, vol) in covs.keys():
        w = cp.Variable(9)
        obj =  gamma*cp.quad_form(w, covs[vix, vol]) + w@mu
        cons = [sum(w)==1, cp.norm1(w) <= 2]
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(verbose=False, solver="MOSEK")
        ws[vix, vol] = w.value

    return ws

def ANLL(ys, zs, K, theta):
    nlls = []
    for i in range(K):
        if sum(zs == i) == 0:
            if min(np.linalg.eigvals(theta[i])) > 0:
                nlls += [0]
            else:
                nlls += [np.inf]
            continue
        Yi = ys[zs == i].T @ ys[zs == i]/(sum(zs==i))
        nll = np.trace( Yi@theta[i] ) - np.linalg.slogdet(theta[i])[1]
        nlls += [nll]
    return np.mean(nlls)/ys.shape[0], nlls

def soft_threshold(a, k):
    return np.maximum(a - k, 0) - np.maximum(-a - k, 0)

def offdiag(A):
    idx = np.where(~np.eye(A.shape[0],dtype=bool))
    return A[idx]

def fit(zs, ys, L, lam_1, lam_2, rho=10, maxiter=100, verbose=True, warm_start=None,
       eps_abs = 1e-5, eps_rel = 1e-5):
    """
    minimize sum_{i=1}^K Tr(Y_i S_i) - N_i \log\det(S_i) + lam_1 \Tr(\tilde S_i) + lam_2 \|\tilde S_i\|_{od, i} +
        \mathcal L(\hat S_1,\ldots,\hat S_K).
    
    S_i^{k+1} = \argmin_S Tr(Y_i S) - N_i \log\det(S) + (rho / 2) ||S - \hat S_i^k + U_1^k||_F^2
    or S_i^{k+1} = \argmin_S Tr(Y_i S) / N_i - \log\det(S) + (rho / N_i / 2) ||S - \hat S_i^k + U_1^k||_F^2
    \diag(\tilde S_i^{k+1}) = SoftThreshold_{lam_1 / rho}(\diag(\hat S_i^k - U_2^k))
    \offdiag(\tilde S_i^{k+1}) = SoftThreshold_{lam_2 / rho}(\offdiag(\hat S_i^k - U_2^k))
    \hat S^{k+1} = \argmin_S \sum_{i,j} W_{ij} ||S_i - S_j||_F^2 + rho ||S - (S^{k+1} + \tilde S^{k+1}) / 2 -
        (U_1^k + U_2^k) / 2||_F^2
    U_1^{k+1} = U_1^k + S^{k+1} - \hat S^{k+1}
    U_2^{k+1} = U_2^k + \tilde S^{k+1} - \hat S^{k+1}
    """
    K = int(zs.max() + 1)
    N, n = ys.shape
    Ys, cts = [], []
    for i in range(K):
        idx = zs == i
        cts.append(idx.sum()) #N_i, number of samples per z
        ys_i = ys[idx]
        Ys.append(ys_i.T @ ys_i)
    
    if verbose:
        print ("Fitting covariance stratified model.")
        print ("%d stratification values, %d data points, %d dimensions" % (K, N, n))
        print ("%d" % (K * n * n), "optimization variables")
        print ("lam_1 = %3.3e, lam_2 = %3.3e, rho = %3.3e, maxiter=%d" % (lam_1, lam_2, rho, maxiter))
        print ("count per stratification value:", cts)
        print (Ys[0].shape)

    shape = (K, n, n)
    if warm_start is None:
        warm_start = []
        for _ in range(5):
            warm_start.append(np.zeros(shape))
    inv_covs_loss, inv_covs_reg, inv_covs_lapl, U_1, U_2 = warm_start
    
    solve = factorized(L.tocsc() + rho * sparse.eye(K, format='csc'))
    
    for _ in range(maxiter):
        # inv_covs_loss
        for i in range(K):
            if cts[i] == 0:
                inv_covs_loss[i] = (inv_covs_lapl[i] - U_1[i])
                continue
            w, v = np.linalg.eigh((rho/cts[i]) * (inv_covs_lapl[i] - U_1[i]) - Ys[i]/cts[i])
            w_new = (w*cts[i]/rho + np.sqrt((w*cts[i]/rho)**2 + 4*cts[i]/rho))/2
            inv_covs_loss[i] = v @ np.diag(w_new) @ v.T        
        
        # inv_covs_reg
        for i in range(K):
            inv_covs_reg[i][np.arange(n), np.arange(n)] = np.diag(inv_covs_lapl[i] - U_2[i] - lam_1/rho) #diagonal elements
            
            st2 = soft_threshold(inv_covs_lapl[i] - U_2[i], lam_2 / rho)
            od_idx = np.where(~np.eye(n,dtype=bool)) #gets off_diags
            inv_covs_reg[i][od_idx] = st2[od_idx]            
        
        # inv_covs_lapl
        rhs = (inv_covs_loss + inv_covs_reg) / 2 + (U_1 + U_2) / 2
        rhs *= rho
        inv_covs_lapl_new = solve(rhs.reshape(K, n*n)).reshape(shape)
        S = rho * np.repeat(inv_covs_lapl_new - inv_covs_lapl, 2, axis=0)
        inv_covs_lapl = inv_covs_lapl_new.copy()

        # U_1
        R_1 = inv_covs_loss - inv_covs_lapl
        U_1 += R_1
        
        # U_2
        R_2 = inv_covs_reg - inv_covs_lapl
        U_2 += R_2
        
        R = np.concatenate([R_1, R_2], axis=0)
        
        # stopping criterion
        eps_pri = np.sqrt(2 * K * n * n) * eps_abs + eps_rel * max(np.linalg.norm(np.concatenate([inv_covs_loss, inv_covs_reg], axis=0)),
                                                                   np.linalg.norm(np.repeat(inv_covs_lapl, 2, axis=0)))
        eps_dual = np.sqrt(K * n * n) * eps_abs + eps_rel * np.linalg.norm(np.concatenate([U_1, U_2], axis=0))
        if verbose:
            print (np.linalg.norm(R), np.linalg.norm(S), eps_pri, eps_dual)
        
    return inv_covs_loss, inv_covs_reg, inv_covs_lapl