########################################################
## Methods for GOPHER-LP and GOPHER-MultiLP           ##
## Implements RR-on-bins and MultiLP algorithms       ##
########################################################

from utils import *
import scipy.stats as st
from sklearn.cluster import KMeans
import multiprocessing as mp
import time
from scipy import stats

# Helper Functions
def find_indices(arr, bin_centers):
    """Find index of the nearest bin center for each element in arr."""
    diff_matrix = np.abs(arr[:, None] - bin_centers[None, :])
    return np.argmin(diff_matrix, axis=1)


def norm_prob_dist(Yarr, mean, std, bins):
    """Compute normal probability distribution with DP mean/std."""
    prob_dis = stats.norm(loc=mean, scale=std)
    low, up = stats.norm.interval(0.99, loc=mean, scale=std)

    low = max(np.min(Yarr), low)
    up = min(np.max(Yarr), up)
    # low =np.min(Yarr)
    # up = np.max(Yarr)
    print(f"Clipped range: [{low:.3f}, {up:.3f}]", flush=True)

    bin_edges = np.linspace(low, up, bins + 1)
    Y_uniq = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    inter = (Y_uniq.max() - Y_uniq.min()) / (len(Y_uniq) - 1)
    counts = prob_dis.pdf(Y_uniq) * inter
    prob_y = dict(zip(Y_uniq, counts))

    return prob_y

def save_RR_Yhat(label_cls, Y_full, Y_uniq, temp_y, eps, seed):
    """Apply Randomized Response to each bin of Y_full."""
    rng = np.random.RandomState(seed)
    n = Y_full.shape[0]
    priv_y = np.zeros(n, dtype=np.float32)

    bin_indices = find_indices(Y_full, Y_uniq)
    bin_indices[np.isnan(Y_full)] = -1

    for i in range(n):
        if bin_indices[i] == -1 or np.isnan(Y_full[i]):
            priv_y[i] = np.nan
        else:
            idx = bin_indices[i]
            priv_y[i] = compute_RR_bins(temp_y[idx], label_cls, eps, rng)
    return priv_y

########################################################
# GOPHER-LP (RR-on-bins)
########################################################

def RR_on_bins(Yarr,mean_dp,var_dp, eps, eps1, bins, seed):
    """Run RR-on-bins mechanism on continuous phenotype values."""
    Y = Yarr[~np.isnan(Yarr)]
    n = Y.shape[0]
    print(f"Max(Y)={Y.max():.3f}, Min(Y)={Y.min():.3f}", flush=True)

    # Differentially Private mean and variance
    # mean_dp = np.mean(Y) + laplace_noise((Y.max() - Y.min()) / n, eps1 * 0.1, 1)
    # std_dp = np.sqrt(np.var(Y) + np.abs(laplace_noise((Y.max() - Y.min()) ** 2 / n, 0.9 * eps1, 1)))
    std_dp = np.sqrt(var_dp)
    print(f"DP mean={mean_dp:.3f}, DP std={std_dp:.3f}", flush=True)

    # Construct probability map
    prob_map = norm_prob_dist(Y,mean_dp,std_dp, bins)
    Y_uniq = np.fromiter(prob_map.keys(), dtype=np.float32)
    prob_y = np.fromiter(prob_map.values(), dtype=float)

    eps2 = eps - eps1
    print(f"Effective epsilon for optimization: {eps2}", flush=True)

    loss_mat, y_opt = create_opt_mat(prob_y, Y_uniq, eps2)
    mapping_mat, min_idx = compute_map(bins, loss_mat)
    temp_y, label_cls = get_Yopt(y_opt, mapping_mat, min_idx, bins, eps2)

    Y_priv = save_RR_Yhat(label_cls, Yarr, Y_uniq, temp_y, eps2, seed)
    return Y_priv

########################################################
# GOPHER-MultiLP (Parallel RR-on-bins)
########################################################
def pool_rr_on_bins(Yarr, bins, eps, eps1, mean_dp, var_dp,overall_mean,overall_var,num_chunks,seed):
    """Run parallel RR-on-bins for personalized DP priors."""
    Y = Yarr[~np.isnan(Yarr)]
    n = Y.shape[0]

    eps2 = eps - eps1
    print(eps2,flush=True)
    rng = np.random.RandomState(seed)
    
    # indices = np.arange(mean_dp.shape[0])
    # rng.shuffle(indices)
    # chunks = np.array_split(indices, num_chunks)
    
    kmeans = KMeans(n_clusters=num_chunks, random_state=seed)
    kmeans.fit(mean_dp.reshape(-1, 1))
    labels = kmeans.labels_
    clustered_indices = {i: np.where(labels == i)[0] for i in range(np.max(labels) + 1)}
    chunks = [clustered_indices[i] for i in range(len(clustered_indices))]
    print(f"Formed {len(chunks)} clusters", flush=True)
    
    print(f"Running Multi-LP with {len(chunks)} chunks on {mp.cpu_count()} cores...", flush=True)
    
    # Y_priv= RR_on_bins(Yarr, eps, eps1, bins, seed)
    if len(chunks) < 10:
        with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=5) as pool:
            results = pool.map(real_opt_rr, [(Y[chunk],np.mean(mean_dp[chunk]), np.mean(var_dp[chunk]),bins, eps2, rng, i) for i, chunk in enumerate(chunks)])
        temp = np.concatenate(results)

    else:
        Y_uniq, prob_y = ind_prob_dist(Y, bins, mean_dp, var_dp,overall_mean,overall_var, eps1)

        indices = np.arange(mean_dp.shape[0])
        rng.shuffle(indices)
        chunks = np.array_split(indices, num_chunks)
        # print(f"Formed {len(chunks)} clusters", flush=True)
        with mp.Pool(processes=mp.cpu_count(), maxtasksperchild=5) as pool:
            results = pool.map(opt_rr, [(prob_y[chunk, :], Y_uniq, Y[chunk], bins, eps2, rng, i)
                                        for i, chunk in enumerate(chunks)])
        temp = np.concatenate(results)
        
    nan_idx = np.where(np.isnan(Yarr))[0]
    Y_priv = np.zeros_like(Yarr)
    index = 0
    for i, chunk in enumerate(chunks):
        for j in range(len(chunk)):
            Y_priv[chunk[j]] = temp[index+j]
        index +=  len(chunk)
    Y_priv[nan_idx] = np.nan

    return Y_priv


def ind_prob_dist(Yarr, bins, mean, variance,overall_mean,overall_var, eps1):
    """Estimate private prior using Gaussian distribution."""

    
    sd = np.sqrt(overall_var)
    low, up = stats.norm.interval(0.99, loc=overall_mean, scale=sd)
    
    low, up = max(np.min(Yarr), low), min(np.max(Yarr), up)
    print(f"Clipped range: [{low:.3f}, {up:.3f}]", flush=True)
    
    bin_edges = np.linspace(low, up, bins + 1)
    Y_uniq = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    n = len(variance)
    prob_y = np.zeros((n, len(Y_uniq)))
    inter = (Y_uniq.max() - Y_uniq.min()) / (len(Y_uniq) - 1)

    for i in range(n):
        prob = stats.norm(loc=mean[i], scale=np.sqrt(variance[i]))
        prob_y[i, :] = prob.pdf(Y_uniq) * inter
        # prob_y[i, :] /= np.sum(prob_y[i, :])
        
    # print(prob_y,flush=True)
    return Y_uniq, prob_y

def real_opt_rr(args):
    """Apply RR-on-bins algorithm to a subset of Y values (for real phenotypes)."""
    Y, mean_dp,var_dp,bins, eps, rng, _ = args
    n = Y.shape[0]

    Y_priv = np.zeros(n, dtype=np.float32)
    
    std_dp = np.sqrt(var_dp)
    # print(f"DP mean={mean_dp.astype('float32')[0]:.3f}, DP std={std_dp.astype('float32')[0]:.3f}", flush=True)

    prob_map = norm_prob_dist(Y,mean_dp,std_dp, bins)
    Y_uniq = np.fromiter(prob_map.keys(), dtype=np.float32)
    prob_y = np.fromiter(prob_map.values(), dtype=float)

    loss_mat, y_opt = create_opt_mat(prob_y, Y_uniq, eps)
    mapping_mat, min_idx = compute_map(bins, loss_mat)
    temp_y, label_cls = get_Yopt(y_opt, mapping_mat, min_idx, bins, eps)

    Y_priv = save_RR_Yhat(label_cls, Y, Y_uniq, temp_y, eps, rng)
    return Y_priv

    
def opt_rr(args):
    """Apply RR-on-bins algorithm to a subset of Y values (for parallel execution)."""
    prob_y, Y_uniq, Y, bins, eps, rng, _ = args
    n = Y.shape[0]
    Y_priv = np.zeros(n, dtype=np.float32)

    for i in range(n):
        loss_mat, y_opt = create_opt_mat(prob_y[i, :], Y_uniq, eps)
        mapping_mat, min_idx = compute_map(bins, loss_mat)
        temp_y, label_cls = get_Yopt(y_opt, mapping_mat, min_idx, bins, eps)

        idx = np.abs(Y_uniq - Y[i]).argmin()
        
        Y_priv[i] = compute_RR_bins(temp_y[idx], label_cls, eps, rng)
     
   
    return Y_priv


# RR-on-bins dynamic programming algorithm computation functions
def create_opt_mat(prob_y, Y_uniq, eps):
    """Create optimization matrix for RR binning."""
    k = len(prob_y)
    L = np.ones((k + 1, k + 1), dtype=np.float32) * np.inf
    Yopt = np.zeros((k + 1, k + 1), dtype=np.float32)

    for r in range(1, k + 1):
        t1, t2, t3 = np.sum(prob_y * Y_uniq), np.sum(prob_y), np.sum(prob_y * (Y_uniq ** 2))
        for i in range(r, k + 1):
            t1 += (prob_y[i - 1] * Y_uniq[i - 1] * (np.exp(eps) - 1))
            t2 += (prob_y[i - 1] * (np.exp(eps) - 1))
            t3 += (prob_y[i - 1] * (Y_uniq[i - 1] ** 2) * (np.exp(eps) - 1))
            Yopt[r][i] = t1 / t2
            L[r, i] = t3 - (2 * t1 * Yopt[r][i]) + t2 * (Yopt[r][i] ** 2)
    return L, Yopt

def compute_map(bins, L):
    """Compute mapping matrix from optimization losses."""
    A = np.ones((bins + 1, bins + 1), dtype=np.float32) * np.inf
    A[0, 0] = 0
    min_idx = np.zeros((bins + 1, bins + 1), dtype=np.int32)

    for i in range(1, bins + 1):
        for j in range(1, i + 1):
            arr = [A[r][j - 1] + L[r + 1][i] for r in range(i)]
            idx = np.argmin(arr)
            A[i][j] = A[idx][j - 1] + L[idx + 1][i]
            min_idx[i, j] = idx
    return A, min_idx


def get_Yopt(Yopt, A, min_idx, bins, eps):
    """Extract optimized Y values and labels from mapping."""
    k = bins
    newY = np.zeros(k)
    i = k
    j = np.argmin([A[k][d] / (d - 1 + np.exp(eps)) for d in range(1, k + 1)]) + 1
    labelCls = np.zeros(j)

    while i > 0 and j > 0:
        idx = min_idx[i, j]
        newY[idx:i] = Yopt[idx + 1, i]
        labelCls[j - 1] = Yopt[idx + 1, i]
        i = idx
        j -= 1
    return newY, labelCls


def compute_RR_bins(old_label, label_cls, eps, rng):
    """Perform Randomized Response between bins."""
    numBins = label_cls.size
    labelIdx = np.where(np.isclose(label_cls, old_label))[0][0]
    rate = 1 / (np.exp(eps) + numBins - 1)
    prob = np.ones(numBins) * rate
    prob[labelIdx] = 1 - rate * (numBins - 1)
    new_idx = rng.choice(numBins, 1, p=prob)
    return label_cls[new_idx]

    




    
    
