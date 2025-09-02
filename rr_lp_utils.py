########################################################
## Methods used for LP (RR-on-bins) and Multi-LP     ##
########################################################

from utils import *
import scipy.stats as st 
import multiprocessing as mp
from sklearn.cluster import KMeans
import time

def find_indices(arr, bin_centers):
    
    diff_matrix = np.abs(arr[:, None] - bin_centers[None, :])
    indices = np.argmin(diff_matrix, axis=1)
    return indices


def norm_prob_dist(Yarr,mean,std,bins):

    prob_dis = stats.norm(loc=mean,scale=std)
    
    # low,up = st.t.interval(0.99, df=len(Yarr)-1, loc=mean,  scale=sd)
    low, up = stats.norm.interval(0.95, loc=mean, scale=std)
    low = max(np.min(Yarr),low)
    up = min(np.max(Yarr),up)
    
    # low = Yarr.min()
    # up = Yarr.max()

    print(low,up,flush=True)
    
    bin_edges = np.linspace(low,up,bins+1)
    Y_uniq = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Y_uniq = np.linspace(low,up,bins)
    # print(Y_uniq.shape)
    # Y_uniq = np.unique(np.linspace(low,up,num=bins+1,dtype=dtype)[:-1])
    # Y_uniq= Y_uniq[:-1]
    
    print(f"Y_uniq size: {len(Y_uniq)}")
    inter = (Y_uniq.max()-Y_uniq.min())/(len(Y_uniq)-1)
    counts = prob_dis.pdf(Y_uniq)*inter
    prob_y = dict(zip(Y_uniq,counts))

    return prob_y

def save_RR_Yhat(label_cls,Y_full,Y_uniq,temp_y,eps,seed):

    rng = np.random.RandomState(seed)
    
    n= Y_full.shape[0]
    priv_y = np.zeros(n,dtype=np.float32)

    # bin_indices = np.digitize(Y_full, Y_uniq)
    bin_indices = find_indices(Y_full, Y_uniq)
    bin_indices[np.isnan(Y_full)] = -1
    
    for i in range(n):

        
        if bin_indices[i] == -1 or np.isnan(Y_full[i]):
            priv_y[i] = np.nan
        else:
            # if bin_indices[i] == 0:
            #     bin_indices[i]= 1
            # idx = bin_indices[i]-1
            # new_label = temp_y[idx]
            idx = bin_indices[i]
            priv_y[i] = compute_RR_bins(temp_y[idx],label_cls,eps,rng)

    return priv_y


def RR_on_bins(Yarr,eps,eps1,bins,seed):
	
    Y=Yarr[~np.isnan(Yarr)]
    n=Y.shape[0]
    print(f"Max of phenotype: {Y.max()}, min: {Y.min()}", flush=True)

    mean_dp = np.mean(Y) + laplace_noise((Y.max()-Y.min())/n, eps1*0.1,1)
    std_dp = np.sqrt(np.var(Y)+laplace_noise((Y.max()-Y.min())**2/n, 0.9*eps1, 1))
    print(f"DP mean of phenotype: {mean_dp} and DP standard dev: {std_dp}",flush=True)
    
    prob_map = norm_prob_dist(Y,mean_dp.astype('float32')[0],std_dp.astype('float32')[0],bins)
    
    Y_uniq = np.fromiter(prob_map.keys(),dtype=np.float32)
    prob_y = np.fromiter(prob_map.values(),dtype=float)

    print(f"Original Y: # of bins={bins} and # of unique values={Y_uniq.shape}", flush=True)

    eps2 = eps - eps1

    print(f"Epsilon for optimization: {eps2}")
    
    loss_mat,y_opt = create_opt_mat(prob_y,Y_uniq, eps2)
    mapping_mat,min_idx = compute_map(bins,loss_mat) 
    temp_y, label_cls = get_Yopt(y_opt,mapping_mat,min_idx,bins,eps2) #get_Yopt(Yopt,A,min_idx,bins)
    num_bins = label_cls.size

    Y_priv = save_RR_Yhat(label_cls,Yarr,Y_uniq,temp_y,eps2,seed)

    return Y_priv


def pool_rr_on_bins(Y_full,Yarr,bins,eps,eps1,mean_dp,var_dp,seed):

    Y=Yarr[~np.isnan(Yarr)]
    n=Y.shape[0]
    
    Y_uniq, prob_y = ind_prob_dist(Y,bins,mean_dp,var_dp,eps1) #np.repeat(y_var,n))

    print(f"Original Y: # of bins={bins} and # of unique values={Y_uniq.shape}", flush=True)

    eps2 = eps- eps1
    rng = np.random.RandomState(seed)

    num_chunks = 1000 #int(n/bins) #1000
    kmeans = KMeans(n_clusters=num_chunks, random_state=seed)  
    # mean_dp_reshaped = mean_dp.reshape(-1, 1)  
    kmeans.fit(mean_dp.reshape(-1, 1))
    labels =  kmeans.labels_
    
    clustered_indices = {i: np.where(labels == i)[0] for i in range(np.max(labels) + 1)}
    # for cluster_id, indices in clustered_indices.items():
    #     print(f"Cluster {cluster_id} has {len(indices)} elements.")
    
    chunks = [clustered_indices[i] for i in range(len(clustered_indices))]
    
    print(f'Number of clusters: {len(chunks)}', flush=True)

    print(mp.cpu_count(),flush=True)
    pool = mp.Pool(processes=mp.cpu_count(),maxtasksperchild=5)
    
    results = pool.map(opt_rr,[(prob_y[chunk,:], Y_uniq, Y[chunk], bins,eps2,rng,i) for i, chunk in enumerate(chunks)])
    pool.close()
    pool.join()

    temp = np.concatenate(results)

    index =0
    Y_priv = np.zeros_like(Yarr)
    for i, chunk in enumerate(chunks):
        for j in range(len(chunk)):
            Y_priv[chunk[j]] = temp[index+j]
        # print(Y[chunk],Y_priv[chunk])
        index += len(chunk)
    
    print(Y_priv.shape,flush=True)
        
    return Y_priv


"""Estimates prior using Gaussian dist with private mean & variance for Yarr array. For continuous Y, a discrete subset is created of size bins"""
def ind_prob_dist(Yarr,bins,mean,variance,eps1):

    sd = np.sqrt(variance[0])
    low,up = st.t.interval(0.95, df=len(Yarr)-1, loc=np.mean(mean),  scale=sd)
    print(low,up)
    # low, up = stats.norm.interval(0.99, loc=mean, scale=np.sqrt(np.mean(variance)))
    # low = float(low)
    # up = float(up)
    low = max(np.min(Yarr),low)
    up = min(np.max(Yarr),up)
    
    # low = Yarr.min()
    # up = Yarr.max()

    print(low,up,flush=True)
    
    bin_edges = np.linspace(low,up,bins+1)
    Y_uniq = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    # Y_uniq = np.linspace(low,up,bins)
    print(Y_uniq.shape)
    
    n = len(variance)
    prob_y = np.zeros((n,len(Y_uniq)))
    inter = (Y_uniq.max()-Y_uniq.min())/(len(Y_uniq)-1)

    for i in range(n):
        mean_dp = mean[i]
        var_dp = variance[i]
        
        prob = stats.norm(mean_dp,np.sqrt(var_dp))
        prob_y[i,:] = prob.pdf(Y_uniq)*inter
        prob_y[i,:] = prob_y[i,:]/np.sum(prob_y[i,:])
#     tol = 1e-16
#     prob_y[abs(prob_y) < tol] = 0.0    
    print(prob_y.shape)
    return Y_uniq,prob_y

def opt_rr(args):

    prob_y,Y_uniq,Y,bins,eps,rng, id = args
    n =  Y.shape[0]
    Y_priv = np.zeros(n,dtype=np.float32)

    rand_idx = np.random.randint(n)
    
    for i in range(n):

        loss_mat,y_opt = create_opt_mat(prob_y[i,:],Y_uniq, eps) #opt(prob_y, Y_uniq, eps)
        
        mapping_mat,min_idx = compute_map(bins,loss_mat) #compute_map(bins,L)

        # print(f"min_idx size: {min_idx.shape}")
        
        temp_y, label_cls = get_Yopt(y_opt,mapping_mat,min_idx,bins,eps) #get_Yopt(Yopt,A,min_idx,bins)
        idx = np.abs(Y_uniq-Y[i]).argmin()

        if idx == len(Y_uniq):
            idx=idx-1
        # new_label = temp_y[idx]
        Y_priv[i] =  compute_RR_bins(temp_y[idx],label_cls,eps,rng)

        
    return Y_priv


def create_opt_mat(prob_y, Y_uniq, eps):
    
    k = len(prob_y)
    L = np.ones((k+1,k+1),dtype=np.float32) * np.inf
    Yopt = np.zeros((k+1,k+1),dtype=np.float32)
    # y_keys = list(prob_y.keys())
    
    for r in range(1,k+1):
        t1 = np.sum(prob_y*Y_uniq) 
        t2 = np.sum(prob_y)
        t3 = np.sum(prob_y*(Y_uniq**2))
        for i in range(r,k+1):
            t1 += (prob_y[i-1]*Y_uniq[i-1]*(np.exp(eps)-1))
            t2 += (prob_y[i-1]*(np.exp(eps) - 1))
            t3 += (prob_y[i-1]*(Y_uniq[i-1]**2)*(np.exp(eps)-1))
            
            Yopt[r][i] = t1/t2
            L[r,i] = t3 - (2*t1*Yopt[r][i]) + t2*(Yopt[r][i]**2) 
    
    
    return L,Yopt

def compute_map(bins,L):
    
    A = np.ones((bins+1,bins+1),dtype=np.float32) * np.inf
    A[0,0]=0
    min_idx = np.zeros((bins+1,bins+1), dtype=np.int32)
    for i in range(1,bins+1):
        for j in range(1,i+1):
            arr = [(A[r][j-1]+ L[r+1][i]) for r in range(i)]
            idx = arr.index(min(arr))
            #idx = np.argmin([A[r][j-1]+ L[r+1][i] for r in range(i)])
            #print(idx)
            A[i][j] = A[idx][j-1]+ L[idx+1][i]
            min_idx[i,j] = idx
            
    return A,min_idx

def get_Yopt(Yopt,A,min_idx,bins,eps):
    
    k = bins
    newY = np.zeros(k)
    i = k
    j = np.argmin([A[k][d]/(d-1+np.exp(eps)) for d in range(1,k+1)])+1
    labelCls = np.zeros(j)
    # print(i,j)    
    while (i>0 and j>0):
        idx = min_idx[i,j]
        #print(min_idx[i,j])
        newY[idx:i] = Yopt[idx+1,i]
        labelCls[j-1] = Yopt[idx+1,i]
        i = idx
        j -= 1
    return newY, labelCls        

def compute_RR_bins(old_label,label_cls,eps,rng):
    
    numBins = label_cls.size
    labelIdx = np.where(np.isclose(label_cls, old_label))[0][0]
    rate = 1/(np.exp(eps)+ numBins -1)
    prob = np.zeros(numBins) + rate
    prob[labelIdx]  = 1 - rate * (numBins -1)
    new_idx = rng.choice(numBins,1,p=prob)
    new_label = label_cls[new_idx]
    return new_label
    




    
    
