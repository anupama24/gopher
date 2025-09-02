###############################
##Some commonly used methods
###############################

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


"""Generate randomized response for x with privacy budget epsilon"""
def rand_response(x,xc,eps: float,rng):

    e = np.exp(eps)
    prob = [e/(1.0+e),1.0/(1.0+e)]
    idx = rng.choice(2,1,p=prob)
    if idx == 0:
        return x
    else:
        return xc
    
    
"""Generate Laplace noise with scale = sensitivity/epsilon. Returns noise vector of length size"""
def laplace_noise(sensitivity: float, epsilon: float, size: int = 1):
    scale = sensitivity/epsilon
    return np.random.laplace(loc=0,scale=scale, size=size)

"""Generate Gaussian noise with mean 0 and variance sensitivity^2/epsilon^2 (approx). Returns noise vector of length size """
def gauss_noise(sensitivity: float, epsilon: float, delta: float, size: int = 1):
    scale = np.sqrt(2 * np.power(sensitivity,2) * np.log(1.25 / delta) / np.power(epsilon,2))
    return np.random.normal(loc=0,scale=scale, size=size)

"""Estimates correlation between standardized x and y. Optinal input: mean and std of y"""
def get_corr(x, y, y_mean = None, y_std = None):
    y_mean = y_mean if y_mean is not None else y.mean()
    y_std = y_std if y_std is not None else y.std()
    y_stand = (y - y_mean) / y_std
    x_stand = (x - x.mean(axis=0)) / x.std(axis=0)
    x_stand[np.isnan(x_stand)] = 0
    
    r = x_stand.T @ y_stand
    r /=  x.shape[0]
    return r 

"""Estimate p-value for statistic with n samples and c covariates"""
def log_pv(r, n,c=1.0):
    #return r
    t2 = np.power(r, 2) * (n - c) / (1.0 - np.power(r, 2))
    pv = stats.distributions.chi2.sf(t2, df=1)
    pvlog = -np.log10(pv)
    return pvlog

"""Read all lines as a list from file"""
def read_list(file_name):
    
    f = open(file_name,"r")
    out_list=[]
    for line in f:
        out_list.append(line.strip('\n'))
    # print (out_list)
    return out_list



"""Plots for different value of epsilons"""
def exp_plot(df,plot_label,eps_all,rho_all=None,fname=None):
    
    plt.rcParams['figure.figsize'] = [35/2.54, 35/2.54]

    #colors = cm.rainbow(np.linspace(0, 1, len(eps_all)))
    a = len(eps_all)//2
    row = 1 if a==0 else a
    
    fig, axs = plt.subplots(row, 2, sharey=False, sharex=False)
    for i, (ax, eps) in enumerate(zip(axs.flatten(), eps_all)):
        x = df["original"]
        y1 = df.iloc[:, i]
        rmse = np.sqrt(((y1 - x)**2).mean())
        ax.plot(x,x,label='perfect accuracy')
        if rho_all is not None:
            ax.scatter(x, y1,label= f"$\\rho=${rho_all[i]}, rmse = {rmse}")
        else:
            ax.scatter(x, y1, label = f"$\\varepsilon=${eps}, rmse = {rmse}")
        #ax.scatter(x, y2, c="C1", label="Y pert")
        ax.set_ylim(min(x.min(),y.min()), max(x.max(),y.max()))
        ax.set_xlim((x.min(), x.max()))
        ax.set_title(f"$\\varepsilon=${eps}")
        ax.legend()
    if row == 1:
        axs[row-1].set_ylabel("DP correlation")
        axs[0].set_xlabel("Ground truth correlation")
        axs[1].set_xlabel("Ground truth correlation")
    else:
        for i in range(row):
            axs[i,0].set_ylabel("DP correlation")
        axs[row-1,0].set_xlabel("Ground truth correlation")
        axs[row-1,1].set_xlabel("Ground truth correlation")
    plt.suptitle("Correlations between LDL and Chromosome 19 SNPs - {} ($M={}$)".format(plot_label,y1.shape[0]))
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
    

"""Plots for different value of samples"""
def exp_plot_sample(df,plot_label,var_all,eps,fname=None):
    
    plt.rcParams['figure.figsize'] = [35/2.54, 35/2.54]

    #colors = cm.rainbow(np.linspace(0, 1, len(eps_all)))
#     row =len(var_all)
    a = len(var_all)//2
    row = 1 if a==0 else a
    
    
    fig, axs = plt.subplots(row, 2, sharey=False, sharex=False)
    for i, (ax, var) in enumerate(zip(axs.flatten(), var_all)):
        
        x = df[f"orig{var}"]
        y1 = df.iloc[:, 2*i]
        rmse1 = np.sqrt(((y1 - x)**2).mean())
        
        ax.plot(x,x,label='perfect accuracy')
        
        ax.scatter(x, y1, label = f"$n=${var}, rmse = {rmse1}")
        ax.set_ylim((x.min(), x.max()))
        ax.set_xlim((x.min(), x.max()))
        
        ax.set_title(f"$n=${var}")
        ax.legend()
        ax.set_ylabel("DP correlation")
        ax.set_xlabel("Ground truth correlation")
    
        
    plt.suptitle("Correlations between LDL and Chromosome 19 SNPs - {} ($\\varepsilon={}$)".format(plot_label,y1.shape[0],eps))
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
    


# """Plots manhattan plot"""
# def manhattan_plot(vals, sites, ax, n_sites = 10, ref_line = None):
    
#     ax = sns.barplot(x=sites, y=vals, ax=ax)
#     ax.set_ylabel("$\\log_{10}(P)$")
    

#     # plot x-ticks of certain positions
#     xticklabels = ax.get_xticklabels()
#     xticks = ax.get_xticks()
#     idxs = np.linspace(0, len(sites)-1, n_sites)
#     idxs = idxs.astype(int)
    
#     xticklabels = [sites[i] for i in idxs]
#     xticks = [xticks[i] for i in idxs]
#     ax.set_xticks(xticks)
#     ax.set_xticklabels(xticklabels, rotation=-90, fontsize=8)
#     ax.set_xlabel("Position")

#     if ref_line is not None:
#         ax.axhline(ref_line, color="r")

#     return ax




