"""
Program to create a prior by iterating over a range of datasets
Author: Rhys Seeburger, utilising code provided by Dan Foreman-Mackey
"""

#import relevant packages
import numpy as np
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
import time
start_time = time.time()

#define functions
def func(omega,sigma8,c):
    y = omega**0.5 * sigma8 * c + omega*10**-5.5
    return y
def funcS(omega,s8,c):
    y = s8 * c + omega*10**-6
    return y

#define initial parameters
parameters=2
sigma8 = 0.8
omega = 0.3

#import covariance matrix from file
filename="covariance.ascii"
file=open(filename)
cov=np.loadtxt(file,comments='#')

#create datapoints from file
filename="input_function.ascii"
file=open(filename)
input=np.loadtxt(file,comments='#')

theta = input[:,0]
c_raw = input[:,1]
y_model = func(0.3,0.8,c_raw)

#generate data, adding error using cholesky decomposition
length = 9
nReal = 10

L = np.linalg.cholesky(cov)
err = np.dot(L,np.random.normal(0.,1,(length,nReal)))
y_arr=np.zeros((length,nReal))
for iReal in range(nReal):
    y_arr[:,iReal] = err[:,iReal] + y_model

#now for the mcmc, using code by DFM
#this takes all available datasets, computing likelihood, and hence, posterior
import emcee

factor = cho_factor(cov)
logdet = 2*np.sum(np.log(np.diag(factor[0])))
loglike0 = logdet + cov.shape[0] * np.log(2*np.pi)

def like(x, datasets, loglike0, factor, length, c_raw):
    ndata = datasets.shape[1]
    omega, sigma8 = x
    func_val = np.asarray(func(omega, sigma8, c_raw))
    ll = ndata * loglike0
    for i in range(ndata):
        resid = datasets[:, i] - func_val
        ll += np.dot(resid, cho_solve(factor, resid))

    return -0.5 * ll

def post(x, datasets, logprior, loglike0, factor, length, c_raw):
    lp = logprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + like(x, datasets, loglike0, factor, length, c_raw)

def flat(x):
    om, sig = x
    if 0. <= om <= 1. and 0. <= sig <= 1.:
        return 0.
    return -np.inf

#set up mcmc conditions
ndim, nwalkers, steps, burn = 2, 10, 100, 10

#initialise mcmc
p0 = [[0.3,0.8] + 0.01*np.random.randn(ndim) for i in range(nwalkers)]

#run mcmc, printing mean acceptance fraction
sampler0 = emcee.EnsembleSampler(nwalkers, ndim, post, a = 5, args=(y_arr, flat, loglike0, factor, length, c_raw))
pos, prob, state = sampler0.run_mcmc(p0, burn)
sampler0.reset()
sampler0.run_mcmc(pos, steps)
print(" Flat Mean acceptance fraction: {0:.3f}".format(np.mean(sampler0.acceptance_fraction)))

#plot posterior using chainconsumer
c = ChainConsumer()
c.add_chain(sampler0.flatchain, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = "dan")
c.configure(kde=1.7)
#fig = c.plotter.plot_walks(display=True,truth=[0.3,0.8])
fig = c.plotter.plot(display=True,truth=[0.3,0.8], filename = "dan.png")
plt.close()


#timekeeping
print("My program took", time.time() - start_time, "s to run")