"""
Program to create a prior by iterating over a range of datasets
Author: Rhys Seeburger
"""

#import relevant packages
import numpy as np
import sympy as sym
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
import time
import sys

#get filename from command line
outfile = sys.argv[1]

#start timer
start_time = time.time()

#define functions
def func(omega,sigma8,c):
    y = omega**0.5 * sigma8 * c + omega*10**-20
    return y
def funcS(omega,s8,c):
    y = s8 * c + omega*10**-6
    return y

#define initial parameters
parameters=2
sigma8 = 0.8
omega = 0.3

#get conditions for MCMC from terminal
ndim = 2
nwalkers = int(input("number of walkers: "))
steps = int(input("number of steps: "))
burn = int(input("number of burn steps: "))

length = 9
nReal = int(input("number of realisations: "))


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

#generate data, adding error using cholesky decomp
L = np.linalg.cholesky(cov)
err = np.dot(L,np.random.normal(0.,1,(length,nReal)))
y_arr=np.zeros((length,nReal))
for iReal in range(nReal):
    y_arr[:,iReal] = err[:,iReal] + y_model

#Now for the MCMC part
import emcee

#define log likelihood
def like(x, data, cov, length, c_raw):
    omega, sigma8 = x
    first = np.log(np.linalg.det(cov))
    second = length * np.log(2 * np.pi)
    third = (np.dot(np.dot((np.transpose(np.asarray(data)-np.asarray(func(omega, sigma8, c_raw)))),np.linalg.inv(cov)),np.asarray(data)-np.asarray(func(omega, sigma8, c_raw))))
    return -0.5 * (first + second + third)

#define posterior for initial mcmc run
def post(x, data, logprior, cov, length, c_raw):
    lp = logprior(x)
    if not np.isfinite(lp):
        return -np.inf
    return lp + like(x, data, cov, length, c_raw)

#define posterior for subsequent run
def post_new(x, data, logprior, cov, length, c_raw):
    om, sig = x
    if 0. <= om <= 1. and 0. <= sig <= 1.:
        lp = logprior.lnprobfn(x)
        return lp + like(x, data, cov, length, c_raw)
    else:
        return -np.inf

#define flat prior for initial run
def flat(x):
    om, sig = x
    if 0. <= om <= 1. and 0. <= sig <= 1.:
        return 0.
    return -np.inf

#set up samplet
p0 = [[0.3,0.8] + 0.01*np.random.randn(ndim) for i in range(nwalkers)]

#initial run
sampler0 = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr[:,0], flat, cov, length, c_raw))
pos, prob, state = sampler0.run_mcmc(p0, burn)
sampler0.reset()
sampler0.run_mcmc(pos, steps)
print("iteration 1 of " + str(nReal) + " complete")

#set the sampler to be the new prior
prior_new = sampler0


"""
#plot this initial posterior/new prior
c = ChainConsumer()
c.add_chain(sampler0.flatchain, parameters=[r"$\Omega_m$", r"$\sigma_8$"])
c.configure(kde=1.7)
#fig = c.plotter.plot_walks(display=True,truth=[0.3,0.8])
fig = c.plotter.plot(display=True,truth=[0.3,0.8], filename = outfile + "_iterate_0.png")
plt.close()
"""

#iterate, using posterior from previous iteration as new prior
for i in range(1,nReal):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, post_new, args=(y_arr[:,0], prior_new, cov, length, c_raw))
    pos, prob, state = sampler.run_mcmc(p0, burn)
    sampler.reset()
    sampler.run_mcmc(p0, steps)
    prior_new = sampler
    print("iteration " + str(i+1) +  " of " + str(nReal) + " complete")


#plot initial and final posteriors using chainconsumer
c = ChainConsumer()
c.add_chain(sampler.flatchain, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = str(nReal)+"th")
c.add_chain(sampler0.flatchain, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = "First")
c.configure(kde=1.7)
fig = c.plotter.plot(display=True,truth=[0.3,0.8], filename = outfile + "_" + str(nReal) + "_iterate.png")
fig = c.plotter.plot_summary(display=True,truth=[0.3,0.8], filename = outfile + "_" + str(nReal) + "_iterate_summary.png")

#timekeeping
print("My program took", time.time() - start_time, "s to run")