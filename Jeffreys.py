"""
Plotting a range of (Jeffreys) Priors
Author: Rhys Seeburger
"""

#import relevant packages
import numpy as np
import sympy as sym
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
import time
import tqdm
start_time = time.time()

#set up data
parameters=2
length = 9
nReal = 100

#define conditions for MCMC
ndim, nwalkers, steps, burn = 2, 10, 200, 10

#define functions
def funcS8(omega,s8,c):
    y = s8 * c + omega*10**-6
    return y
def func(omega,sigma8,c):
    y = omega**0.5 * sigma8 * c + omega*10**-6
    return y
def funcLog(omega,A,c):
    y = omega**0.5 * sym.exp(A) * c + omega*10**-6
    return y

#define Fisher matrix
def Fisher(derivative,covariance):
    I = sym.transpose(derivative)*np.linalg.inv(covariance)*derivative
    return I

#define how Jeffreys prior is computed
def JeffDiag(Fisher): 
    diag_prod = Fisher[0,0] * Fisher[1,1]
    J = sym.sqrt(diag_prod)
    return J
def JeffDet(Fisher):
    diag_prod = Fisher[0,0] * Fisher[1,1]
    anti_prod = Fisher[1,0] * Fisher[0,1]
    det = diag_prod - anti_prod
    J = sym.sqrt(det)
    return J

#import covariance matrix from file
filename="covariance.ascii"
file=open(filename)
cov=np.loadtxt(file,comments='#')/3.

#create datapoints from file
filename="input_function.ascii"
file=open(filename)
input=np.loadtxt(file,comments='#')

theta = input[:,0]
c_raw = input[:,1]

#apply function to c_raw values
y_model_s8 = funcS8(0.3,0.8*np.sqrt(0.3),c_raw)
y_model_log = funcLog(0.3,np.log(0.8),c_raw)
y_model_sig = func(0.3,0.8,c_raw)

#generate data with error using cholesky decomposition
L = np.linalg.cholesky(cov)
err = np.dot(L,np.random.normal(0.,1,(length,nReal)))
y_arr_s8=np.zeros((length,nReal))
y_arr_sig=np.zeros((length,nReal))
y_arr_log=np.zeros((length,nReal))
for iReal in range(nReal):
    y_arr_s8[:,iReal] = err[:,iReal] + y_model_s8
    y_arr_sig[:,iReal] = err[:,iReal] + y_model_sig
    y_arr_log[:,iReal] = err[:,iReal] + y_model_log

#define symbols and functions in sympy
om, sig8, s8, A = sym.symbols("om sig8 s8 A")
function = func(om,sig8,c_raw)
functionS8 = funcS8(om, s8, c_raw)
functionLog = funcLog(om, A, c_raw)

#initialise derivative matrices
der_sig = sym.zeros(length,parameters)
der_s8 = sym.zeros(length,parameters)
der_log = sym.zeros(length,parameters)

#fill with derviative values
der_sig[:,0] = sym.diff(function,sig8)
der_sig[:,1] = sym.diff(function,om)
der_s8[:,0] = sym.diff(functionS8,s8)
der_s8[:,1] = sym.diff(functionS8,om)
der_log[:,0] = sym.diff(functionLog,A)
der_log[:,1] = sym.diff(functionLog,om)

#calculate Fisher Matrices
Fisher_sig = sym.simplify(Fisher(der_sig,cov))
Fisher_s8 = sym.simplify(Fisher(der_s8,cov))
Fisher_log = sym.simplify(Fisher(der_log,cov))

#calculate both Jeffreys priors for each Fisher Matrix
JeffDiag_sig = JeffDiag(Fisher_sig)
JeffDet_sig = JeffDet(Fisher_sig)
JeffDiag_s8 = JeffDiag(Fisher_s8)
JeffDet_s8 = JeffDet(Fisher_s8)
JeffDiag_log = JeffDiag(Fisher_log)
JeffDet_log = JeffDet(Fisher_log)

#Plotting

#define function to evaluate Jeffreys prior
def prior(omega,alp,Jeff):
    return float(Jeff.evalf(subs={om:omega, A:alp}))

#set up grid for sig8 and omega_m values
steps = 100

om_t = np.repeat(np.linspace(0.01,0.99,steps),steps)
logsig8_t = np.tile(np.linspace(-0.99,-0.01,steps),steps)
Jeffrey = []

#step around and calculate value of Jeffreys at each point
print("Calculating")
for i in range(0,steps**2):
    Jeff_ = prior(om_t[i],logsig8_t[i],JeffDet_log)
    Jeffrey.append(Jeff_)
print("Done")

#convert into plottable format
JeffP = np.array(Jeffrey).reshape(steps,steps)
ome = np.array(om_t[0::steps])
logsigm8 = np.array(logsig8_t[0:steps])

#plot Jeffreys prior
c = ChainConsumer()
c.add_chain([ome,logsigm8], parameters=[r"$\Omega_m$", r"$log(\sigma_8)$"], weights=JeffP, grid=True)
fig = c.plotter.plot(display=True,truth=[0.3,np.log(0.8)])


#Now, for the posterior
import emcee

#define flat prior
Flat = 1.0

#define prior
def logprior(x, prior):
    omega, param_2 = x
    if 0. <= omega <= 1.:
        if isinstance(prior,float):
            return np.log(prior)
        else:
            return float(sym.ln(prior.evalf(subs={om:omega, sig8:param_2, s8:param_2, A:param_2})))
    else:
        return -np.inf

#define log likelihood
def like(x, data, cov, length, c_raw, fn):
    omega, param_2 = x
    first = np.log(np.linalg.det(cov))
    second = length * np.log(2 * np.pi)
    third = (np.dot(np.dot((np.transpose(np.asarray(data)-np.asarray(fn(omega, param_2, c_raw)))),np.linalg.inv(cov)),np.asarray(data)-np.asarray(fn(omega, param_2, c_raw))))
    return -0.5 * (first + second + third)

#define log posterior
def post(x, data, prior, cov, length, c_raw, fn, condition):
    cond = condition(x)
    lp = logprior(x, prior)
    if not np.isfinite(cond):
        return -np.inf
    return lp + like(x, data, cov, length, c_raw, fn)

#define limiting conditions
def condlog(x):
    omega, logsig = x
    if 0. <= omega <= 1. and 0. <= np.exp(logsig) <= 1.5:
        return 1
    return -np.inf
def conds8(x):
    omega, s8 = x
    if 0. <= omega <= 1. and 0. <= s8 / omega**0.5 <= 1.5:
        return 1
    return -np.inf
def cond(x):
    omega, sig8 = x
    if 0. <= omega <= 1. and 0. <= sig8 <= 1.5:
        return 1
    return -np.inf

#setup

#set up chains
sig_chain_JDiag = np.zeros((1,2))
sig_chain_JDet = np.zeros((1,2))
sig_chain_F = np.zeros((1,2))
#s8_chain_JDiag = np.zeros((1,2))
#s8_chain_JDet = np.zeros((1,2))
s8_chain_F = np.zeros((1,2))
log_chain_JDiag = np.zeros((1,2))
log_chain_JDet = np.zeros((1,2))
log_chain_F = np.zeros((1,2))

#run mcmc

for i in range(0,nReal):

    #set up samplers
    number = i+1

    #s8_sampler_JDiag = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_s8[:,i], JeffDiag_s8, cov, length, c_raw, funcS8, conds8))
    #s8_sampler_JDet = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_s8[:,i], JeffDet_s8, cov, length, c_raw, funcS8, conds8))
    s8_sampler_F = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_s8[:,i], Flat, cov, length, c_raw, funcS8, conds8))

    log_sampler_JDiag = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_log[:,i], JeffDiag_log, cov, length, c_raw, funcLog, condlog))
    log_sampler_JDet = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_log[:,i], JeffDet_log, cov, length, c_raw, funcLog, condlog))
    log_sampler_F = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_log[:,i], Flat, cov, length, c_raw, funcLog, condlog))

    sig_sampler_JDiag = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_sig[:,i], JeffDiag_sig, cov, length, c_raw, func, cond))
    sig_sampler_JDet = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_sig[:,i], JeffDet_sig, cov, length, c_raw, func, cond))
    sig_sampler_F = emcee.EnsembleSampler(nwalkers, ndim, post, args=(y_arr_sig[:,i], Flat, cov, length, c_raw, func, cond))

    #initialise and run mcmc
    p0_s8 = [[0.3,0.8*0.3**0.5] + 0.01*np.random.randn(ndim) for i in range(nwalkers)]
    #s8_sampler_JDiag.run_mcmc(p0_s8,steps)
    #s8_sampler_JDet.run_mcmc(p0_s8,steps)
    s8_sampler_F.run_mcmc(p0_s8,steps)

    p0_log = [[0.3,np.log(0.8)] + 0.01*np.random.randn(ndim) for i in range(nwalkers)]
    log_sampler_JDiag.run_mcmc(p0_log,steps)
    log_sampler_JDet.run_mcmc(p0_log,steps)
    log_sampler_F.run_mcmc(p0_log,steps)

    p0_sig = [[0.3,0.8] + 0.01*np.random.randn(ndim) for i in range(nwalkers)]
    sig_sampler_JDiag.run_mcmc(p0_sig,steps)
    sig_sampler_JDet.run_mcmc(p0_sig,steps)
    sig_sampler_F.run_mcmc(p0_sig,steps)

    #some timekeeping to allow user to assess if they want to continue or restart with different parameters
    print("MCMC run " + str(number) + " done.")
    print(str(round(time.time()-start_time)) + " s elapsed, " + str((round(start_time-time.time()+nReal*(time.time()-start_time)/number))) + " s remaining")

    #append values from MCMC into chains
    sig_chain_JDiag = np.concatenate((sig_chain_JDiag, sig_sampler_JDiag.flatchain), axis = 0)
    sig_chain_JDet = np.concatenate((sig_chain_JDet, sig_sampler_JDet.flatchain), axis = 0)
    sig_chain_F = np.concatenate((sig_chain_F, sig_sampler_F.flatchain), axis = 0)
    #s8_chain_JDiag = np.concatenate((s8_chain_JDiag, s8_sampler_JDiag.flatchain), axis = 0)
    #s8_chain_JDet = np.concatenate((s8_chain_JDet, s8_sampler_JDet.flatchain), axis = 0)
    s8_chain_F = np.concatenate((s8_chain_F, s8_sampler_F.flatchain), axis = 0)
    log_chain_JDiag = np.concatenate((log_chain_JDiag, log_sampler_JDiag.flatchain), axis = 0)
    log_chain_JDet = np.concatenate((log_chain_JDet, log_sampler_JDet.flatchain), axis = 0)
    log_chain_F = np.concatenate((log_chain_F, log_sampler_F.flatchain), axis = 0)

#delete first row (of zeros)
sig_chain_F = np.delete(sig_chain_F, (0), axis=0)
sig_chain_JDiag = np.delete(sig_chain_JDiag, (0), axis=0)
sig_chain_JDet = np.delete(sig_chain_JDet, (0), axis=0)
#s8_chain_JDiag = np.delete(s8_chain_JDiag, (0), axis=0)
#s8_chain_JDet = np.delete(s8_chain_JDet, (0), axis=0)
s8_chain_F = np.delete(s8_chain_F, (0), axis=0)
log_chain_JDiag = np.delete(log_chain_JDiag, (0), axis=0)
log_chain_JDet = np.delete(log_chain_JDet, (0), axis=0)
log_chain_F = np.delete(log_chain_F, (0), axis=0)

#transform s8 to sigma8
def s8_sigma(samples):
    sam = np.zeros(samples.shape)
    sam[:,0] = samples[:,0]        
    sam[:,1] = samples[:,1]/np.sqrt(samples[:,0])
    return sam #samples[samples[:,1]<=1.]

#sig_s8_JDiag = s8_sigma(s8_chain_JDiag)
#sig_s8_JDet = s8_sigma(s8_chain_JDiag)
sig_s8_F = s8_sigma(s8_chain_F)

#transform sigma8 to s8
def sigma_s8(samples):
    sam = np.zeros(samples.shape)
    sam[:,0] = samples[:,0]        
    sam[:,1] = samples[:,1]*np.sqrt(samples[:,0])
    return sam #samples[samples[:,1]<=1.]

s8_sig_JDiag = sigma_s8(sig_chain_JDiag)
s8_sig_JDet = sigma_s8(sig_chain_JDet)
s8_sig_F = sigma_s8(sig_chain_F)

#transform log to sigma8
def log_sigma(samples):
    sam = np.zeros(samples.shape)
    sam[:,0] = samples[:,0]        
    sam[:,1] = np.exp(samples[:,1])
    return sam #samples[samples[:,1]<=1.]

sig_log_JDiag = log_sigma(log_chain_JDiag)
sig_log_JDet = log_sigma(log_chain_JDet)
sig_log_F = log_sigma(log_chain_F)

#transform sigma8 to log
def sigma_log(samples):
    sam = np.zeros(samples.shape)
    sam[:,0] = samples[:,0]        
    sam[:,1] = np.log(samples[:,1])
    return sam #samples[samples[:,1]<=1.]

log_sig_JDiag = sigma_log(sig_chain_JDiag)
log_sig_JDet = sigma_log(sig_chain_JDet)
log_sig_F = sigma_log(sig_chain_F)

#plot in sigma

#plot flats
csigf = ChainConsumer()

csigf.add_chain(sig_chain_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 F$")
csigf.add_chain(sig_s8_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$S_8 F$")
csigf.add_chain(sig_log_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) F$")

#plot jeff
csigj = ChainConsumer()

csigj.add_chain(sig_chain_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDiag$")
csigj.add_chain(sig_chain_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDet$")
csigj.add_chain(sig_s8_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$S_8 F=S_8 J$")
csigj.add_chain(sig_log_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) JDiag$")
csigj.add_chain(sig_log_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) JDet$")

#plot logs in sigma
csiglog = ChainConsumer()

csiglog.add_chain(sig_chain_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDiag$")
csiglog.add_chain(sig_chain_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDet$")
csiglog.add_chain(sig_chain_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 F$")
csiglog.add_chain(sig_log_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) JDiag$")
csiglog.add_chain(sig_log_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) JDet$")
csiglog.add_chain(sig_log_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) F$")

#plot s8 in sigma
csigs8 = ChainConsumer()

csigs8.add_chain(sig_chain_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDiag$")
csigs8.add_chain(sig_chain_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDet$")
csigs8.add_chain(sig_chain_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 F$")
csigs8.add_chain(sig_s8_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$S_8 F$")

#plot all (messy)
csig = ChainConsumer()

csig.add_chain(sig_chain_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDiag$")
csig.add_chain(sig_chain_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 JDet$")
csig.add_chain(sig_chain_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$\sigma_8 F$")

#csig.add_chain(sig_s8_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$S_8 JDiag$")
#csig.add_chain(sig_s8_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$S_8 JDet$")
csig.add_chain(sig_s8_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$S_8 F$")

#csig.add_chain(sig_log_JDiag, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) JDiag$")
csig.add_chain(sig_log_JDet, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) JDet$")
csig.add_chain(sig_log_F, parameters=[r"$\Omega_m$", r"$\sigma_8$"], name = r"$log(\sigma_8) F$")

#plot in s8
cs8 = ChainConsumer()

#cs8.add_chain(s8_chain_JDiag, parameters=[r"$\Omega_m$", r"$S_8$"], name = r"$S_8 JDiag$")
#cs8.add_chain(s8_chain_JDet, parameters=[r"$\Omega_m$", r"$S_8$"], name = r"$S_8 JDet$")
cs8.add_chain(s8_chain_F, parameters=[r"$\Omega_m$", r"$S_8$"], name = r"$S_8 F$")

cs8.add_chain(s8_sig_JDiag, parameters=[r"$\Omega_m$", r"$S_8$"], name = r"$\sigma_8 JDiag$")
cs8.add_chain(s8_sig_JDet, parameters=[r"$\Omega_m$", r"$S_8$"], name = r"$\sigma_8 JDet$")
cs8.add_chain(s8_sig_F, parameters=[r"$\Omega_m$", r"$S_8$"], name = r"$\sigma_8 F$")

#plot in log
clog = ChainConsumer()

clog.add_chain(log_chain_JDiag, parameters=[r"$\Omega_m$", r"$log(\sigma_8)$"], name = r"$log(\sigma_8) JDiag$")
clog.add_chain(log_chain_JDet, parameters=[r"$\Omega_m$", r"$log(\sigma_8)$"], name = r"$log(\sigma_8) JDet$")
clog.add_chain(log_chain_F, parameters=[r"$\Omega_m$", r"$log(\sigma_8)$"], name = r"$log(\sigma_8) F$")

clog.add_chain(log_sig_JDiag, parameters=[r"$\Omega_m$", r"$log(\sigma_8)$"], name = r"$\sigma_8 JDiag$")
clog.add_chain(log_sig_JDet, parameters=[r"$\Omega_m$", r"$log(\sigma_8)$"], name = r"$\sigma_8 JDet$")
clog.add_chain(log_sig_F, parameters=[r"$\Omega_m$", r"$log(\sigma_8)$"], name = r"$\sigma_8 F$")

#config
csig.configure(kde=1.7)
csigf.configure(kde=1.7)
csigj.configure(kde=1.7)
csiglog.configure(kde=1.7)
csigs8.configure(kde=1.7)
cs8.configure(kde=1.7)
clog.configure(kde=1.7)

#plot
fig = csig.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_all_cov3.pdf")#,extents=[[0., 1.], [0.3, 1.]])
fig = csigf.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_flat_cov3.pdf")#,extents=[[0., 1.], [0.3, 1.]])
fig = csigj.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_jeff_cov3.pdf")#,extents=[[0., 1.], [0.3, 1.]])
fig = csiglog.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_log_cov3.pdf")#,extents=[[0., 1.], [0.3, 1.]])
fig = csigs8.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_s8_cov3.pdf")#,extents=[[0., 1.], [0.3, 1.]])
fig = cs8.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8*0.3**0.5], filename = str(nReal) + "_s8_cov3.pdf")#,extents=[[0., 1.], [0.1, 0.7]])
fig = clog.plotter.plot(figsize = "column", display=False,truth=[0.3,np.log(0.8)], filename = str(nReal) + "_log_cov3.pdf")#,extents=[[0., 1.], [-1., 0.0]])

fig = csig.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_all_cov3.png")#,extents=[[0., 1.], [0.3, 1.]])
fig = csigf.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_flat_cov3.png")#,extents=[[0., 1.], [0.3, 1.]])
fig = csigj.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_jeff_cov3.png")#,extents=[[0., 1.], [0.3, 1.]])
fig = csiglog.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_log_cov3.png")#,extents=[[0., 1.], [0.3, 1.]])
fig = csigs8.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8], filename = str(nReal) + "_sig_s8_cov3.png")#,extents=[[0., 1.], [0.3, 1.]])
fig = cs8.plotter.plot(figsize = "column", display=False,truth=[0.3,0.8*0.3**0.5], filename = str(nReal) + "_s8_cov3.png")#,extents=[[0., 1.], [0.1, 0.7]])
fig = clog.plotter.plot(figsize = "column", display=False,truth=[0.3,np.log(0.8)], filename = str(nReal) + "_log_cov3.png")#,extents=[[0., 1.], [-1., 0.0]])

#timekeeping
print("My program took", time.time() - start_time, "s to run")