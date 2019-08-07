"""
Program to step towards cosmological analysis
By creating a slightly more complicated model using real parameters, corresponding data points and performing chi squared analysis
author: Rhys Seeburger
"""

#import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.stats as sci
import time

#start timer
start_time = time.time()

#define function
def func(omega,sigma8,c):
    y = omega**0.5 * sigma8 * c + omega*10**-5.5
    return y

#define chi_2
def chi_2(omega, sigma8, c, y, cov):
    model = func(omega,sigma8,c)
    chi_2 = (np.dot(np.dot((np.asarray(y)-np.asarray(model)),np.linalg.inv(cov)),np.transpose(np.asarray(y)-np.asarray(model))))
    return chi_2

#define s_8
def s_8(omega, sigma8):
    s8 = omega**0.5 * sigma8
    return s8

#define sigma from s8 and omega
def sigma_8(omega,s8):
    sigma8 = s8/(omega**0.5)
    return sigma8

#main body of program

#choose true values for omega and sigma8
omega = 0.3
sigma8 = 0.8
s8 = s_8(omega,sigma8)
std = 1

#create cholesky matrix from file
filename="covariance.ascii"
file=open(filename)
cov=np.loadtxt(file,comments='#')
L = np.linalg.cholesky(cov)

#create datapoints from file
filename="input_function.ascii"
file=open(filename)
input=np.loadtxt(file,comments='#')

theta = input[:,0]
c_raw = input[:,1]
y_model = func(omega,sigma8,c_raw)

#add errors using cholesky decomposition
length = 9
nReal = 10000

err = np.dot(L,np.random.normal(0.,1,(length,nReal)))
y_plot_arr=np.zeros((length,nReal))
for iReal in range(nReal):
    y_plot_arr[:,iReal] = err[:,iReal] + y_model

#create plot
plt.plot(theta,y_model,"r-",label="model")
plt.xscale("log")
plt.yscale("log")
plt.errorbar(theta,y_plot_arr[:,0],yerr=np.sqrt(np.diag(cov)),ecolor='b',
				fmt='d',markeredgecolor='b',mew=1,markerfacecolor='none',  
					markersize=6,label='data')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\xi_+$")
plt.title(r"$\xi_+$" +" vs "+ r"$\theta$")
plt.legend()
plt.show()
# plt.savefig('Plots/data.pdf',bbox_inches='tight')  


#now apply chi_2 with sigma8

#set up a grid of om and sig8 values to step around
steps = 100

om_t = np.repeat(np.linspace(0.0,0.99,steps),steps)
sig8_t = np.tile(np.linspace(0.0,0.99,steps),steps)
chis = []

#calculate chi2 at each point of the grid
for i in range(0,steps**2):
    chi = chi_2(om_t[i], sig8_t[i], c_raw, y_plot_arr[:,0], cov)
    chis.append(chi)

#convert into correct format for chainconsumer
chisq = np.array(chis).reshape(steps,steps)

om = np.array(om_t[0::steps])
sig8 = np.array(sig8_t[0:steps])
pdf = np.array(np.exp(-0.5*chisq))

#plot using chainconsumer
c = ChainConsumer()
c.add_chain([om,sig8], parameters=[r"$\Omega_m$", r"$\sigma_8$"], weights=pdf, grid=True)
fig = c.plotter.plot(display=True,truth=[omega,sigma8])


"""
#now apply chi_2 with s8

#set up a grid of om and s8 values to step around
steps = 100

om_t = np.repeat(np.linspace(0.01,0.7,steps),steps)
s8_t = np.tile(np.linspace(s_8(omega,0.3),s_8(omega,1.3),steps),steps)
sig8_t = []
chis = []

#calculate chi2 at each point of the grid
for i in range(0,steps**2):
    sig8 = sigma_8(om_t[i],s8_t[i])
    chi = chi_2(om_t[i], sig8, c_raw, y_plot_arr[:,0], cov)
    sig8_t.append(sig8)
    chis.append(chi)

#convert into correct format for chainconsumer
chisq = np.array(chis).reshape(steps,steps)

om = np.array(om_t[0::steps])
ess8 = np.array(s8_t[0:steps])
pdf = np.array(np.exp(-0.5*chisq))

#plot using chainconsumer
c = ChainConsumer()
c.add_chain([om,ess8], parameters=[r"$\Omega_m$", r"$S_8$"], weights=pdf, grid=True)
fig = c.plotter.plot(display=True,truth=[omega,s8])
"""


#find minimum chi_2 and omega and sig8 at that chi_2
chimin = np.argmin(chis)
omin = om_t[chimin]
sigmin = sig8_t[chimin]

print("omin is ", omin)
print("sigmin is ", sigmin)

"""
#test to see if min chisquared was computed correctly
kaisq = chi_2(omin,sigmin,c_raw,y_plot_arr[:,0],cov)
print(kaisq)
print(np.amin(chis))
"""


#calculate y values for this omin and sigmin and plot

y_chimin =  func(omin,sigmin,c_raw)

plt.plot(theta,y_model,"r:",label="model")
plt.plot(theta,y_chimin,"g-",label="min chisq")
plt.xscale("log")
plt.yscale("log")
plt.errorbar(theta,y_plot_arr[:,0],yerr=np.sqrt(np.diag(cov)),ecolor='b',
				fmt='d',markeredgecolor='b',mew=1,markerfacecolor='none',  
					markersize=6,label='data')
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\xi_+$")
plt.title(r"$\xi_+$" +" vs "+ r"$\theta$")
plt.legend()
plt.show()


#chi_2 function plot

#compute chisquared for the realisations
x = np.linspace(0.,30.,num=100)

y_chis =[]
for i in range(0,nReal):
    y_chi = chi_2(omega, sigma8, c_raw, y_plot_arr[:,i], cov)
    y_chis.append(y_chi)

y_chisq = np.array(y_chis)

#plot this
plt.hist(y_chisq,bins=100,normed=1,histtype='stepfilled', label="data")
plt.plot(x,sci.chi2.pdf(x,7),"k--",label="7 dof")
plt.plot(x,sci.chi2.pdf(x,8),ls="--",color="orange",label="8 dof")
plt.plot(x,sci.chi2.pdf(x,9),ls="--",color="r",label="9 dof")
plt.legend()
plt.ylim(0,0.15)
plt.show()

#timekeeping
print("My program took", time.time() - start_time, "s to run")
