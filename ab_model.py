"""
Program to familiarise myself with python data analysis
by creating a linear model, corresponding data points and performing chi squared analysis
author: Rhys Seeburger
"""

#import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.stats as sci

#define function
def func(alpha,beta,x):
    y = alpha * np.asarray(x) + beta
    return y

#define chi2 with covariance
def chi2(alpha, beta, x, y, cov):
    model = func(alpha,beta,x)
    chi2 = (np.dot(np.dot((np.asarray(y)-np.asarray(model)),np.linalg.inv(cov)),np.transpose(np.asarray(y)-np.asarray(model))))
    return chi2

#define chi2 without covariance
def chi_2(alpha, beta, x, y, error):
    model = func(alpha,beta,x)
    chi_2 = np.sum((np.asarray(y)-np.asarray(model))**2/np.asarray(error))
    return chi_2

#main body of program

#choose true values for alpha and beta
alpha = 7.83
beta = 115.3
sigma = 2

#create datapoints
x_plot = []
y_plot = []
y_data = []
y_model = []

for i in range (0,10):
    x_val = float(i+1)
    x_plot.append(x_val)
    y_val = func(alpha,beta,x_val)
    y_model.append(y_val)

length = 10
nReal = 1

#add error using Cholesky decomposition
cov = (sigma**2 * np.identity(10))
L = np.linalg.cholesky(cov)
err = np.dot(L,np.random.normal(0.,sigma,(length,nReal)))
y_err=np.sqrt(np.diag(cov))

for iReal in range(nReal):
    y_plot = err[:,iReal] + y_model

    
#create plot of model values and data with added error
plt.scatter(x_plot,y_plot)
plt.plot(x_plot,y_model,"r-")
plt.errorbar(x_plot,y_plot,y_err, ls = "none")
plt.show()

#now apply chi2

#create grid of a and b values to step around
steps = 100

trial_a = np.repeat(np.linspace(7.2,8.7,steps),steps)
trial_b = np.tile(np.linspace(110,120,steps),steps)
chis = []

#step around grid, calculating chi2 at each point
for i in range(0,steps**2):
    chi = chi2(trial_a[i], trial_b[i], x_plot, y_plot, cov)
    chis.append(chi)

#convert results into a format useable by chainconsumer
chisq = np.array(chis).reshape(steps,steps)
 
a = np.array(trial_a[0::steps])
b = np.array(trial_b[0:steps])
pdf = np.array(np.exp(-0.5*chisq))

#plot chi2 using chainconsumer
c = ChainConsumer()
c.add_chain([a,b], parameters=[r"$\alpha$", r"$\beta$"], weights=pdf, grid=True)
fig = c.plotter.plot(display=True,truth=[alpha,beta])


