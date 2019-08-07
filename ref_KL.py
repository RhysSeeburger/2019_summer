"""
Program to test a variety of priors, computing their average KL divergence
The prior with the largest KL divergence from the posterior is the least informative
author: Rhys Seeburger
"""

#import relevant packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import stats

#set a large, arbitrary maximum number of iterations
it = 100000

#set up lists to hold KL values
KLflat = []
KLbeta05 = []
KLbeta2 = []
KLgauss = []
KLslide = []

#choose true value For alpha and ranges
alpha = 0.75
sigma = 1
lower = 0
upper = 1

#define function to compute the average of a list
def avg(list):
    return sum(list)/len(list)

#define model function
def func(alpha, x):
    y = alpha * np.asarray(x)
    return y

#define likelihood
def like(alpha, x, y_data, cov):
    model = func(alpha, x)
    like = np.exp(-0.5 * (np.dot(np.dot((np.asarray(y_data)-np.asarray(model)), np.linalg.inv(cov)), np.transpose(np.asarray(y_data)-np.asarray(model)))))
    return like

#define constraints
def const(alpha):
    if alpha < lower or alpha > upper:
        return 0
    else:
        return 1

#start iterating
for i in range(0,it):

    #set up datapoints
    length = 1
    nReal = 1

    #create covariance, use it to get cholesky matrix
    cov = (sigma**2 * np.identity(length))
    L = np.linalg.cholesky(cov)

    #create datapoints
    x_val = np.linspace(1,10,length)
    y_data = []
    y_model = func(alpha, x_val)

    #add error using cholesky matrix
    err = np.dot(L, np.random.normal(0., 1., (length, nReal)))
    y_err=np.sqrt(np.diag(cov))

    for iReal in range(nReal):
        y_data = err[:, iReal] + y_model

    #define some priors
    def flat(a, param1, param2):
        return 1
    def beta(a, param1, param2):
        return stats.beta.pdf(a, param1, param2)
    def gauss(a, param1, param2):
        return stats.norm.pdf(a, loc=param1, scale=param2)
    def slide(a,param1,param2):
        return a**-0.5 / (1+a)

    #define function for evidence
    def integ(a, x, y_data, cov, prior, param1, param2, priornorm, likenorm):
        return const(a) * like(a, x, y_data, cov)/(likenorm) * prior(a, param1, param2)/(priornorm)

    def evidence(x, y_data, cov, prior, param1, param2, priornorm, likenorm, lower, upper):
        Int = integrate.quad(integ, lower, upper, args=(x, y_data, cov, prior, param1, param2, priornorm, likenorm))
        return Int

    #define function to compute KL divergence
    def integrand(a, x, y_data, cov, prior, param1, param2, priornorm, likenorm, evidence):
        return like(a, x, y_data, cov) / (likenorm) * const(a) * prior(a, param1, param2)/ (priornorm * evidence) * np.log(like(a,x,y_data,cov)/(likenorm * evidence))

    def KL(x,y_data,cov,prior,param1,param2,priornorm,likenorm,evidence,integrand):
        return integrate.quad(integrand, 0.0001, 0.9999, args=(x, y_data, cov, prior, param1, param2, priornorm, likenorm, evidence))


    #find normalisation constants
    flatconst = integrate.quad(flat, lower, upper, args=(0, 0))[0]
    beta05const = integrate.quad(beta, lower, upper, args=(0.5, 0.5))[0]
    beta2const = integrate.quad(beta, lower, upper, args=(2, 2))[0]
    gaussconst = integrate.quad(gauss, lower, upper, args=(0.7, 0.2))[0]
    slideconst = integrate.quad(slide, lower, upper, args=(0, 0))[0]
    likeconst = integrate.quad(like, lower, upper, args=(x_val,y_data,cov))[0]

    #compute evidences
    evflat = evidence(x_val, y_data, cov, flat, 1, 1, flatconst, likeconst, lower, upper)[0]
    evbeta05 = evidence(x_val, y_data, cov, beta, 0.5, 0.5, beta05const, likeconst, lower, upper)[0]
    evbeta2 = evidence(x_val, y_data, cov, beta, 2., 2., beta2const, likeconst, lower, upper)[0]
    evgauss = evidence(x_val, y_data, cov, gauss, 0.7, 0.2, gaussconst, likeconst, lower, upper)[0]
    evslide = evidence(x_val, y_data, cov, slide, 1., 1., slideconst, likeconst, lower, upper)[0]

    #compute KL values
    flatp = KL(x_val, y_data, cov, flat, 1, 1, flatconst, likeconst, evflat, integrand)
    betap05 = KL(x_val, y_data, cov, beta, 0.5, 0.5, beta05const, likeconst, evbeta05, integrand)
    betap2 = KL(x_val, y_data, cov, beta, 2., 2., beta2const, likeconst, evbeta2, integrand)
    gaussp = KL(x_val, y_data, cov, gauss, 0.7, 0.2, gaussconst, likeconst, evgauss, integrand)
    slidep = KL(x_val, y_data, cov, slide, 1., 1., slideconst, likeconst, evslide, integrand)

    #append KL values into the lists
    KLflat.append(flatp[0])
    KLbeta05.append(betap05[0])
    KLbeta2.append(betap2[0])
    KLgauss.append(gaussp[0])
    KLslide.append(slidep[0])

    #create masterlist holding sublists of KL values
    KLmaster = [KLflat,KLbeta05,KLbeta2,KLgauss,KLslide]
    
    #define function to calculate the average of the KL values excluding the most recent one
    def old(KL_list, index):
        return avg(KL_list[0:index])

    #for all iterations from the second onwards, calculate both current and previous KL averages
    #take the ratio, if ratio is close to 1 for all of them, convergence is reached, break iteration
    if i >= 1:

        KL_old = [old(KLmaster[0],i), old(KLmaster[1],i), old(KLmaster[2],i), old(KLmaster[3],i), old(KLmaster[4],i)]

        KL_new = [avg(KLmaster[0]), avg(KLmaster[1]), avg(KLmaster[2]), avg(KLmaster[3]), avg(KLmaster[4])]

        KL_ratios = [KL_new[0]/KL_old[0], KL_new[1]/KL_old[1], KL_new[2]/KL_old[2], KL_new[3]/KL_old[3], KL_new[4]/KL_old[4]]

        if all(x > 0.99 for x in KL_ratios) and all(x < 1.01 for x in KL_ratios):
            break

#define scale factor for readability
scale = 1000

#print average KL values for various priors
print("\n")
print("Convergence at iteration " + str(i))
print("For a Beta(0.5, 0.5) Prior, the (average, scaled) KL is " + str(round(scale*KL_new[1])))
print("For a Slide Prior, the (average, scaled) KL is " + str(round(scale*KL_new[4])))
print("For a Flat Prior, the (average, scaled) KL is " + str(round(scale*KL_new[0])))
print("For a Beta(2, 2) Prior, the (average, scaled) KL is " + str(round(scale*KL_new[2])))
print("For a Gaussian Prior, the (average, scaled) KL is " + str(round(scale*KL_new[3])))