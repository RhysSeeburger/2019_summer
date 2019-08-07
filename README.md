# 2019_summer
The main results of my summer research project at the IFA in Edinburgh

I started with the ab_model program, in order to familiarise myself with concepts such as parameter estimation using chi-squared, and general practises in research and programming. I also gained insight into the chainconsumer package

Next was the omsig_model, a step up from the simple linear model, incorporating things such as importing data from external files, as well as sampling over different parameters.

I then spent some time on developing an iterative method, following the "core" idea of Bayesian analysis, that each posterior can be used as a prior for a new experiment. This was not very useful in terms of my project but gave me some more insight, as well as allowing me to dabble in some MCMC methods. Along with this, I emailed the creator of the emcee package, Dan Foreman-Mackey, asking him for input and advice, and received some code allowing me to use all my input data from multiple iterations "at once", rather than iterating.

I then moved on to the major part of my project, computing and handling Jeffrey's priors. I explored different ways of doing so, and wrote a program, Jeffreys, that allows the user to plot different Posteriors from different Priors in a selection of parameter spaces. The program uses MCMC methods throughout.

Last, I started working on some Reference prior ideas. Rather than computing a reference prior analytically, following advice from my supervisor, I instead decided to make a program that would allow me to test a range of priors as to how much they diverged from the posterior, averaging over many iterations and checking convergence to simulate taking the expectation value. In the future, I would like to expand this into multiple dimensions.


The ascii files simply hold the information required to run some of the programs, as I got data from my supervisor to perform the relevant analysis.
