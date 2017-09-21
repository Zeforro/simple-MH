'''
A Simple Metropolis-Hastings MCMC in R by Florian Hartig
(https://theoreticalecology.wordpress.com/2010/09/17/metropolis-hastings-mcmc-in-r/)

Python implementation by zeforro (September 21th, 2017)

Bayesian linear regression using Metropolis-Hastings.

func:	y = a*x + b + sd
		
		a  : slope
		b  : intercept
		sd : noise

		x  : data
		y  : observation

goal:	p(a,b,sd | x,y) 

		: posterior distribution
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,uniform
eps = 1e-6

# initial value
trueA = 5 # slope
trueB = 0 # intercept
trueSD = 10 # noise

# creating sample data
N = 51
x = np.zeros(N)
y = np.zeros(N)
for i in range(N):
	x[i] = i - 15
	y[i] = trueA * x[i] + trueB + np.random.randn() * trueSD
'''
# plot sample data
plt.plot(x,y,'o')
plt.show()
'''

# likelihood (logarithm)
def likelihood(a,b,sd):
	pred = a*x + b # predictions
	single_likelihood = norm.logpdf(y, loc=pred, scale=sd) # loc=mean, scale=var
	return sum(single_likelihood)
'''
# example of likelihood for slope a : [3,7]
slope_A = np.zeros(81)
for i in range(81):
	slope_A[i] = 3 + i*0.05
slopell_A = map(lambda z: likelihood(z,trueB,trueSD), slope_A)
plt.plot(slope_A,slopell_A)
plt.show()
'''

# prior (logarithm)
def prior(a,b,sd):
	# 'uninformative' paramters for a_prior and b_prior
	# (uniform distribution and normal distribution)
	a_prior = uniform.logpdf(a,loc=0,scale=10) # MIN=loc, MAX=loc+scale
	b_prior = norm.logpdf(b,scale=5) # loc=mean(default:0),scale=var
	# 1/sigma is applied for denying standard deviation's informativeness.
	# (check 'Jeffreys Prior' for more detail.)
	# (add epsilon constant to avoid division by zero.)
	sd_prior = 1/(sd+eps) * uniform.logpdf(sd,loc=0,scale=30) # MIN=loc, MAX=loc+scale
	return a_prior + b_prior + sd_prior

# posterior
def posterior(parameter):
	a = parameter[0]
	b = parameter[1]
	sd = parameter[2]
	return likelihood(a,b,sd) + prior(a,b,sd)

# Metropolis-Hastings Algorithm
def Metropolis_Hastings(parameter_init, iteration_time):
	result = []
	result.append(parameter_init)
	#count = 0
	for t in range(iteration_time):
		step_var = [0.4, 0.4, 0.4]
		proposal = np.zeros(3)
		for i in range(3):
			proposal[i] = norm.rvs(loc=result[-1][i], scale=step_var[i]) 
			# mean=previous parameter, var=step_var
		probability = np.exp(posterior(proposal) - posterior(result[-1]))
	
		if (uniform.rvs() < probability):
			result.append(proposal)
			#count += 1
		else:
			result.append(result[-1])
	return result

# main program
parameter_0 = [4,0,10]
iter_t = 20000
result = Metropolis_Hastings(parameter_0, iter_t)
burnIn = 10000
result = result[burnIn:]

# plotting the result
a_result = np.zeros(burnIn)
b_result = np.zeros(burnIn)
sd_result = np.zeros(burnIn)
for i in range(burnIn):
	a_result[i] = result[i][0]
	b_result[i] = result[i][1]
	sd_result[i] = result[i][2]
fig, axarr = plt.subplots(2,2)
axarr[0,0].hist(a_result, 50, facecolor='green', alpha=0.75)
axarr[0,0].axvline(5.0,color='r')
axarr[0,0].set_xlim(3.0,7.0)
axarr[0,0].title.set_text('Posterior of a')
axarr[0,1].hist(b_result, 50, facecolor='green', alpha=0.75)
axarr[0,1].axvline(0.0,color='r')
axarr[0,1].set_xlim(-10.0,10.0)
axarr[0,1].title.set_text('Posterior of b')
axarr[1,0].hist(sd_result, 50, facecolor='green', alpha=0.75)
axarr[1,0].axvline(10.0,color='r')
axarr[1,0].set_xlim(0.0,20.0)
axarr[1,0].title.set_text('Posterior of sd')
fig.delaxes(axarr[1,1])
plt.tight_layout()
plt.show()

