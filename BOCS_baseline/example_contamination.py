import numpy as np
import matplotlib.pyplot as plt
from BOCS import BOCS
from sample_models import sample_models
import sys
#print(sys.argv)
#np.random.seed(int(sys.argv[1]))
np.random.seed(42)
# Save inputs in dictionary
inputs = {}
inputs['n_vars']     = 25
inputs['evalBudget'] = 80
inputs['n_init']     = 20
inputs['lambda']     = 1e-4

def model_run(x):
    '''
        x = prevention {0, 1} vector
        n_samples = no of Monte Carlo samples to generate (T in the paper)
        lambda_reg = regularization parameter
    '''
    #print(x.shape)
    nGen = 100   # Number of Monte Carlo samples
    n = inputs['n_vars']              # no of stages
    x = x.reshape(n) 
    #print(x.shape)
    #nGen = n_samples            # no of samples to generate
    Z = np.zeros((nGen, n))     # contamination variable
    epsilon = 0.05 * np.ones(n) # error probability
    u = 0.1*np.ones(n)          # upper threshold for contamination
    cost = np.ones(n)           # cost for prevention at stage i
    # Beta parameters
    initialAlpha=1
    initialBeta=30
    contamAlpha=1
    contamBeta=17/3
    restoreAlpha=1
    restoreBeta=3/7
    
    # generate initial contamination fraction for each sample
    initialZ = np.random.beta(initialAlpha, initialBeta, nGen)
    # generate rates of contamination for each stage and sample
    lambdad = np.random.beta(contamAlpha, contamBeta, (nGen, n))
    # generate rates of restoration for each stage and sample
    gamma = np.random.beta(restoreAlpha, restoreBeta, (nGen, n))
    
    # calculate rates of contamination 
    Z[:, 0] = lambdad[:, 0]*(1-x[0])*(1-initialZ) + (1-gamma[:, 0]*x[0])*initialZ
    for i in range(1, n):
        Z[:, i] = lambdad[:, i]*(1-x[i])*(1-Z[:, i-1]) + (1-gamma[:, i]*x[i])*Z[:, i-1]
    #print(Z)
    con = np.zeros((nGen, n))
    #on = np.zeros((n, nGen))
    for j in range(nGen):
        con[j, :] = Z[j, :] >= u
    #print(con)
    
    con = con.T
    loss_function = 0
    for i in range(n):
        loss_function += (cost[i]*x[i]+(np.sum(con[i, :])/nGen)) 
    #loss_function += lam*np.sum(x)
    #print(loss_function)
    return loss_function

# Save objective function and regularization term
#Q = quad_matrix(inputs['n_vars'], 100)
inputs['model']    = lambda x: model_run(x)
inputs['penalty']  = lambda x: inputs['lambda']*np.sum(x)

# Generate initial samples for statistical models
inputs['x_vals']   = sample_models(inputs['n_init'], inputs['n_vars'])
y_vals = []
for i in range(len(inputs['x_vals'])):
	x_val = inputs['x_vals'][i]
	y_vals.append(inputs['model'](x_val))
inputs['y_vals'] = np.array(y_vals)
#print(inputs['y_vals'])
# Run BOCS-SA and BOCS-SDP (order 2)
(BOCS_SA_model, BOCS_SA_obj)   = BOCS(inputs.copy(), 2, 'SA')
(BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, 'SDP-l1')

# Compute optimal value found by BOCS
iter_t = np.arange(BOCS_SA_obj.size)
BOCS_SA_opt  = np.minimum.accumulate(BOCS_SA_obj)
BOCS_SDP_opt = np.minimum.accumulate(BOCS_SDP_obj)
print(BOCS_SA_opt)
print(BOCS_SDP_opt)
# Compute minimum of objective function
n_models = 2**inputs['n_vars']
x_vals = np.zeros((n_models, inputs['n_vars']))
str_format = '{0:0' + str(inputs['n_vars']) + 'b}'
f_vals = []
for i in range(n_models):
	model = str_format.format(i)
	x_vals[i,:] = np.array([int(b) for b in model])
	f_vals.append(inputs['model'](x_vals[i, :]) + inputs['penalty'](x_vals[i, :]))
f_vals = np.array(f_vals)
opt_f  = np.min(f_vals)
print(np.argmin(f_vals))
print(opt_f)
#tv =  np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 0])[np.newaxis, :]
#print(inputs['model'](tv) + inputs['penalty'](tv))
# Plot results
fig = plt.figure()
ax  = fig.add_subplot(1,1,1)
ax.plot(iter_t, BOCS_SA_opt, color='r', label='BOCS-SA')
ax.plot(iter_t, BOCS_SDP_opt, color='b', label='BOCS-SDP')
ax.set_yscale('log')
ax.set_xlabel('$t$')
ax.set_ylabel('Best $f(x)$')
ax.legend()
fig.savefig('BOCS_contamination.pdf')
plt.close(fig)

# -- END OF FILE --
