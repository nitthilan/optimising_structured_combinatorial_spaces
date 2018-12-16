import numpy as np
import matplotlib.pyplot as plt
from BOCS import BOCS
from sample_models import sample_models
import sys
from ising_model import model_run
np.random.seed(42)
# Save inputs in dictionary
inputs = {}
inputs['n_vars']     = 12
inputs['evalBudget'] = 50
inputs['n_init']     = 20
inputs['lambda']     = 1e-4

# Save objective function and regularization term
#Q = quad_matrix(inputs['n_vars'], 100)
inputs['model']    = lambda x:  model_run(x, 9)
inputs['penalty']  = lambda x:  inputs['lambda']*np.sum(x)

# Generate initial samples for statistical models
inputs['x_vals']   = sample_models(inputs['n_init'], inputs['n_vars'])
y_vals = []
for i in range(len(inputs['x_vals'])):
	x_val = inputs['x_vals'][i]
	y_vals.append(inputs['model'](x_val))
inputs['y_vals'] = np.array(y_vals)
print(inputs['y_vals'])
# Run BOCS-SA and BOCS-SDP (order 2)

(BOCS_SA_model, BOCS_SA_obj)   = BOCS(inputs.copy(), 2, 'SA')
(BOCS_SDP_model, BOCS_SDP_obj) = BOCS(inputs.copy(), 2, 'SDP-l1')

# Compute optimal value found by BOCS
iter_t = np.arange(BOCS_SA_obj.size)
BOCS_SA_opt  = np.minimum.accumulate(BOCS_SA_obj)
BOCS_SDP_opt = np.minimum.accumulate(BOCS_SDP_obj)
print(BOCS_SA_opt)
print(BOCS_SDP_opt)
'''
# Compute minimum of objective function
n_models = 2**inputs['n_vars']
x_vals = np.zeros((n_models, inputs['n_vars']))
str_format = '{0:0' + str(inputs['n_vars']) + 'b}'
f_vals = []
for i in range(n_models):
    model = str_format.format(i)
    x_vals[i,:] = np.array([int(b) for b in model])
    if (i == 4007):
        print(x_vals[i, :])
    f_vals.append(inputs['model'](x_vals[i, :]) + inputs['penalty'](x_vals[i, :]))
f_vals = np.array(f_vals)
opt_f  = np.min(f_vals)
print(f_vals[np.argmin(f_vals)])
print(opt_f)
'''
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
fig.savefig('BOCS_ising.pdf')
plt.close(fig)

# -- END OF FILE --
