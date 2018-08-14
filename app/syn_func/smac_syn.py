"""
An example for the usage of SMAC within Python.
We optimize a simple SVM on the IRIS-benchmark.

Note: SMAC-documentation uses linenumbers to generate docs from this file.
"""

import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

import simulator as sim


func_name = "park2" #"currinExponential" #"univariate" "hosaki"
num_levels = 2
num_bits_per_dim = 10
sim1 = sim.Simulator(func_name, True, num_levels, num_bits_per_dim)
num_features, _, _, _ = sim1.get_config()

total_features = num_bits_per_dim*num_features


def simulator_fun(cfg):
  # print("SIMULATOR CFG ", cfg)
  feature_vector = np.zeros(total_features)
  for k in cfg:
    # print(k, int(k), cfg[k])
    feature_vector[int(k)] = cfg[k]
  output = sim1.run(feature_vector)
  # print(output)
  return -1*output


#logger = logging.getLogger("SVMExample")
logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()

for i in range(total_features):
  param_id = UniformIntegerHyperparameter(str(i), 0, 1, default_value=0)     # Only used by kernel poly
  cs.add_hyperparameter(param_id)


# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 200,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = simulator_fun(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=simulator_fun)

incumbent = smac.optimize()

inc_value = simulator_fun(incumbent)

print("Optimized Value: %.2f" % (inc_value))


# We can also validate our results (though this makes a lot more sense with instances)
# smac.validate(config_mode='inc',      # We can choose which configurations to evaluate
#               #instance_mode='train+test',  # Defines what instances to validate
#               repetitions=100,        # Ignored, unless you set "deterministic" to "false" in line 95
#               n_jobs=1)               # How many cores to use in parallel for optimization
