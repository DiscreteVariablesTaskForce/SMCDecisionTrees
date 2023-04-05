# DiscreteSamplingFramework
Python classes describing discrete variable sampling/proposals/targets

# ADD LICENSES

## Requirements
 - Python 3.x
 - numpy
 - scipy
 - scikit-learn (for examples)

## Variables and Distributions
### Discrete Variables
Each example of these should, at minimum implement functions:
 - `__init__` - a constructor with some arguments setting up the basic attributes of the variable
 - `getProposalType` - a class method returning the relevant proposal distribution type for this variable type
 - `getInitialProposalType` - a class method returning the relevant initial proposal distribution type for this variable type
- `getTargetType` - a class method returning the relevant target distribution type for this variable type

We may at some point in the future need to also include:
 - `getLKernelType` - a class method returning the relevant LKernel distribution type for this variable type
 - `getOptimalLKernelType` - a class method returning the relevant Optimal LKernel distribution type for this variable type (a special case of the above probably)
 
 as well as implementing these distribution classes for each variable type.


### Discrete Variable Proposal Distributions
Proposal distributions, q(x'|x), for each variable type should be described.
Each example of these should, at minimum implement functions:
 - `__init__` - constructor with a single argument (a DiscreteVariable, x) which will act as the "starting point" for this proposal
 - `eval` - function taking a single argument (a DiscreteVariable, x') that returns the log-probability of sampling that DiscreteVariable from this proposal (P(x|x'))
 - `sample` - function with no arguments which returns a sample of a DiscreteVariable, x', from this proposal, q (x' ~ q(x))

For more efficient evaluation, optionally implement class methods:
 - `norm` - function that takes a single additional argument (a DiscreteVariable x) and returns some position-like value for that proposal/variable type. This value should impose some sort of ordering on DiscreteVariables.
 - `heuristic` - function that takes two arguments, which will be returns from the above `norm` function and, given
 these values, either return true if the proposal could potentially make transitions between two DiscreteVariables with these norm values or otherwise return false. Note that `heuristic(norm(x), norm(y)) == true` need not guarantee that the transition probability between x and y is non-zero. The inverse is however true: `heuristic(norm(x), norm(y)) == false` guarantees that the transition probability between x and y is zero.

 These two functions will allows for efficient culling of proposal evaluations when implementing e.g. optimal L-kernels


### Discrete Variable Initial Proposal Distributions
Similarly, distributions of initial proposals. q0(x) should be described.
Each example of these should, at minimum implement functions:
 - `__init__` - constructor, possibly with some extra parameters
 - `eval` - function taking a single argument (a DiscreteVariable, x) that returns the log-probability of sampling that DiscreteVariable from this proposal (q0(x))
 - `sample` - function with no arguments which returns a sample of a DiscreteVariable, x', from this initial proposal (x ~ q0())

### Discrete Variable Target Distributions
Each example of these should, at minimum implement functions:
 - `__init__` - constructor, possibly with some parameters theta and data D
 - `eval` - function taking a single argument (a DiscreteVariable, x) that returns the log-posterior density logP(D|x,theta) + logP(x|theta)

## Algorithms

MCMC and SMC samplers for discrete variables are implemented. Working examples for the implemented variable types can be found in the `examples` directory.

### MCMC sampler
Basic MCMC sampler which can be initialised with a variable type, target distribution and initial proposal (initial samples are drawn from this). Samples can then be drawn with the `.sample` function, which takes one argument, `N`, the number of iterations. An example using the decision tree variable type is shown below:

```python
from discretesampling import decision_tree as dt
from discretesampling.algorithms import DiscreteVariableMCMC
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

data = datasets.load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=5)

a = 0.01
b = 5
target = dt.TreeTarget(a,b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

mcmcSampler = DiscreteVariableMCMC(dt.Tree, target, initialProposal)
samples = mcmcSampler.sample(N = 5000)
```

### SMC sampler
Similary, a basic SMC sampler which can be initialised with a variable type, target distribution and initial proposal (initial samples are drawn from this and weights calculated appropriately). Samples can then be drawn with the `sample` function, which takes two arguments, `N` and `P` the number of iterations and number of particles. An example using the decision tree variable type is shown below:

```python
from discretesampling import decision_tree as dt
from discretesampling.algorithms import DiscreteVariableSMC
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

data = datasets.load_wine()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=5)

a = 0.01
b = 5
target = dt.TreeTarget(a,b)
initialProposal = dt.TreeInitialProposal(X_train, y_train)

smcSampler = DiscreteVariableSMC(dt.Tree, target, initialProposal)
samples = smcSampler.sample(N=10,P=1000)
```
