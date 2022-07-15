# aggforce

A package to aggregate atomistic forces to estimate the forces of a given
manybody potential of mean force. 

### Example usage

The following code shows how to generate an optimal linear force aggregation
map. We grab test data, create a carbon alpha mapping, detect constrained 
bonds from the trajectory, and then produce and apply an optimize force
aggregation map to the trajectory.

```python
from aggforce import linearmap as lm
from aggforce import agg as ag
from aggforce import constfinder as cf
import numpy as np
import re
import mdtraj as md

# get data
forces = np.load("tests/data/cln025_record_2_prod_97.npz")['Fs']
coords = np.load("tests/data/cln025_record_2_prod_97.npz")['coords']
pdb = md.load("tests/data/cln025.pdb")

# generate carbon alpha map of the form [[inds1],[inds2],[inds3] where each
# element of the parent list coresponds to the atoms contributing to 
# a particular cg particle 
inds = []
atomlist = list(pdb.topology.atoms)
for ind,a in enumerate(atomlist):
    if re.search(r"CA$",str(a)):
            inds.append([ind])

# linear transformations are represented by LinearMap instances
# create our configurational c-alpha map, which is needed to optimize 
# the force map
cmap = lm.LinearMap(inds,n_fg_sites=coords.shape[1])
# detect which atoms have bond constraints based on statistics, only use 10
#frames
constraints = cf.guess_pairwise_constraints(coords[0:10],threshold=1e-3)
# get force map which uniformly aggregates forces inside the cg bead and adds
# other atoms to satisfy constraint rules
basic_results = ag.project_forces(xyz=None,
                                  forces=forces,
				  config_mapping=cmap,
				  constrained_inds=constraints,
				  method=lm.constraint_aware_uni_map)
# get _optimized_ force map which uniformly aggregates forces inside the cg bead and adds
# other atoms to satisfy constraint rules
optim_results = ag.project_forces(xyz=None,
                                  forces=forces,
				  config_mapping=cmap,
				  constrained_inds=constraints)

# optimal map itself optim_results['map']
#   this object is callable on mdtraj formatted force/position arrays
# optimal map _matrix_ is optim_results['map'].standard_matrix
# forces processed via the optimal map are under optim_results['project_forces']
# similarly for basic map
```
