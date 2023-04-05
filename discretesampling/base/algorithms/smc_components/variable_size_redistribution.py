from discretesampling.domain.decision_tree.util import pad, restore
from discretesampling.base.algorithms.smc_components.distributed_fixed_size_redistribution.redistribution import fixed_size_redistribution


def variable_size_redistribution(particles, ncopies):
    x = pad(particles)

    x = fixed_size_redistribution(x, ncopies)

    particles = restore(x, particles)

    return particles
