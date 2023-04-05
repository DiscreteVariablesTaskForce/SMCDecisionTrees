import multiprocessing
import copy
from scipy.special import logsumexp
from ...base.random import RNG
from mpi4py import MPI
import numpy as np
import math
from discretesampling.base.algorithms.smc_components.normalisation import normalise
from discretesampling.base.algorithms.smc_components.effective_sample_size import ess
from discretesampling.base.algorithms.smc_components.resampling import systematic_resampling


class DiscreteVariableSMC():

    def __init__(self, variableType, target, initialProposal,
                 use_optimal_L=False,
                 parallel=False,
                 num_cores=None):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.use_optimal_L = use_optimal_L
        self.parallel = parallel
        self.num_cores = num_cores
        self.P = MPI.COMM_WORLD.Get_size()  # number of MPI nodes/ranks
        self.rank = MPI.COMM_WORLD.Get_rank()

        if (self.parallel and (num_cores is None)):
            num_cores = multiprocessing.cpu_count()
            print("WARNING: `parallel=True` but `num_cores` not specified; "
                  + "setting `num_cores = ", num_cores, "`")
            self.num_cores = num_cores

        if use_optimal_L:
            self.LKernelType = variableType.getOptimalLKernelType()
        else:
            # By default getLKernelType just returns
            # variableType.getProposalType(), the same as the forward_proposal
            self.LKernelType = variableType.getLKernelType()

        self.initialProposal = initialProposal
        self.target = target

    def sample(self, Tsmc, N, seed=0):
        loc_n = int(N/self.P)
        mvrs_rng = RNG(seed)
        rngs = [RNG(i + self.rank*loc_n + 1 + seed) for i in range(loc_n)]  # RNG for each particle

        initialParticles = [self.initialProposal.sample(rngs[i], self.target) for i in range(loc_n)]
        current_particles = initialParticles
        logWeights = np.array([self.target.eval(p) - self.initialProposal.eval(p, self.target) for p in initialParticles])

        for t in range(Tsmc):
            logWeights = normalise(logWeights)
            neff = ess(logWeights)
            #if MPI.COMM_WORLD.Get_rank() == 0:
            #    print("Neff = ", neff)
            if math.log(neff) < math.log(N) - math.log(2):
                #if MPI.COMM_WORLD.Get_rank() == 0:
                #    print("Resampling...")

                current_particles, logWeights = systematic_resampling(current_particles, logWeights, mvrs_rng)  #resample(current_particles, logWeights, rngs[0])

            new_particles = copy.deepcopy(current_particles)

            forward_logprob = np.zeros(len(current_particles))

            # Sample new particles and calculate forward probabilities
            for i in range(loc_n):
                forward_proposal = self.proposalType(current_particles[i], rng=rngs[i])
                new_particles[i] = forward_proposal.sample()
                forward_logprob[i] = forward_proposal.eval(new_particles[i])

            if self.use_optimal_L:
                Lkernel = self.LKernelType(
                    new_particles, current_particles, parallel=self.parallel,
                    num_cores=self.num_cores
                )
            for i in range(loc_n):
                if self.use_optimal_L:
                    reverse_logprob = Lkernel.eval(i)
                else:
                    Lkernel = self.LKernelType(new_particles[i])
                    reverse_logprob = Lkernel.eval(current_particles[i])

                current_target_logprob = self.target.eval(current_particles[i])
                new_target_logprob = self.target.eval(new_particles[i])

                logWeights[i] += new_target_logprob - current_target_logprob + reverse_logprob - forward_logprob[i]

            current_particles = new_particles

        return current_particles

