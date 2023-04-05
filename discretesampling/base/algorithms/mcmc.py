import math
import copy
from ...base.random import RNG


class DiscreteVariableMCMC():

    def __init__(self, variableType, target, initialProposal):
        self.variableType = variableType
        self.proposalType = variableType.getProposalType()
        self.initialProposal = initialProposal
        self.target = target

    def sample(self, N, seed=0):
        rng = RNG(seed)
        initialSample = self.initialProposal.sample(rng)
        current = initialSample

        samples = []
        for i in range(N):
            forward_proposal = self.proposalType(current, rng)
            proposed = forward_proposal.sample()

            reverse_proposal = self.proposalType(proposed, rng)

            forward_logprob = forward_proposal.eval(proposed)
            reverse_logprob = reverse_proposal.eval(current)

            current_target_logprob = self.target.eval(current)
            proposed_target_logprob = self.target.eval(proposed)

            log_acceptance_ratio = proposed_target_logprob -\
                current_target_logprob + reverse_logprob - forward_logprob
            if log_acceptance_ratio > 0:
                log_acceptance_ratio = 0
            acceptance_probability = min(1, math.exp(log_acceptance_ratio))

            q = rng.random()
            # Accept/Reject
            if (q < acceptance_probability):
                current = proposed
            else:
                # Do nothing
                pass

            samples.append(copy.deepcopy(current))

        return samples
