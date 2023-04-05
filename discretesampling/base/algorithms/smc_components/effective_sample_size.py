import numpy as np
from discretesampling.base.algorithms.smc_components.logsumexp import log_sum_exp


def ess(logw):
    """
    Description
    -----------
    Computes the Effective Sample Size of the given normalised weights

    Parameters
    ----------
    logw : array of logged importance normalised weights

    Returns
    -------
    double scalar : Effective Sample Size

    """

    mask = np.invert(np.isneginf(logw))  # mask to filter out any weight = 0 (or -inf in log-scale)

    inverse_neff = np.exp(log_sum_exp(2*logw[mask]))

    return 1 / inverse_neff

