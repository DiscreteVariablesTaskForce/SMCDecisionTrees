import numpy as np
from discretesampling.base.algorithms.smc_components.logsumexp import log_sum_exp


def normalise(logw):
    """
    Description
    -----------
    Normalise importance weights. Note that we remove the mean here
        just to avoid numerical errors in evaluating the exponential.
        We have to be careful with -inf values in the log weights
        sometimes. This can happen if we are sampling from a pdf with
        zero probability regions, for example.

    Parameters
    ----------
    logw : array of logged importance weights

    Returns
    -------
    - : array of log-normalised importance weights

    """

    mask = np.invert(np.isneginf(logw))  # mask to filter out any weight = 0 (or -inf in log-scale)

    log_wsum = log_sum_exp(logw[mask])

    return logw - log_wsum
