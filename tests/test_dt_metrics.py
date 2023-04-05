import pytest
from discretesampling.domain.decision_tree import calculate_leaf_occurences, Tree
from collections import Counter
from sklearn import datasets


def build_tree(feature, threshold):
    data = datasets.load_wine()

    X = data.data
    y = data.target
    leafs = [1, 2]

    tree = [[0, 1, 2, feature, X[threshold, feature]]]
    return Tree(X, y, tree, leafs)


@pytest.mark.parametrize(
    "tree,expected",
    [(build_tree(2, 10), (-182.25009765866628, [Counter({1: 0.6000000000000001,
                                                         0: 0.22666666666666668,
                                                         2: 0.17333333333333334}),
                                                Counter({0: 0.4077669902912621,
                                                         2: 0.3398058252427184,
                                                         1: 0.2524271844660194})])),
     (build_tree(2, 20), (-182.469284951157, [Counter({1: 0.6212121212121212,
                                                       0: 0.21212121212121213,
                                                       2: 0.16666666666666669}),
                                              Counter({0: 0.40178571428571425,
                                                       2: 0.33035714285714285,
                                                       1: 0.26785714285714285})])),
     (build_tree(4, 10), (-184.44864692209964, [Counter({1: 0.4841269841269841,
                                                         2: 0.2698412698412698,
                                                         0: 0.24603174603174602}),
                                                Counter({0: 0.5384615384615385,
                                                         2: 0.2692307692307693,
                                                         1: 0.19230769230769232})]))]
)
def test_calculate_leaf_occurences(tree, expected):
    leafs_possibilities_for_prediction = calculate_leaf_occurences(tree)
    assert leafs_possibilities_for_prediction == expected
