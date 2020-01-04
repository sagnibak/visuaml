import numpy as np

from dataclasses import dataclass
from typing import *


@dataclass(frozen=True)
class Tree:
    """A decision tree class. It is either a leaf or an internal node, as
    described below. This is a quick and dirty way of making an abstract class
    in Python (there are better ways but come on, this is a personal project
    and I am writing these docstrings is that not enough).
    """

    def __init__(self):
        raise NotImplementedError("Do not instantiate abstract class")


@dataclass(frozen=True)
class LeafNode(Tree):
    """Stores the indices of the examples in a leaf.
    Technically, leaves should only contain one label (predicted class for
    classification, predicted y value for regression). However, storing all
    corresponding examples in a leaf allows this same effective behavior using
    an appropriate prediction function (along with, possibly, caching), but is
    also conducive to splitting after an early stop.
    """

    indices: List[int]


@dataclass(frozen=True)
class InternalNode(Tree):
    """An internal node in the decision tree stores a splitting feature, which
    is the single feature that the node considers while making its decision. It
    has two subtrees: the left subtree is chosen if the splitting feature has
    value less than the splitting value, and the right subtree is chosen if the
    splitting feature has value greater than or equal to the splitting value.
    But of course, the class itself does not know all this. This is, however,
    why we have the fields we do.
    """

    split_feature: int
    split_value: float
    left_subtree: Tree
    right_subtree: Tree


def predict(
    tree: Tree,
    X: np.ndarray,
    y: np.ndarray,
    agg_fn: Callable[[np.ndarray], Union[int, float]],
) -> Union[int, float]:
    """Traverses the decision tree `tree` and aggregates the values stored in
    the appropriate leaf node using the monadic function `agg_fn`. The original
    labels `y` must be passed in since the tree only stores indices to predict
    the class or y-value of `X`.
    """
    if isinstance(tree, LeafNode):
        return agg_fn(y[tree.indices])
    elif X[tree.split_feature] < tree.split_value:
        return predict(tree.left_subtree, X, y, agg_fn)
    else:  # X[tree.split_feature] >= tree.split_value
        return predict(tree.right_subtree, X, y, agg_fn)


def predict_iter(
    tree: Tree,
    X: np.ndarray,
    y: np.ndarray,
    agg_fn: Callable[[np.ndarray], Union[int, float]],
) -> Union[int, float]:
    """Same logic as the recursive predict function but is iterative, so we can
    predict using arbitrarily deep trees.
    """
    while not isinstance(tree, LeafNode):
        if X[tree.split_feature] < tree.split_value:
            tree = tree.left_subtree
        else:
            tree = tree.right_subtree
    return agg_fn(y[tree.indices])


def grow(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    early_stop: Callable[[np.ndarray, np.ndarray], bool],
) -> Tree:
    """
    If there is only one class in y[indices] or we have reached an early
    stopping condition, then we return a leaf node containing all the elements
    being considered. Otherwise we need to find the split that minimizes the
    Gini Impurity of the children nodes, as follows:
        For each feature, do:
            Find and sort the unique values in X[indices, feature].
            Compute the Gini Impurity of the children after each non-trivial
            split.
        Split at the feature and value that minimizes the children's Gini
        Impurity and construct the children recursively.

    Parameters
    ----------
    X  design matrix of shape (n, m), i.e., there are n observations
       and m features
    y  vector of training labels
    indices  elements of `X` and `y` to be considered
    early_stop  a function that determines if tree growing should stop early

    Returns
    -------
    a decision tree trained on X[indices] and y[indices]
    """
    # if there is only one class in the current node or we have reached an
    # early stopping condition, then return a leaf
    if np.allclose(gini_impurity(y[indices]), 0.0) or early_stop(
        X[indices], y[indices]
    ):
        return LeafNode(indices.tolist())

    # otherwise find the best split (the one that minimizes the Gini Impurity)
    left_indices = indices
    right_indices = []
    split_feature = 0
    split_value = X[0, 0]
    min_gini_impurity = 1  # the Gini Impurity is never more than 1

    for feature in range(X.shape[1]):
        # np.unique sorts the unique values!
        unique_values = np.unique(X[indices, feature])

        # the first split is trivial since no element is less than the smallest
        # but there is at least one element >= the largest (emphasis on `=`)
        for val in unique_values[1:]:
            left_indices_ = indices[np.where(X[indices, feature] < val)[0]]
            right_indices_ = indices[np.where(X[indices, feature] >= val)[0]]
            left_impurity = gini_impurity(y[left_indices])
            right_impurity = gini_impurity(y[right_indices])

            if len(left_indices_) * left_impurity + len(
                right_indices_
            ) * right_impurity <= min_gini_impurity * len(indices):
                left_indices = left_indices_
                right_indices = right_indices_
                split_feature = feature
                split_value = val

    # return the decision tree
    return InternalNode(
        split_feature,
        split_value,
        grow(X, y, left_indices, early_stop),
        grow(X, y, right_indices, early_stop),
    )


def gini_impurity(y: np.ndarray) -> float:
    """Computes the Gini Impurity given labels `y`. Of the various ways to
    compute it, I chose the following formulation: 1 - the sum of the squared
    probability of each label. Note that `y` will often be a subset of the
    training labels, precisely, the labels of the elements of a node that may
    split.
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - sum([pi ** 2 for pi in probs])


def entropy(y: np.ndarray) -> float:
    """Computes the entropy given labels `y`. Used in similar contexts as Gini
    Impturity while constructing decision trees.
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -1 * sum([pi * np.log(pi) for pi in probs])


def leaves_too_small(
    threshold: int,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Returns an early stopping function which stops tree construction if there
    are at most `threshold` elements in the node being considered.
    """
    return lambda X, y: len(y) <= threshold
