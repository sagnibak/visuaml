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


def predict(tree: Tree,
            X: np.ndarray,
            agg_fn: Callable[[np.ndarray], Union[int, float]])
    -> Union[int, float]:
    """Traverses the decision tree `tree` and aggregates the values stored in
    the appropriate leaf node using the function `agg_fn`. The original design
    matrix `X` must be passed in since the tree only stores indices.
    """
    if isinstance(tree) LeafNode:
        return agg_fn(X[tree.indices])
    elif X[tree.split_feature] < tree.split_value:
        return predict(tree.left_subtree, X, agg_fn)
    else:  # X[tree.split_feature] >= tree.split_value
        return predict(tree.right_subtree, X, agg_fn)

