import numpy as np
import pydot

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
    depth: int = 0,
    max_depth: Union[int, float] = float("inf"),
    split_between: bool = False,
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
    indices     elements of `X` and `y` to be considered
    depth       the depth of the current node
    max_depth   maximum depth of the tree (infinity by default)
    early_stop  a function that determines if tree growing should stop early
    split_between  whether to make splits at unique values or in between them

    Returns
    -------
    a decision tree trained on X[indices] and y[indices]
    """
    # if there is only one class in the current node or we have reached an
    # early stopping condition, then return a leaf
    if (
        np.allclose(gini_impurity(y[indices]), 0.0)
        or early_stop(X[indices], y[indices])
        or depth >= max_depth
    ):
        return LeafNode(indices.tolist())

    # otherwise find the best split (the one that minimizes the Gini Impurity)
    left_indices = indices
    right_indices = []
    # split_feature = 0
    # split_value = X[0, 0]
    min_gini_impurity = 1  # the Gini Impurity is never more than 1

    for feature in range(X.shape[1]):
        # np.unique sorts the unique values!
        unique_values = np.unique(X[indices, feature])
        if split_between:
            # split between every pair of unique points
            split_points = np.correlate(unique_values, np.array([0.5, 0.5]))
        else:
            # split at exactly the unique points (but skip the first since that is trivial)
            split_points = unique_values[1:]

        # the first split is trivial since no element is less than the smallest
        # but there is at least one element >= the largest (emphasis on `=`)
        for val in split_points:
            left_indices_ = indices[np.where(X[indices, feature] < val)[0]]
            right_indices_ = indices[np.where(X[indices, feature] >= val)[0]]
            left_impurity = gini_impurity(y[left_indices_])
            right_impurity = gini_impurity(y[right_indices_])

            weighted_children_impurity = (
                len(left_indices_) * left_impurity
                + len(right_indices_) * right_impurity
            ) / len(indices)
            if weighted_children_impurity < min_gini_impurity:
                min_gini_impurity = weighted_children_impurity
                left_indices = left_indices_
                right_indices = right_indices_
                split_feature = feature
                split_value = val

    # return the decision tree
    return InternalNode(
        split_feature,
        split_value,
        grow(X, y, left_indices, early_stop, depth + 1, max_depth),
        grow(X, y, right_indices, early_stop, depth + 1, max_depth),
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


def visualize(
    tree: Tree, vis: pydot.Dot, y: np.ndarray, clock: int = 0
) -> Tuple[pydot.Node, int]:
    """Perform a depth-first traversal of `tree` to visualize the decision tree.
    Original labels `y` must be provided to get the classifications at the leaf
    nodes. This is a **mutative** function: it mutates |0w0| the pydot.Dot
    instance `vis`. We don't do anything in the pre-visit. In the post-visit
    we make a new node for the root node and add the two edges to its children.

    Parameters
    ----------
    tree   the decision tree to be visualized
    vis    pydot.Dot instance to be **mutated**
    y      training labels
    clock  tree traversal clock (to generate unique names for nodes)

    Returns
    -------
    a tuple containing
        the root node
        the incremented clock
    """
    # base case: Leaf node
    if isinstance(tree, LeafNode):
        vals, counts = list(
            map(
                np.ndarray.tolist,
                np.unique(y[tree.indices], return_counts=True),
            )
        )
        node = pydot.Node(
            str(clock), label=f"{list(zip(vals, counts))}", shape="box"
        )
        vis.add_node(node)

    # otherwise make both children subtrees, then add edges
    else:
        left_node, clock = visualize(tree.left_subtree, vis, y, clock)
        right_node, clock = visualize(tree.right_subtree, vis, y, clock)

        node = pydot.Node(
            str(clock),
            label=f"Split feature {tree.split_feature}\n"
            + f"Split value {tree.split_value}",
        )
        vis.add_node(node)
        vis.add_edge(pydot.Edge(node, left_node, label="<"))
        vis.add_edge(pydot.Edge(node, right_node, label=">="))

    return node, clock + 1
