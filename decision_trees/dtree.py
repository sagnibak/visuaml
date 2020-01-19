import numpy as np
import pydot

from dataclasses import dataclass, field
from typing import *
import warnings


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


@dataclass
class DecisionTreeClassifier:
    """A convenience class to tie together a decision tree, a prediction
    function, and the training data required to make a classification. This
    provides an interface like scikit-learn, where `clf.fit(X, y)` trains a
    classifier and `clf.predict(x)` can be used to find the prediction of `x`.
    """

    # the decision tree
    tree: Tree
    # aggregation function to use while predicting
    agg_fn: Callable[[np.ndarray], Union[int, float]]
    # the training data
    X: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    # private flag to mark whether the classifier has been fit
    _trained = field(init=False, default_factory=lambda: False)

    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> None:
        """Fit the tree `self.tree` on data `X` and `y`. `args` and `kwargs`
        are passed to the function `grow`. Also, save the training data to
        enable prediction.
        """
        # warn the user if the model has been trained once
        if self._trained is True:
            warnings.warn(
                RuntimeWarning(
                    "Re-training a pretrained classifier, previous training will be lost."
                )
            )

        self.X = X
        self.y = y
        self.tree = grow(X, y, np.arange(len(y)), *args, **kwargs)
        self._trained = True

    def predict(self, x: np.ndarray) -> Union[int, float]:
        """Predict the label for data point `x`."""
        if not self._trained:
            raise RuntimeError(
                "Cannot predict using untrained model. Please run `clf.fit(X, y)` first."
            )
        return predict_iter(self.tree, x, self.y.astype(np.int64), self.agg_fn)


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
    min_leaf_size: int = 1,
    early_stop: Callable[[np.ndarray, np.ndarray], bool] = lambda X, y: False,
    weights_pt: Optional[np.ndarray] = None,
    weights_ft: Optional[np.ndarray] = None,
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
    y  vector of training labels, shape (n, 1)
    indices     elements of `X` and `y` to be considered
    min_leaf_size  the minimum size of any leaf node
    early_stop  a function that determines if tree growing should stop early
    weights_pt  the weight of each point in the training set, shape (n,)
    weights_ft  the weight of each feature in the training set, shape (m,)
                if a weight vector is None, then each point/feature is equally
                weighted
    depth       the depth of the current node
    max_depth   maximum depth of the tree (infinity by default)
    split_between  whether to make splits at unique values or in between them

    Returns
    -------
    a decision tree trained on X[indices] and y[indices]
    """
    # if there is only one class in the current node or we have reached an
    # early stopping condition, then return a leaf
    if (
        np.allclose(gini_impurity(y[indices]), 0.0)  # node is pure
        or early_stop(X[indices], y[indices])  # hit stopping condition
        or depth >= max_depth  # reached max depth
        or _all_points_same(X, indices)  # all points same
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
            # split at exactly the unique points (but skip the first since that
            # results in the trivial split)
            split_points = unique_values[1:]

        # the first split is trivial since no element is less than the smallest
        # but there is at least one element >= the largest (emphasis on `=`)
        for val in split_points:
            left_indices_ = indices[np.where(X[indices, feature] < val)[0]]
            right_indices_ = indices[np.where(X[indices, feature] >= val)[0]]
            # if any leaf is too small after this split, don't consider it
            if (
                len(left_indices_) < min_leaf_size
                or len(right_indices_) < min_leaf_size
            ):
                continue
            # otherwise see how good this split is
            left_impurity = gini_impurity(y[left_indices_], w=weights_pt)
            right_impurity = gini_impurity(y[right_indices_], w=weights_pt)

            weighted_children_impurity = (
                len(left_indices_) * left_impurity
                + len(right_indices_) * right_impurity
            ) / len(indices)

            if weights_ft is not None:
                weighted_children_impurity *= weights_ft[feature]

            if weighted_children_impurity < min_gini_impurity:
                min_gini_impurity = weighted_children_impurity
                left_indices = left_indices_
                right_indices = right_indices_
                split_feature = feature
                split_value = val

    # if we did not decide to split further, return a leaf
    if right_indices == []:
        return LeafNode(indices.tolist())

    # otherwise return the decision tree
    return InternalNode(
        split_feature,
        split_value,
        grow(
            X,
            y,
            left_indices,
            min_leaf_size=min_leaf_size,
            early_stop=early_stop,
            weights_pt=weights_pt,
            weights_ft=weights_ft,
            depth=depth + 1,
            max_depth=max_depth,
            split_between=split_between,
        ),
        grow(
            X,
            y,
            right_indices,
            min_leaf_size=min_leaf_size,
            early_stop=early_stop,
            weights_pt=weights_pt,
            weights_ft=weights_ft,
            depth=depth + 1,
            max_depth=max_depth,
            split_between=split_between,
        ),
    )


def _all_points_same(X: np.ndarray, indices: np.ndarray) -> bool:
    """Determine if all the training points X[indices] are the same.
    The algorithm finds the number of unique elements in each of the features
    of X[indices]. If there is a single unique value in each of the features
    or all the unique values are very close then all the points must be the same.
    """
    featurewise_unique = np.unique(X[indices], axis=0)
    return featurewise_unique.shape[0] == 1


def gini_impurity(y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Computes the Gini Impurity given labels `y`. Of the various ways to
    compute it, I chose the following formulation: 1 - the sum of the squared
    probability of each label. Note that `y` will often be a subset of the
    training labels, precisely, the labels of the elements of a node that may
    split.

    `w` is an array of point weights with the same shape as `y`. Its elements
    correspond to the weights of each of the elements of `y`.
    """
    _, inv_idxs, counts = np.unique(y, return_inverse=True, return_counts=True)
    if w is None:
        weighted_probs = counts / len(y)
    else:  # w is not None
        weighted_probs = np.array(
            [np.sum(w[inv_idxs == i]) for i in range(len(counts))]
        ) / np.sum(
            w
        )  # find the total weight in each class
    return 1 - sum([pi ** 2 for pi in weighted_probs])


def entropy(y: np.ndarray) -> float:
    """Computes the entropy given labels `y`. Used in similar contexts as Gini
    Impturity while constructing decision trees.
    """
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -1 * sum([pi * np.log(pi) for pi in probs])


def leaves_too_small(threshold: int) -> Callable[[np.ndarray, np.ndarray], bool]:
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
                np.ndarray.tolist, np.unique(y[tree.indices], return_counts=True),
            )
        )
        node = pydot.Node(
            str(clock),
            label=f"{list(zip(vals, counts))}",
            shape="box",
            fillcolor="green",
            style="filled",
        )
        vis.add_node(node)

    # otherwise make both children subtrees, then add edges
    else:
        left_node, clock = visualize(tree.left_subtree, vis, y, clock)
        right_node, clock = visualize(tree.right_subtree, vis, y, clock)

        node = pydot.Node(
            str(clock),
            label=f"Split feature {tree.split_feature}\n"
            + f"Split value {tree.split_value:.3f}",
        )
        vis.add_node(node)
        vis.add_edge(pydot.Edge(node, left_node, label="<"))
        vis.add_edge(pydot.Edge(node, right_node, label=">="))

    return node, clock + 1

def plurality(x: np.ndarray) -> int:
    """Returns the element of x that has plurality. `x` should have an integer
    dtype. This is used for classification tasks.
    """
    return np.argmax(np.bincount(x))
