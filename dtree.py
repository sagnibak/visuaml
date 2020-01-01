import numpy as np

from dataclasses import dataclass
from typing import *

@dataclass(frozen=True)
class Tree:
    def __init__(self):
        raise NotImplementedError("Do not instantiate abstract class")

@dataclass(frozen=True)
class LeafNode(Tree):
    indices: List[int]

@dataclass(frozen=True)
class InternalNode(Tree):
    split_feature: int
    split_value: float
    left_subtree: Tree
    right_subtree: Tree

