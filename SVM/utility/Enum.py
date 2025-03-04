from enum import Enum


class KernelType(Enum):
    POLYNOMIAL = "polynomial"
    RBF = "radial basis function"


class LossFunctionType(Enum):
    HUBER = "humber-like loss"
    QUANTILE = "quantile loss"
    EPSILON_INTENSITIVE = "epsilon insensitive"
