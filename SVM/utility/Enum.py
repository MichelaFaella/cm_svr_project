from enum import Enum


class KernelType(Enum):
    POLYNOMIAL = "polynomial"
    RBF = "radial basis function"


class LossFunctionType(Enum):
    HUBER = "humber-like loss"
    LOG_SUM_EXP = "softmax-like loss"
    SQUARED_HINGE = "squared hinge loss"
