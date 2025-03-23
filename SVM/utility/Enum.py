from enum import Enum


class KernelType(Enum):
    POLYNOMIAL = "polynomial"
    RBF = "radial basis function"
    LINEAR = "linear"
