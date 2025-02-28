import numpy as np
from SVM.utility.Enum import KernelType, LossFunctionType
from SVM.utility.Kernels import compute_kernel
from SVM.utility.LossFunction import huber_like_loss, log_sum_exp_loss, squared_hinge_loss


class SupportVectorRegression:
    def __init__(self, C=1.0, epsilon=0.1, kernel_type=KernelType.RBF, loss_function=LossFunctionType.HUBER, sigma=1.0,
                 degree=3, coef=1):
        self.C = C
        self.epsilon = epsilon
        self.kernel_type = kernel_type
        self.loss_function = loss_function
        self.sigma = sigma
        self.degree = degree
        self.coef = coef
        self.alpha = None
        self.b = None
        self.X_train = None

    def fit(self, X, Y, learning_rate=0.01, max_iter=100, smoothing_factor=0.9):
        self.X_train = X
        n_samples = X.shape[0]
        self.b = 0.0

        K = compute_kernel(X, X, kernel_type=self.kernel_type, sigma=self.sigma, degree=self.degree, coef=self.coef)
        self.alpha = np.random.randn(n_samples) * 0.01

        velocity = None
        training_loss = []

        for iteration in range(max_iter):
            gradients = self.compute_smooth_gradients(X, Y, epsilon=self.epsilon)

            if gradients is None:
                raise ValueError("compute_smooth_gradients ha restituito None, verifica le funzioni di perdita!")

            if velocity is None:
                velocity = gradients
            else:
                velocity = smoothing_factor * velocity + (1 - smoothing_factor) * gradients

            self.alpha -= learning_rate * velocity

            self.alpha = np.clip(self.alpha, 0, self.C)

            support_vector_indices = np.where((self.alpha > 1e-4) & (self.alpha < self.C))[0]
            if len(support_vector_indices) > 0:
                self.b = np.mean(Y[support_vector_indices] - np.dot(K[support_vector_indices], self.alpha))
            elif np.any(self.alpha > 0):
                self.b = np.mean(Y - np.dot(K, self.alpha))
            else:
                self.b = 0

            loss = np.mean((Y - np.dot(K, self.alpha) - self.b) ** 2)
            training_loss.append(loss)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")

        return training_loss

    def predict(self, X_test):
        if self.alpha is None or self.b is None:
            raise ValueError("Model has not been trained yet.")

        K_test = compute_kernel(X_test, self.X_train, kernel_type=self.kernel_type,
                                sigma=self.sigma, degree=self.degree, coef=self.coef)

        return np.dot(K_test, self.alpha) + self.b

    def compute_smooth_gradients(self, X, y, epsilon=0.1, delta=1.0):
        K = compute_kernel(X, X, kernel_type=self.kernel_type, sigma=self.sigma, degree=self.degree, coef=self.coef)
        y_pred = np.dot(K, self.alpha) + self.b

        if self.loss_function == LossFunctionType.HUBER:
            loss, grad = huber_like_loss(y, y_pred, epsilon, delta)
        elif self.loss_function == LossFunctionType.LOG_SUM_EXP:
            loss, grad = log_sum_exp_loss(y, y_pred, epsilon, self.b)
        elif self.loss_function == LossFunctionType.SQUARED_HINGE:
            loss, grad = squared_hinge_loss(y, y_pred, epsilon)
        else:
            raise ValueError("Tipo di funzione di perdita non supportata!")

        return np.dot(K.T, grad)
