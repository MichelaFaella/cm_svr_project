import numpy as np
from SVM.utility.Enum import KernelType, LossFunctionType
from SVM.utility.Kernels import compute_kernel
from SVM.utility.LossFunction import huber_like_loss, quantile_loss, epsilon_insensitive_loss


class SupportVectorRegression:
    def __init__(self, C=1.0, epsilon=0.1, kernel_type=KernelType.RBF, loss_function=LossFunctionType.HUBER, sigma=1.0,
                 degree=3, coef=1, learning_rate=0.1):
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
        self.learning_rate = learning_rate

    # -------------------------
    # Section smooth gradient
    # -------------------------

    def fit(self, X, Y, max_iter=100, smoothing_factor=0.9):
        self.X_train = X
        n_samples = X.shape[0]
        self.b = 0.0

        K = compute_kernel(X, X, kernel_type=self.kernel_type, sigma=self.sigma, degree=self.degree, coef=self.coef)
        self.alpha = np.zeros(n_samples) # More stable because with random alpha could be outside 0 and C

        # Velocity help to smooth the gradient
        velocity = np.zeros_like(self.alpha)
        training_loss = []

        for iteration in range(max_iter):
            gradients = self.compute_smooth_gradients(X, Y, self.alpha, epsilon=self.epsilon)

            if gradients is None:
                raise ValueError("compute_smooth_gradients ha restituito None, verifica le funzioni di perdita!")

            # Aggiorniamo la velocità (momentum) combinando il valore precedente con il nuovo gradiente
            velocity = smoothing_factor * velocity - self.learning_rate * gradients

            # Aggiorniamo i coefficienti alpha usando la velocità aggiornata
            self.alpha += velocity

            # Clipping of alpha
            self.alpha = np.clip(self.alpha, 0, self.C)

            support_vector_indices = np.where((self.alpha > 1e-6) & (self.alpha < self.C))[0]
            # There are support vector
            if len(support_vector_indices) > 0:
                self.b = np.mean(Y[support_vector_indices] - np.dot(K[support_vector_indices], self.alpha))
            elif np.any(self.alpha > 1e-6):
                self.b = np.mean(Y - np.dot(K, self.alpha))
            else:
                # Se non ci sono support vectors e α è tutto zero, usa una stima basata sulla media di Y
                self.b = np.mean(Y)

            # Calculate the loss
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

    def compute_smooth_gradients(self, X, y, alpha, epsilon=0.1, delta=1.0):
        K = compute_kernel(X, X, kernel_type=self.kernel_type, sigma=self.sigma, degree=self.degree, coef=self.coef)
        y_pred = np.dot(K, alpha) + self.b

        if self.loss_function == LossFunctionType.HUBER:
            loss, grad = huber_like_loss(y, y_pred, epsilon, delta)
        elif self.loss_function == LossFunctionType.EPSILON_INSENSITIVE:
            loss, grad = epsilon_insensitive_loss(y, y_pred)
        elif self.loss_function == LossFunctionType.QUANTILE:
            loss, grad = quantile_loss(y, y_pred)
        else:
            raise ValueError("Tipo di funzione di perdita non supportata!")

        return np.dot(K, (grad.reshape(-1, 1))).flatten()
