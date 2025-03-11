import numpy as np
from SVM.utility.Enum import KernelType, LossFunctionType
from SVM.utility.Kernels import compute_kernel
from SVM.utility.LossFunction import epsilon_insensitive_loss


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

    def fit(self, X, Y, max_iter=100, smoothing_factor=0.9, mu=0.01):
        """
        Train the Support Vector Regression model using a smooth optimization approach.

        Parameters:
        - X: Training data (numpy array).
        - Y: Target values (numpy array).
        - max_iter: Number of training iterations.
        - smoothing_factor: Momentum factor for accelerated gradient descent.
        - mu: Regularization parameter to smooth the gradients.

        Returns:
        - A list containing the loss at each iteration.
        """

        self.X_train = X
        n_samples = X.shape[0]
        self.b = 0.0

        # Compute kernel matrix
        K = compute_kernel(X, X, kernel_type=self.kernel_type, sigma=self.sigma, degree=self.degree, coef=self.coef)
        self.alpha = np.zeros(n_samples)  # Initialize with zero for stability

        # Initialize momentum velocity
        velocity = np.zeros_like(self.alpha)
        training_loss = []

        for iteration in range(max_iter):
            prev_alpha = np.copy(self.alpha)  # Store previous alpha values

            # Compute smoothed gradients
            gradients = self.compute_smooth_gradients(X, Y, self.alpha, epsilon=self.epsilon, mu=mu)

            if gradients is None:
                raise ValueError("compute_smooth_gradients returned None. Check the loss function implementation!")

            # Nesterov's Accelerated Gradient update
            self.alpha += smoothing_factor * (self.alpha - prev_alpha)
            velocity = smoothing_factor * velocity - self.learning_rate * gradients
            self.alpha += velocity

            # Clip alpha values to stay within [0, C] bounds
            self.alpha = np.clip(self.alpha, 0, self.C)

            # Compute bias term 'b' using only support vectors
            support_vector_indices = np.where((self.alpha > 1e-6) & (self.alpha < self.C))[0]
            if len(support_vector_indices) > 0:
                self.b = np.mean(Y[support_vector_indices] - np.dot(K[support_vector_indices], self.alpha))
            elif np.any(self.alpha > 1e-6):
                self.b = np.mean(Y - np.dot(K, self.alpha))
            else:
                self.b = np.mean(Y)  # Default bias if no support vectors are found

            # Compute training loss
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

    def compute_smooth_gradients(self, X, y, alpha, epsilon=0.1, mu=0.01):
        """
        Compute the smoothed gradient for Support Vector Regression.

        Parameters:
        - X: Input data (numpy array).
        - y: Target values (numpy array).
        - alpha: Current weight coefficients.
        - epsilon: Insensitivity parameter.
        - mu: Smoothing parameter for gradient stabilization.

        Returns:
        - The computed gradient with regularization.
        """

        # Compute kernel matrix
        K = compute_kernel(X, X, kernel_type=self.kernel_type, sigma=self.sigma, degree=self.degree, coef=self.coef)
        y_pred = np.dot(K, alpha) + self.b

        # Compute smoothed loss and gradient
        loss, grad = epsilon_insensitive_loss(y, y_pred, epsilon=epsilon, mu=mu)

        if grad is None:
            raise ValueError("Gradient computation failed in epsilon_insensitive_loss.")

        # Apply a smoothing term to the gradient for stability
        smoothing_term = mu * alpha
        return np.dot(K, grad) - smoothing_term

