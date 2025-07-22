import numpy as np
from sklearn.gaussian_process.kernels import (
    RBF,
    DotProduct,
    ConstantKernel,
    WhiteKernel,
    Kernel,
)
from sklearn.metrics.pairwise import polynomial_kernel, sigmoid_kernel


class Polynomial(Kernel):
    """
    Custom Polynomial kernel for Gaussian Process models.
    Wraps sklearn.metrics.pairwise.polynomial_kernel.
    """

    def __init__(self, degree=3, coef0=1, gamma=None):
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise NotImplementedError("Gradient is not implemented for this custom kernel.")
        return polynomial_kernel(X, Y, degree=self.degree, coef0=self.coef0, gamma=self.gamma)

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        return {"degree": self.degree, "coef0": self.coef0, "gamma": self.gamma}


class Sigmoid(Kernel):
    """
    Custom Sigmoid kernel for Gaussian Process models.
    Wraps sklearn.metrics.pairwise.sigmoid_kernel.
    """

    def __init__(self, coef0=1, gamma=None):
        self.coef0 = coef0
        self.gamma = gamma

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise NotImplementedError("Gradient is not implemented for this custom kernel.")
        return sigmoid_kernel(X, Y, coef0=self.coef0, gamma=self.gamma)

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return False

    def get_params(self, deep=True):
        return {"coef0": self.coef0, "gamma": self.gamma}


class KernelBuilder:
    """
    Constructs a composite kernel object from a trained SVR multi-kernel model
    and a list of kernel specifications.

    The class takes a trained model and a list of kernel parameters, and builds
    a composite kernel object from sklearn.gaussian_process.kernels. The final
    kernel is a linear combination of base kernels, weighted by the coefficients
    (mu values) learned by the model.
    """

    def __init__(self, kernel_parameters: list, model=None, weights=None):
        """
        Initializes the KernelBuilder.

        Args:
            kernel_parameters (list): A list of kernel specifications. Each
                                      specification can be a tuple `(str, dict)`
                                      or `(Kernel, dict)`.
            model: A trained SVR multi-kernel model object. The model is
                   expected to have 'mu_' or 'mu' attributes containing the
                   kernel weights, and a 'kronecker_kernel' boolean attribute.
            weights (list, optional): A list of weights to be used if model
                                      is not provided. Defaults to None.
        """
        if model is None and weights is None:
            raise ValueError("Either 'model' or 'weights' must be provided.")

        self.model = model
        self.kernel_parameters = kernel_parameters
        self.weights = weights

    def get_kernel_weights(self) -> np.ndarray:
        """
        Extracts kernel weights from the model or the provided weights list.

        If a model is provided, it first checks for the 'mu_' attribute,
        then falls back to 'mu'. If no model is provided, it uses the
        'weights' argument.

        Returns:
            np.ndarray: An array of kernel weights.

        Raises:
            AttributeError: If a model is provided but neither 'mu_' nor 'mu'
                            attribute is found.
        """
        if self.model is not None:
            if hasattr(self.model, "mu_"):
                return self.model.mu_
            if hasattr(self.model, "mu"):
                return self.model.mu
            raise AttributeError("Model does not have 'mu_' or 'mu' attribute.")
        return np.array(self.weights)

    def _translate_kernel(self, kernel_spec) -> Kernel:
        """
        Translates a kernel specification into a scikit-learn kernel object.

        Args:
            kernel_spec: A tuple specifying the kernel, e.g.,
                         ("rbf", {"gamma": 0.1}) or (RBF(), {}).

        Returns:
            Kernel: An instance of a scikit-learn kernel.

        Raises:
            ValueError: If the kernel name is not recognized.
        """
        kernel, params = kernel_spec
        if isinstance(kernel, str):
            kernel_name = kernel.lower()
            if kernel_name == "rbf":
                # scikit-learn's RBF kernel uses 'length_scale'. 'gamma' is
                # related by length_scale = 1 / sqrt(2 * gamma).
                if "gamma" in params:
                    gamma = params["gamma"]
                    params["length_scale"] = np.sqrt(1 / (2 * gamma)) if gamma > 0 else 1.0
                    del params["gamma"]
                return RBF(**params)
            if kernel_name == "linear":
                return DotProduct(**params)
            if kernel_name == "poly":
                return Polynomial(**params)
            if kernel_name == "sigmoid":
                return Sigmoid(**params)
            raise ValueError(f"Unknown kernel: {kernel}")
        if isinstance(kernel, Kernel):
            # If it's already a kernel object, just return it
            return kernel
        raise TypeError(f"Invalid kernel specification type: {type(kernel)}")

    def build_kernel(self) -> Kernel:
        """
        Builds the composite kernel.

        It combines the kernels specified in kernel_parameters, weighted by
        the 'mu' values from the model. Kernels with a weight of zero are
        excluded. If the model's 'kronecker_kernel' attribute is True, a
        WhiteKernel is added to the list of kernels.

        If a model is not provided, it uses the `weights` list and adds a
        WhiteKernel if the number of weights is one greater than the number
        of kernels.

        Returns:
            Kernel: The final composite scikit-learn kernel object.
        """
        weights = self.get_kernel_weights()
        k_params = self.kernel_parameters.copy()

        if self.model is not None:
            if hasattr(self.model, "kronecker_kernel") and self.model.kronecker_kernel:
                k_params.append((WhiteKernel(), {}))
        elif self.weights is not None:
            if len(self.weights) == len(k_params) + 1:
                k_params.append((WhiteKernel(), {}))

        if len(weights) != len(k_params):
            raise ValueError(
                "The number of weights must match the number of kernels."
            )

        weighted_kernels = []
        for weight, kernel_spec in zip(weights, k_params):
            if weight > 0:
                kernel = self._translate_kernel(kernel_spec)
                weighted_kernels.append(weight * kernel)

        if not weighted_kernels:
            # Return a default kernel if all weights are zero
            return ConstantKernel(0.0, constant_value_bounds="fixed")

        # Sum of weighted kernels
        final_kernel = weighted_kernels[0]
        for i in range(1, len(weighted_kernels)):
            final_kernel += weighted_kernels[i]

        return final_kernel
