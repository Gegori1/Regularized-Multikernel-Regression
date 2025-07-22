# %%

import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels


class svr_sdp_multi_kernel_l1:
    def __init__(
            self, C: float=1,
            epsilon: float=1,
            tau: float=1,
            kernel_params: list=[("linear", {})],
            c: float=1,
            verbose: bool=False,
            trace_min: bool=True,
            trace_min_factor: int=1,
            kronecker_kernel: bool=False,
        ):
        
        self.C = C
        self.epsilon = epsilon
        self.tau = tau
        self.kernel_params = kernel_params
        self.c = c
        self.verbose = verbose
        self.trace_min = trace_min
        self.trace_min_factor = trace_min_factor
        self.kronecker_kernel = kronecker_kernel
        
        
    def check_x_y(self, X, y):
        X, y = check_X_y(X, y)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        return X, y

    def delta_kronecker_kernel(self, X, Y: None):
        if Y is None:
            Y = X.copy()
        else:
            try:
                Y = np.asarray(Y)
            except TypeError:
                raise TypeError(f"Y could not be converted to numpy array. Y is of type {type(Y)}")
            
        if np.array_equal(X, Y):
            return np.eye(X.shape[0])
        
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Reshape X and Y for broadcasting
        X_reshaped = X[:, np.newaxis, :]  # Shape (n, 1, p)
        Y_reshaped = Y[np.newaxis, :, :]  # Shape (1, m, p)

        # Perform element-wise comparison using broadcasting
        comparison_matrix = (X_reshaped == Y_reshaped) # Shape (n, m, p)

        # Check if all elements along the last axis (axis=2) are True for each (i, j) pair
        kernel_matrix = np.all(comparison_matrix, axis=2).astype(int) # Shape (n, m)


        return kernel_matrix
    
    def param(self, X):
        m = X.shape[0]
        
        # else:
        K = 0
        if self.kronecker_kernel:
            mu_len = len(self.kernel_params) + 1
        else:
            mu_len = len(self.kernel_params)
            
        self.mu = cp.Variable(mu_len, name="mu", pos=True)

        trace = 0
        for i, kern in enumerate(self.kernel_params):
            if isinstance(kern[0], str):
                kernel = pairwise_kernels(X, metric=kern[0], filter_params=True, **kern[1])
            else:
                kernel = kern[0](X, **kern[1])
            K += self.mu[i] * kernel
            trace += np.trace(kernel)
        # add identity matrix
        if self.kronecker_kernel:
            K += self.mu[-1] * np.eye(m)
        trace += m
            
        self.trace = trace

        self.K = K
        onev = np.ones((m, 1))
        onev_mu = np.ones((self.mu.shape[0], 1))
        
        return onev, onev_mu, m
    
    def solve(self, X, y):
            
            onev, onev_mu, m = self.param(X)
            
            if self.trace_min:
                self.c = self.trace * self.trace_min_factor
            
            # if self.c < self.trace:
            #     raise ValueError(f"c ({self.c}) must be greater than the trace of the kernel matrix ({self.trace}). Choose a higher value for c.")
            
            # variables
            self.delta = cp.Variable((m, 1), name="delta", pos=True)
            self.delta_ = cp.Variable((m, 1), name="delta_", pos=True)
            self.eta = cp.Variable((m, 1), name="eta", pos=True)
            self.lamda = cp.Variable(1, name="lambda")
            self.t = cp.Variable(1, name="t")
            
            # Problem
            self.objective = cp.Minimize(self.t)
            
            self.alpha = - self.epsilon * onev + y + self.eta - self.delta + self.lamda * onev
            
            self.constraints_lmi = cp.bmat(
                [
                    [
                        cp.psd_wrap(self.K), 
                        self.alpha],
                    [
                        self.alpha.T,
                        cp.reshape(self.t, (1, 1), order='F') - 2*self.C*(self.delta + self.delta_).T @ onev - self.tau * self.mu @ onev_mu
                        # cp.reshape(self.t, (1, 1), order='F') - 2*self.C*(self.delta + self.delta_).T @ onev - self.tau * cp.trace(self.K)
                        # cp.reshape(self.t, (1, 1), order='F') - 2*self.C*(self.delta + self.delta_).T @ onev
                    ]
                ]
            )
            
            self.constraints = [
                self.constraints_lmi >> 0,
                self.K >> 0,
                self.eta - (self.delta_ + self.delta) <= 2 * self.epsilon * onev,
                # self.mu.T @ onev_mu <= self.tau
                # cp.trace(self.K) <= self.tau
            ]
            
            problem = cp.Problem(self.objective, self.constraints)
            
            # solve problem
            problem.solve(verbose=self.verbose, solver=cp.MOSEK)
            self.status = problem.status
            if self.status != "optimal":
                return self
            self.objective = problem.value
            
            self.beta = np.linalg.pinv(self.K.value) @ self.alpha.value
                
            return self
        
    def non_optimal(self, y):
        self.sup_num = 0
        self.support_vectors = np.array([0])
        self.support_labels = np.array([0])
        self.support_beta = np.array([0])
        self.b = np.mean(y)
        self.mu = 0
    
    def compute_indx(self):
        # # Use dual-dual kkt conditions to filter support vectors
        # def indx_filter(k):
        #     return (
        #     (self.delta.value.flatten() <= k) & 
        #     (self.delta_.value.flatten() <= k) & 
        #     (self.eta.value.flatten() >= k) & 
        #     (np.abs(self.eta.value.flatten() - 2*self.epsilon) >= k)
        #     )
            
        def indx_filter(k):
            return (
            (self.delta.value.flatten() > k) |
            (self.delta_.value.flatten() > k) |
            (self.eta.value.flatten() <= k) | 
            (np.abs(self.eta.value.flatten() - 2*self.epsilon) <= k)
            )
        
        if self.C <= 1e-5:
            indx = indx_filter(1e-4)

        elif self.C < 1e3:
            indx = indx_filter(1e-3)

        else:
            indx = indx_filter(self.C / 1e5)
            
        self.indx = indx
            
        return indx
    
    def compute_b(self, indx):
        m = self.support_vectors.shape[0]
        # calculate epsilon
        support_kernel = 0
        epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon, self.epsilon)
        for mu, kern in zip(self.mu, self.kernel_params):
            if isinstance(kern[0], str):
                support_kernel += mu*pairwise_kernels(self.support_vectors, metric=kern[0], filter_params=True, **kern[1])
            else:
                support_kernel += mu*kern[0](self.support_vectors, **kern[1])
        # add identity matrix
        if self.kronecker_kernel:
            support_kernel += self.mu[-1] * np.eye(m)
        
        # Identify bounded and unbounded support vectors
        def unbounded_filter(k):
            return (
            (self.delta.value.flatten() <= k) & 
            (self.delta_.value.flatten() <= k) & 
            ((np.abs(self.eta.value.flatten() - 2*self.epsilon) <= k) | (self.eta.value.flatten() <= k))
            )
        
        if self.C <= 1e-5:
            unbounded_sv = unbounded_filter(1e-4)[indx]
        elif self.C < 1e3:
            unbounded_sv = unbounded_filter(1e-3)[indx]
        else:
            unbounded_sv = unbounded_filter(self.C / 1e5)[indx]
        self.unbounded_sv = unbounded_sv
        
        if np.any(unbounded_sv):
            self.b = np.mean(
                epsilon_beta[unbounded_sv] - self.support_beta.T @ support_kernel[:, unbounded_sv] + self.support_labels[unbounded_sv]
            )
        else:
            self.b = np.mean(
                epsilon_beta - self.support_beta.T @ support_kernel + self.support_labels
            )
        
        
    def fit(self, X, y):
        
        # check X and y
        X, y = self.check_x_y(X, y)
        
        # solve
        try:
            self.solve(X, y)
        except cp.SolverError as e:
            self.status = "infeasible"
            print(f"Solver failed: {e}")
        except Exception as e:
            self.status = "error"
            print(f"An unexpected error occurred: {e}")
        
        if self.status != "optimal":
            self.non_optimal(y)
            return self
        

        # filter mus
        self.mu = np.where(self.mu.value <= 1e-5, 0, self.mu.value)
        
        # compute support values index
        indx = self.compute_indx()

        # filter supports
        if np.sum(indx) != 0:
            self.sup_num = np.sum(indx)
            self.support_vectors = X[indx, :]
            self.support_labels = y[indx, :]
            self.support_beta = self.beta[indx, :]
            # compute b
            self.compute_b(indx)
        else:
            self.non_optimal(y)        

        return self
    
    
    def predict(self, X):
        return self.decision_function(X)
    
    def decision_function(self, X):
        if self.sup_num == 0:
            return np.ones(X.shape[0]) * self.b
        K = 0
        for mu, kern in zip(self.mu, self.kernel_params):
            if mu == 0:
                continue
            if isinstance(kern[0], str):
                K += mu*pairwise_kernels(self.support_vectors, X, metric=kern[0], filter_params=True, **kern[1])
            else:
                K += mu*kern[0](self.support_vectors, X, **kern[1])
        
        # add identity matrix
        if self.kronecker_kernel:
            K += self.mu[-1] * self.delta_kronecker_kernel(self.support_vectors, X)
        return self.support_beta.T @ K + self.b
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def get_params(self):
        return {"C": self.C, "epsilon": self.epsilon}
    
    def set_params(self, **params):
        self.C = params["C"]
        self.epsilon = params["epsilon"]
        return self
# %%
if __name__ == "__main__":
    
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    import matplotlib.pyplot as plt
    
    # Create a regression dataset
    X, y = make_regression(
        n_samples=100,
        n_features=1, 
        noise=20,
        random_state=1
    )
    
    # np.random.seed(0)
    # X = np.linspace(-5, 5, 200).reshape(-1, 1)
    # y = np.sinc(X).flatten() + np.random.normal(0, 0.1, X.shape[0])

    kernel_params = [
        # ("linear", {}),
        # ("poly", {"degree": 2}),
        # ("rbf", {"gamma":1e-1}),
        ("rbf", {"gamma": 1e2})
    ]

    C = 1e2
    epsilon = 1e1
    gamma = 1e2
# %%

    # Create an instance of svr_sdp_multi_kernel
    model_sdp_multi = svr_sdp_multi_kernel_l1(
        C = C,
        epsilon = epsilon,
        tau = 1,
        kernel_params = kernel_params,
        # c = 10,
        verbose=True,
        kronecker_kernel=True
    )
    
    model_sklrn = SVR(
        C=C, 
        epsilon=epsilon,
        kernel="rbf",
        gamma=gamma
    )
    

    
    model_sdp_multi.fit(X, y)
    model_sklrn.fit(X, y)
    
    y_pred_multi = model_sdp_multi.predict(X)
    y_pred_sklrn = model_sklrn.predict(X)


# %%

    plt.title("Support vectors")
    plt.scatter(X, y)
    plt.scatter(model_sdp_multi.support_vectors, model_sdp_multi.support_labels, color="red", marker="o", label="SDP Support Vectors")
    plt.scatter(X[model_sklrn.support_], y[model_sklrn.support_], color="orange", marker=".", label="Sklearn Support Vectors")
    plt.legend()
    plt.show()
    
    support_indices_sklearn = model_sklrn.support_
    dual_coef_full_sklearn = np.zeros_like(y)
    dual_coef_full_sklearn[support_indices_sklearn] = model_sklrn.dual_coef_

    d = np.linspace(-2, 2, y.shape[0])
    plt.scatter(d, dual_coef_full_sklearn, label="sklearn", marker="*")
    plt.scatter(d, model_sdp_multi.beta.flatten(), label="sdp_multi_l1", marker=".", color="orange")
    # plt.ylim(-0.0001, 0.0001)
    plt.legend()
    plt.show()
    
    plt.scatter(X, y)
    plt.scatter(X, y_pred_sklrn, label="sklearn")
    plt.scatter(X, y_pred_multi, marker=".", label="sdp_multi")
    plt.legend()
    plt.title("Predictions")
    plt.show()
    
    
    print("b from model_sklearn:", model_sklrn.intercept_[0])
    print("b from model_sdp_multi", model_sdp_multi.b)
# %%

    def print_accuracy(y, y_pred, model):
        print(f"{model} accuracy: {mean_absolute_error(y, y_pred.flatten())}")

    print_accuracy(y, y_pred_sklrn, "sklearn")
    print_accuracy(y, y_pred_multi, "sdp_multi_l1")

