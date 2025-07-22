# %%
import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels


class svr_qcqp_multi_kernel_l1:
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


        X_reshaped = X[:, np.newaxis, :]
        Y_reshaped = Y[np.newaxis, :, :]

        comparison_matrix = (X_reshaped == Y_reshaped)

        kernel_matrix = np.all(comparison_matrix, axis=2).astype(int)


        return kernel_matrix
    
    def param(self, X):
        m = X.shape[0]
        
        onev = np.ones((m, 1))
        
        # else:
        kernels = []
        trace = 0
        for i, kern in enumerate(self.kernel_params):
            if isinstance(kern[0], str):
                kernel = pairwise_kernels(X, metric=kern[0], filter_params=True, **kern[1])
            else:
                kernel = kern[0](X, **kern[1])
            kernels.append(kernel)
            trace += np.trace(kernel)
        # add identity matrix
        if self.kronecker_kernel:
            kernels.append(np.eye(m))
            trace += m
            
        self.kernels = kernels
        self.trace = trace
        
        return onev, m
    
    def solve(self, X, y):
            
            onev, m = self.param(X)
            
            if self.trace_min:
                self.c = self.trace * self.trace_min_factor
            
            # variables
            self.t = cp.Variable(1, name="t")
            self.beta = cp.Variable((m, 1), name="beta")

            
            # problem
            self.objective = cp.Maximize(-2*self.epsilon * cp.abs(self.beta.T) @ onev + 2 * self.beta.T @ y - self.tau * self.t)
            # self.objective = cp.Maximize(-2*self.epsilon * cp.abs(self.beta.T) @ onev + 2 * self.beta.T @ y)
            
            
            self.constraints_kernels = [
                # cp.quad_form(self.beta, cp.psd_wrap(K)) <= self.tau for K in self.kernels
                self.t >= cp.quad_form(self.beta, cp.psd_wrap(K)) / np.trace(K) for K in self.kernels
            ]
            
            self.constraints = [
                *self.constraints_kernels,
                self.beta.T @ onev == 0,
                self.beta <= self.C,
                self.beta >= - self.C
            ]
            
            self.problem = cp.Problem(self.objective, self.constraints)
            
            # solve problem
            self.problem.solve(verbose=self.verbose, solver=cp.MOSEK)
            self.status = self.problem.status
            if self.status != "optimal":
                return self
            self.objective = self.problem.value
            
            self.beta = self.beta.value
            
            self.mu = np.asarray([i.dual_value[0] for i in self.constraints_kernels])
                
            return self
        
    def non_optimal(self, y):
        self.sup_num = 0
        self.support_vectors = np.array([0])
        self.support_labels = np.array([0])
        self.support_beta = np.array([0])
        self.b = np.mean(y)
        self.mu = 0
    
    def compute_indx(self):
        
        beta_1 = self.beta.flatten()
        
        indx = np.abs(beta_1) > 1e-5
        
        self.indx = indx
            
        return indx
    
    def compute_b(self):
        m = self.support_vectors.shape[0]
        # calculate epsilon
        epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon, self.epsilon)
        # compute kernel
        support_kernel = 0
        for mu, kern, k in zip(self.mu, self.kernel_params, self.kernels):
            if isinstance(kern[0], str):
                support_kernel += mu*pairwise_kernels(self.support_vectors, metric=kern[0], filter_params=True, **kern[1]) / np.trace(k)
            else:
                support_kernel += mu*kern[0](self.support_vectors, **kern[1]) / np.trace(k)
        # add identity matrix
        if self.kronecker_kernel:
            support_kernel += self.mu[-1] * np.eye(m) / np.trace(self.kernels[-1])
        
        
        bounded_sv = (np.abs(self.support_beta.flatten() - self.C) < 1e-5) | (np.abs(self.support_beta.flatten() + self.C) < 1e-5)
        unbounded_sv = ~bounded_sv
        
        if np.any(unbounded_sv):
            self.b = np.mean(
                epsilon_beta[unbounded_sv] - self.support_beta.T @ support_kernel[:, unbounded_sv] + self.support_labels[unbounded_sv]
            )
        else:
            self.b = np.mean(
                epsilon_beta - self.support_beta.T @ support_kernel + self.support_labels
            )
            
        self.unbounded_sv = unbounded_sv
        
        
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
        if any(self.mu >= 1e-4):
            self.mu = np.where(self.mu <= 1e-4, 0, self.mu)
        else:
            self.mu = np.where(self.mu <= 1e-7, 0, self.mu)
        
        # compute support values index
        indx = self.compute_indx()
        # filter supports
        if np.sum(indx) != 0:
            self.sup_num = np.sum(indx)
            self.support_vectors = X[indx, :]
            self.support_labels = y[indx, :]
            self.support_beta = self.beta[indx, :]
            # compute b
            self.compute_b()
        else:
            self.non_optimal(y)        

        return self
    
    
    def predict(self, X):
        return self.decision_function(X)
    
    def decision_function(self, X):
        if self.sup_num == 0:
            return np.ones(X.shape[0]) * self.b
        K = 0
        for mu, kern, k in zip(self.mu, self.kernel_params, self.kernels):
            if mu == 0:
                continue
            if isinstance(kern[0], str):
                K += mu*pairwise_kernels(self.support_vectors, X, metric=kern[0], filter_params=True, **kern[1]) / np.trace(k)
            else:
                K += mu*kern[0](self.support_vectors, X, **kern[1]) / np.trace(k)
        
        # add identity matrix
        if self.kronecker_kernel:
            K += self.mu[-1] * self.delta_kronecker_kernel(self.support_vectors, X) / np.trace(self.kernels[-1])
        return self.support_beta.T @ K + self.b
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def get_params(self):
        return {"C": self.C, "epsilon": self.epsilon}
    
    def set_params(self, **params):
        self.C = params["C"]
        self.epsilon = params["epsilon"]
        return self
# %%1
if __name__ == "__main__":
    
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    import matplotlib.pyplot as plt
    # from svr_sdp_multi_kernel_l1 import svr_sdp_multi_kernel_l1
    
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ExpSineSquared
    
    # Create a regression dataset
    X, y = make_regression(
        n_samples=100,
        n_features=1, 
        noise=2,
        random_state=1
    )

    kernel_params = [
        ("linear", {}),
        ("poly", {"degree": 2}),
        ("poly", {"degree": 3}),
        ("poly", {"degree": 4}),
        ("rbf", {"gamma":1e-2}),
        ("rbf", {"gamma": 1e-1}),
        ("rbf", {"gamma": 1e0}),
        ("rbf", {"gamma": 1e1}),
        ("rbf", {"gamma": 1e2}),
        ("rbf", {"gamma": 1e3}),
        ("rbf", {"gamma": 1e4}),
        ("sigmoid", {"gamma": 1e-1}),
        ("sigmoid", {"gamma": 1e0}),
        ("sigmoid", {"gamma": 1e1}),
        ("sigmoid", {"gamma": 1e2}),
        ("sigmoid", {"gamma": 1e3}),
        # (ExpSineSquared(periodicity=1), {}),
        # (ExpSineSquared(periodicity=3), {}),
        # (RBF(length_scale=0.1), {}),
        # (RBF(length_scale=10), {}),
        # (RBF(length_scale=50), {}),
    ]

    C = 1e3
    epsilon = 1e-2
    gamma = 1e2
    tau = 1e4
# %%

    # Create an instance of svr_sdp_multi_kernel
    model_qcqp_multi = svr_qcqp_multi_kernel_l1(
        C = C,
        epsilon = epsilon,
        tau = tau,
        kernel_params = kernel_params,
        verbose=False,
        kronecker_kernel=True
    )
    
# %%
    
    # model_sdp_multi = svr_sdp_multi_kernel_l1(
    #     C = C,
    #     epsilon = epsilon,
    #     tau = tau,
    #     kernel_params = kernel_params,
    #     verbose=False,
    #     kronecker_kernel=True
    # )

    model_sklrn = SVR(
        C=C, 
        epsilon=epsilon,
        kernel="linear",
        gamma=gamma
    )
    
    model_qcqp_multi.fit(X, y)
    # model_sdp_multi.fit(X, y)
    model_sklrn.fit(X, y)
    
    # %%
    
    y_pred_multi_qcqp = model_qcqp_multi.predict(X)
    # y_pred_multi_sdp = model_sdp_multi.predict(X)
    y_pred_sklrn = model_sklrn.predict(X);

    model_qcqp_multi.problem.solver_stats.solve_time


    plt.title("Support vectors")
    plt.scatter(X, y)
    plt.scatter(model_qcqp_multi.support_vectors, model_qcqp_multi.support_labels, color="red", marker="o", label="QCQP Support Vectors")
    # plt.scatter(model_sdp_multi.support_vectors, model_sdp_multi.support_labels, color="orange", marker="+", label="SDP Support Vectors")
    plt.scatter(X[model_sklrn.support_], y[model_sklrn.support_], color="orange", marker=".", label="Sklearn Support Vectors")
    plt.legend()
    plt.show()
    
    support_indices_sklearn = model_sklrn.support_
    dual_coef_full_sklearn = np.zeros_like(y)
    dual_coef_full_sklearn[support_indices_sklearn] = model_sklrn.dual_coef_

    d = np.linspace(-2, 2, y.shape[0])
    # plt.scatter(d, dual_coef_full_sklearn, label="sklearn", marker="+")
    # plt.scatter(d, model_sdp_multi.beta.flatten(), label="sdp_multi_l1", marker="o", color="tab:green")
    plt.scatter(d, model_qcqp_multi.beta.flatten(), label="qcqp_multi_l1", marker=".", color="orange")
    # plt.ylim(-0.0001, 0.0001)
    plt.legend()
    plt.show()
    
    plt.scatter(X, y)
    plt.scatter(X, y_pred_sklrn, label="sklearn")
    plt.scatter(X, y_pred_multi_qcqp, marker="+", label="qcqp_multi")
    # plt.scatter(X, y_pred_multi_sdp, marker=".", label="sdp_multi")
    plt.legend()
    plt.title("Predictions")
    plt.show()
    
    
    print("b from model_sklearn:", model_sklrn.intercept_[0])
    print("b from model_qcqp_multi", model_qcqp_multi.b)
    # print("b from model_sdp_multi", model_sdp_multi.b)

    
    def print_accuracy(y, y_pred, model):
        print(f"{model} accuracy: {mean_absolute_error(y, y_pred.flatten())}")

    print_accuracy(y, y_pred_sklrn, "sklearn")
    print_accuracy(y, y_pred_multi_qcqp, "qcqp_multi_l1")
    # print_accuracy(y, y_pred_multi_sdp, "sdp_multi_l1")

# %%
