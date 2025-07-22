# %%
import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels
import warnings

class SvrQcqpMultiKernelL1Trace:
    def __init__(
            self, C: float=1,
            epsilon: float=1,
            tau: float=1,
            kernel_params: list=[("linear", {})],
            verbose: bool=False,
            kronecker_kernel: bool=False,
        ):
        
        self.C = C
        self.epsilon = epsilon
        self.tau = tau
        self.kernel_params = kernel_params
        self.verbose = verbose
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
            
            # variables
            self.beta = cp.Variable((m, 1), name="beta")

            
            # problem
            self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) + 2 * self.beta.T @ y)
            
            self.constraints_kernels = [
                cp.quad_form(self.beta, cp.psd_wrap(K)) <= self.tau * np.trace(K) for K in self.kernels
            ]
            
            self.constraints = [
                *self.constraints_kernels,
                self.beta.T @ onev == 0,
                cp.abs(self.beta) <= self.C,
            ]
            
            warnings.filterwarnings("ignore", message="Forming a nonconvex expression quad_form\\(x, indefinite\\)\\.")
            self.problem = cp.Problem(self.objective, self.constraints)
            
            # solve problem
            self.problem.solve(verbose=self.verbose, solver=cp.MOSEK)
            self.status = self.problem.status
            if self.status != "optimal":
                return self
            self.objective = self.problem.value
            
            self.beta_ = self.beta.value
            
            self.mu_ = np.asarray([i.dual_value[0] for i in self.constraints_kernels])
                
            return self
        
    def non_optimal(self, y):
        self.sup_num_ = 0
        self.support_vectors = np.array([0])
        self.support_labels = np.array([0])
        self.support_beta = np.array([0])
        self.b_ = np.mean(y)
        self.mu_ = 0
    
    def compute_indx(self):
        
        beta_1 = self.beta_.flatten()
        
        indx = np.abs(beta_1) > 1e-5
            
        return indx
    
    def compute_b(self):
        m = self.support_vectors.shape[0]
        # calculate epsilon
        epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon, self.epsilon)
        # compute kernel
        support_kernel = 0
        for mu, kern in zip(self.mu_, self.kernel_params):
            if isinstance(kern[0], str):
                support_kernel += mu*pairwise_kernels(self.support_vectors, metric=kern[0], filter_params=True, **kern[1])
            else:
                support_kernel += mu*kern[0](self.support_vectors, **kern[1])
        # add identity matrix
        if self.kronecker_kernel:
            support_kernel += self.mu_[-1] * np.eye(m)
        
        
        bounded_sv = (np.abs(self.support_beta.flatten() - self.C) < 1e-5) | (np.abs(self.support_beta.flatten() + self.C) < 1e-5)
        unbounded_sv = ~bounded_sv
        
        if np.any(unbounded_sv):
            self.b_ = np.mean(
                epsilon_beta[unbounded_sv] - self.support_beta.T @ support_kernel[:, unbounded_sv] + self.support_labels[unbounded_sv]
            )
        else:
            self.b_ = np.mean(
                epsilon_beta - self.support_beta.T @ support_kernel + self.support_labels
            )
            
        self.unbounded_sv = unbounded_sv
        
        
    def fit(self, X, y):
        
        # check X and y
        X, y = self.check_x_y(X, y)
        
        # solve
        try:
            self.solve(X, y)
            self.status_ = self.problem.status
        except cp.SolverError as e:
            self.status_ = "infeasible"
            print(f"Solver failed: {e}")
        except Exception as e:
            self.status_ = "error"
            print(f"An unexpected error occurred: {e}")
        
        if self.status_ != "optimal":
            self.non_optimal(y)
            return self
        

        # filter mus
        if any(self.mu_ >= 1e-4):
            self.mu_ = np.where(self.mu_ <= 1e-4, 0, self.mu_)
        else:
            self.mu_ = np.where(self.mu_ <= 1e-7, 0, self.mu_)
        
        # compute support values index
        indx = self.compute_indx()
        # filter supports
        if np.sum(indx) != 0:
            self.sup_num_ = np.sum(indx)
            self.support_vectors = X[indx, :]
            self.support_labels = y[indx, :]
            self.support_beta = self.beta_[indx, :]
            # compute b
            self.compute_b()
        else:
            self.non_optimal(y)        

        return self
    
    
    def predict(self, X):
        return self.decision_function(X).flatten()
    
    def decision_function(self, X):
        m = X.shape[0]
        if self.sup_num_ == 0:
            return np.ones(X.shape[0]) * self.b_
        K = 0
        for mu, kern in zip(self.mu_, self.kernel_params):
            if mu == 0:
                continue
            if isinstance(kern[0], str):
                K += mu*pairwise_kernels(self.support_vectors, X, metric=kern[0], filter_params=True, **kern[1])
            else:
                K += mu*kern[0](self.support_vectors, X, **kern[1])
        
        # add identity matrix
        if self.kronecker_kernel:
            K += self.mu_[-1] * self.delta_kronecker_kernel(self.support_vectors, X)
        if np.all(self.mu_ == 0):
            return np.full(m, self.b_)
        return self.support_beta.T @ K + self.b_
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def get_params(self):
        return {"C": self.C, "epsilon": self.epsilon}
    
    def set_params(self, **params):
        self.C = params["C"]
        self.epsilon = params["epsilon"]
        return self
    
# %%

class SvrQcqpMultiKernelL1EpigraphTrace(SvrQcqpMultiKernelL1Trace):
    
    def solve(self, X, y):
            
            onev, m = self.param(X)
            
            # variables
            self.t = cp.Variable(1, name="teta")
            self.beta = cp.Variable((m, 1), name="beta")


            # problem
            self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) + 2 * self.beta.T @ y - self.tau * self.t)
            
            self.constraints_kernels = [
                self.t >= cp.quad_form(self.beta, cp.psd_wrap(K)) / np.trace(K) for K in self.kernels
            ]

            self.constraints = [
                *self.constraints_kernels,
                self.beta.T @ onev == 0,
                cp.abs(self.beta) <= self.C,
            ]

            warnings.filterwarnings("ignore", message="Forming a nonconvex expression quad_form\\(x, indefinite\\)\\.")
            self.problem = cp.Problem(self.objective, self.constraints)

            # solve problem
            self.problem.solve(verbose=self.verbose, solver=cp.MOSEK)
            self.status_ = self.problem.status
            if self.status_ != "optimal":
                return self
            self.objective_ = self.problem.value

            self.beta_ = self.beta.value

            self.mu_ = np.asarray([i.dual_value[0] for i in self.constraints_kernels])

            return self
        
    def compute_b(self):
        m = self.support_vectors.shape[0]
        # calculate epsilon
        epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon, self.epsilon)
        # compute kernel
        support_kernel = 0
        for mu, kern, k in zip(self.mu_, self.kernel_params, self.kernels):
            if isinstance(kern[0], str):
                support_kernel += mu*pairwise_kernels(self.support_vectors, metric=kern[0], filter_params=True, **kern[1]) / np.trace(k)
            else:
                support_kernel += mu*kern[0](self.support_vectors, **kern[1]) / np.trace(k)
        # add identity matrix
        if self.kronecker_kernel:
            support_kernel += self.mu_[-1] * np.eye(m) / m
        
        
        bounded_sv = (np.abs(self.support_beta.flatten() - self.C) < 1e-5) | (np.abs(self.support_beta.flatten() + self.C) < 1e-5)
        unbounded_sv = ~bounded_sv
        
        if np.any(unbounded_sv):
            self.b_ = np.mean(
                epsilon_beta[unbounded_sv] - self.support_beta.T @ support_kernel[:, unbounded_sv] + self.support_labels[unbounded_sv]
            )
        else:
            self.b_ = np.mean(
                epsilon_beta - self.support_beta.T @ support_kernel + self.support_labels
            )
            
        self.unbounded_sv = unbounded_sv
        
        
    def decision_function(self, X):
        m = X.shape[0]
        if self.sup_num_ == 0:
            return np.ones(X.shape[0]) * self.b_
        K = 0
        for mu, kern, k in zip(self.mu_, self.kernel_params, self.kernels):
            if mu == 0:
                continue
            if isinstance(kern[0], str):
                K += mu*pairwise_kernels(self.support_vectors, X, metric=kern[0], filter_params=True, **kern[1]) / np.trace(k)
            else:
                K += mu*kern[0](self.support_vectors, X, **kern[1]) / np.trace(k)
        
        # add identity matrix
        if self.kronecker_kernel:
            K += self.mu_[-1] * self.delta_kronecker_kernel(self.support_vectors, X) / m
        if np.all(self.mu_ == 0):
            return np.full(m, self.b_)
        return self.support_beta.T @ K + self.b_
        


# %%
if __name__ == "__main__":
    
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR as SklearnSVR # Alias
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    
    from sklearn.gaussian_process.kernels import (
        Matern, ExpSineSquared, RationalQuadratic, WhiteKernel, ConstantKernel
    )
    
    X, y_reg = make_regression( 
        n_samples=200, 
        n_features=1, 
        noise=5,    
        random_state=42 
    )

    # kernel_params_list = [
    #     ("linear", {}),
    #     ("rbf", {"gamma": 0.1}),
    #     ("poly", {"degree": 2, "coef0": 1}), 
    # ]
    
    kernel_params_list = [
        ("linear", {}),
        ("rbf", {"gamma": 1e-2}),
        ("rbf", {"gamma": 1e-1}),
        ("rbf", {"gamma": 1e0}),
        ("rbf", {"gamma": 1e2}),
        ("poly", {"degree": 2}),
        ("poly", {"degree": 3}),
        ("sigmoid", {}),
        (ExpSineSquared(periodicity=132), {}),
        (ExpSineSquared(periodicity=132, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=132, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=11), {}),
        (ExpSineSquared(periodicity=11, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=11, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=6), {}),
        (ExpSineSquared(periodicity=6, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=6, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=3), {}),
        (ExpSineSquared(periodicity=3, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=3, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=1), {}),
        (ExpSineSquared(periodicity=1, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=1, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=0.25), {}),
        (ExpSineSquared(periodicity=0.25, length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=0.25, length_scale=1e1), {}),
        (ExpSineSquared(periodicity=0.33), {}),
        (ExpSineSquared(periodicity=0.33 , length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=0.33 , length_scale=1e1), {}),
        (ExpSineSquared(periodicity=0.5), {}),
        (ExpSineSquared(periodicity=0.5 , length_scale=1e-1), {}),
        (ExpSineSquared(periodicity=0.5 , length_scale=1e1), {}),
        (RationalQuadratic(), {}),
        (RationalQuadratic(length_scale=1e-1), {}),
        (RationalQuadratic(length_scale=1e1), {}),
        (Matern(nu=0.5), {}),
        (Matern(nu=0.5, length_scale=1e-1), {}),
        (Matern(nu=0.5, length_scale=1e1), {}),
        (Matern(nu=1.5), {}),
        (Matern(nu=1.5, length_scale=1e-1), {}),
        (Matern(nu=1.5, length_scale=1e1), {}),
        (Matern(nu=2.5), {}),
        (Matern(nu=2.5, length_scale=1e-1), {}),
        (Matern(nu=2.5, length_scale=1e1), {}),
        (ConstantKernel(), {})
    ]

    C_val = 1e3
    epsilon_val = 0.5
    tau_val = 1.0 # Test with tau=1.0 where sqrt(tau)=1
    # tau_val = 4.0 # Test with tau=4.0 where sqrt(tau)=2, scaling factor would be 4
    tau_val = 1e-3 # Test with tau=0.25 where sqrt(tau)=0.5, scaling factor would be 1


    print("--- Testing SvrQcqpMultiKernelL1Mu (Original Formulation) ---")
    model_qcqp_multi = SvrQcqpMultiKernelL1Trace(
        C=C_val, epsilon=epsilon_val, tau=tau_val,
        kernel_params=kernel_params_list, verbose=False, kronecker_kernel=True
    )
    model_qcqp_multi.fit(X, y_reg)
    print(f"QCQP Status: {model_qcqp_multi.status_}")
    if model_qcqp_multi.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        y_pred_multi_qcqp = model_qcqp_multi.predict(X)
        # print(f"QCQP Objective: {model_qcqp_multi.objective_value_}")
        print(f"QCQP b: {model_qcqp_multi.b_:.6f}")
        print(f"QCQP Support Vectors: {model_qcqp_multi.sup_num_}")
        print(f"QCQP Mu: {np.round(model_qcqp_multi.mu_, 6)}")
        print(f"QCQP MAE: {mean_absolute_error(y_reg, y_pred_multi_qcqp):.6f}")
    else:
        print(f"QCQP Model optimization failed.")


    print("\n--- Testing SvrSocpMultiKernelL1MuExplicit (Explicit SOCP Formulation) ---")
    model_socp_explicit = SvrQcqpMultiKernelL1EpigraphTrace(
        C=C_val, epsilon=epsilon_val, tau=tau_val,
        kernel_params=kernel_params_list, verbose=False, kronecker_kernel=True
    )
    model_socp_explicit.fit(X, y_reg)
    print(f"SOCP Status: {model_socp_explicit.status_}")
    if model_socp_explicit.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        y_pred_multi_socp = model_socp_explicit.predict(X)
        # print(f"SOCP Objective: {model_socp_explicit.objective_value_}")
        print(f"SOCP b: {model_socp_explicit.b_:.6f}")
        print(f"SOCP Support Vectors: {model_socp_explicit.sup_num_}")
        print(f"SOCP Mu (scaled): {np.round(model_socp_explicit.mu_, 6)}")
        print(f"SOCP MAE: {mean_absolute_error(y_reg, y_pred_multi_socp):.6f}")

        # Compare betas
        if model_qcqp_multi.beta_ is not None and model_socp_explicit.beta_ is not None:
            beta_diff = np.linalg.norm(model_qcqp_multi.beta_ - model_socp_explicit.beta_)
            print(f"Norm of difference in beta values: {beta_diff:.6e}")
    else:
        print(f"SOCP Model optimization failed.")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y_reg, color='black', label='Data', s=20, alpha=0.7)
    if model_qcqp_multi.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        plt.plot(np.sort(X.flatten()), model_qcqp_multi.predict(np.sort(X, axis=0)), label=f'QCQP (b={model_qcqp_multi.b_:.2f})', linestyle='--')
    if model_socp_explicit.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        plt.plot(np.sort(X.flatten()), model_socp_explicit.predict(np.sort(X, axis=0)), label=f'SOCP Explicit (b={model_socp_explicit.b_:.2f})', linestyle=':')
    
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.title(f"SVR Model Predictions (tau={tau_val})")
    plt.show()


# %%

