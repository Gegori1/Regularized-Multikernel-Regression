# %%
import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels


class BaseSVRL2:
    def __init__(
        
            self, C: float=1,
            epsilon: float=1,
            tau: float=1,
            kernel_params: list=[("linear", {})],
            verbose: bool=False,
            compute_C: bool=False,
        ):
        
        self.C = C
        self.epsilon = epsilon
        self.tau = tau
        self.kernel_params = kernel_params
        self.verbose = verbose
        self.compute_C = compute_C
        
        
    def check_x_y(self, X, y):
        X, y = check_X_y(X, y)
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        return X, y

    
    def param(self, X):
        m = X.shape[0]
        
        onev = np.ones((m, 1))
        I = np.identity(m)

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
        if self.compute_C:
            kernels.append(np.eye(m))
            trace += m
            
        self.kernels = kernels
        self.trace = trace
        
        return onev, m, I
    
    def solve(self, X, y):
        pass
        
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
            
        return indx
    
    def compute_b(self):
        m = self.support_vectors.shape[0]
        # calculate epsilon
        epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon, self.epsilon)
        # compute kernel
        support_kernel = 0
        for mu, kern, k in zip(self.mu, self.kernel_params, self.kernels):
            if isinstance(kern[0], str):
                support_kernel += mu*pairwise_kernels(self.support_vectors, metric=kern[0], filter_params=True, **kern[1])
            else:
                support_kernel += mu*kern[0](self.support_vectors, **kern[1])
        # add identity matrix
        if self.compute_C:
            support_kernel += self.mu[-1] * np.eye(m)
        
        self.b = np.mean(
            epsilon_beta - self.support_beta.T @ support_kernel + self.support_labels - self.support_beta / self.C
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
        if any(self.mu >= 1e-5):
            self.mu = np.where(self.mu <= 1e-5, 0, self.mu)
        else:
            self.mu = np.where(self.mu <= 1e-7, 0, self.mu)
        self.C = 1 / self.mu[-1] if self.compute_C else self.C
        
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
                K += mu*pairwise_kernels(self.support_vectors, X, metric=kern[0], filter_params=True, **kern[1])
            else:
                K += mu*kern[0](self.support_vectors, X, **kern[1])
                
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
    
class SvrQcqpMultiKernelL2Mu(BaseSVRL2):
    def solve(self, X, y):
            
            onev, m, I = self.param(X)
            
            # variables
            self.t = cp.Variable(1, name="t")
            self.beta = cp.Variable((m, 1), name="beta")
            
            self.constraints_kernels = [
                self.tau >= cp.quad_form(self.beta, cp.psd_wrap(K)) for K in self.kernels
            ]
            
            # problem
            if self.compute_C:
                self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) + 2 * self.beta.T @ y)
                
                self.constraints = [
                    *self.constraints_kernels,
                    self.tau >= cp.quad_form(self.beta, I),
                    self.beta.T @ onev == 0
                ]
                
            else:
                self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) - cp.quad_form(self.beta, I) / self.C + 2 * self.beta.T @ y)
            
                self.constraints = [
                    *self.constraints_kernels,
                    self.beta.T @ onev == 0
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
        
# %%

class SvrQcqpMultiKernelL2Trace(BaseSVRL2):
    def solve(self, X, y):
            
            onev, m, I = self.param(X)
            
            # variables
            self.t = cp.Variable(1, name="t")
            self.beta = cp.Variable((m, 1), name="beta")
            
            self.constraints_kernels = [
                np.trace(K) * self.tau >= cp.quad_form(self.beta, cp.psd_wrap(K)) for K in self.kernels
            ]
            
            # problem
            if self.compute_C:
                self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) + 2 * self.beta.T @ y)
                
                self.constraints = [
                    *self.constraints_kernels,
                    m * self.tau >= cp.quad_form(self.beta, I),
                    self.beta.T @ onev == 0
                ]
                
            else:
                self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) - cp.quad_form(self.beta, I) / self.C + 2 * self.beta.T @ y)
            
                self.constraints = [
                    *self.constraints_kernels,
                    self.beta.T @ onev == 0
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
 
# %%
      
class SvrQcqpMultiKernelL2EpigraphTrace(BaseSVRL2):
    def solve(self, X, y):
            
            onev, m, I = self.param(X)
            
            # variables
            self.t = cp.Variable(1, name="t")
            self.beta = cp.Variable((m, 1), name="beta")
            
            self.constraints_kernels = [
                self.t >= cp.quad_form(self.beta, cp.psd_wrap(K)) / np.trace(K) for K in self.kernels
            ]
            
            # problem
            if self.compute_C:
                self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) + 2 * self.beta.T @ y - self.tau * self.t)
                
                self.constraints = [
                    *self.constraints_kernels,
                    self.t >= cp.quad_form(self.beta, I) / m,
                    self.beta.T @ onev == 0
                ]
                
            else:
                self.objective = cp.Maximize(-2*self.epsilon * cp.norm1(self.beta) - cp.quad_form(self.beta, I) / self.C + 2 * self.beta.T @ y - self.tau * self.t)
            
                self.constraints = [
                    *self.constraints_kernels,
                    self.beta.T @ onev == 0
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
        if self.compute_C:
            support_kernel += self.mu[-1] * np.eye(m)
        
        self.b = np.mean(
            epsilon_beta - self.support_beta.T @ support_kernel + self.support_labels - self.support_beta / self.C
        )
        
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
                
        return self.support_beta.T @ K + self.b

# %%
if __name__ == "__main__":
    
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    import matplotlib.pyplot as plt
    # from svr_sdp_multi_kernel_l2 import svr_sdp_multi_kernel_l2
    
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ExpSineSquared
    
    # Create a regression dataset
    X, y = make_regression(
        n_samples=100,
        n_features=1, 
        noise=20,
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
        (ExpSineSquared(periodicity=1), {}),
        (ExpSineSquared(periodicity=3), {}),
        (RBF(length_scale=0.1), {}),
        (RBF(length_scale=10), {}),
        (RBF(length_scale=50), {}),
    ]

    C = 100
    epsilon = 1e-2
    gamma = 1e2
    tau = 1e1
# %%

    # Create an instance of svr_sdp_multi_kernel
    model_qcqp_multi = SvrQcqpMultiKernelL2EpigraphTrace(
        C = C,
        epsilon = epsilon,
        tau = tau,
        kernel_params = kernel_params,
        compute_C=False,
        verbose=False,
    )
    
    model_sdp_multi = SvrQcqpMultiKernelL2Trace(
        C = C,
        epsilon = epsilon,
        tau = tau,
        kernel_params = kernel_params,
        verbose=False,
        # kronecker_kernel=False
    )

    model_sklrn = SVR(
        C=C, 
        epsilon=epsilon,
        kernel="rbf",
        gamma=gamma
    )
    
    model_qcqp_multi.fit(X, y)
    model_sdp_multi.fit(X, y)
    model_sklrn.fit(X, y)
    
    y_pred_multi_qcqp = model_qcqp_multi.predict(X)
    y_pred_multi_sdp = model_sdp_multi.predict(X)
    y_pred_sklrn = model_sklrn.predict(X)

    model_qcqp_multi.problem.solver_stats.solve_time
# %%





# %%
    plt.title("Support vectors")
    plt.scatter(X, y)
    plt.scatter(model_qcqp_multi.support_vectors, model_qcqp_multi.support_labels, color="red", marker="o", label="QCQP Support Vectors")
    plt.scatter(model_sdp_multi.support_vectors, model_sdp_multi.support_labels, color="orange", marker="+", label="SDP Support Vectors")
    # plt.scatter(X[model_sklrn.support_], y[model_sklrn.support_], color="orange", marker=".", label="Sklearn Support Vectors")
    plt.legend()
    plt.show()
    
    # support_indices_sklearn = model_sklrn.support_
    # dual_coef_full_sklearn = np.zeros_like(y)
    # dual_coef_full_sklearn[support_indices_sklearn] = model_sklrn.dual_coef_

    d = np.linspace(-2, 2, y.shape[0])
    # plt.scatter(d, dual_coef_full_sklearn, label="sklearn", marker="*", color="tab:red")
    plt.scatter(d, model_sdp_multi.beta.flatten(), label="sdp_multi_l1", marker="+", color="tab:blue")
    plt.scatter(d, model_qcqp_multi.beta.flatten(), label="qcqp_multi_l1", marker=".", color="orange")
    # plt.ylim(-0.0001, 0.0001)
    plt.legend()
    plt.show()
    
    plt.scatter(X, y)
    # plt.scatter(X, y_pred_sklrn, label="sklearn")
    plt.scatter(X, y_pred_multi_qcqp, marker="+", label="qcqp_multi")
    plt.scatter(X, y_pred_multi_sdp, marker=".", label="sdp_multi")
    plt.legend()
    plt.title("Predictions")
    plt.show()
    
    
    print("b from model_sklearn:", model_sklrn.intercept_[0])
    print("b from model_qcqp_multi", model_qcqp_multi.b)
    print("b from model_sdp_multi", model_sdp_multi.b)
# %%
    
    def print_accuracy(y, y_pred, model):
        print(f"{model} accuracy: {mean_absolute_error(y, y_pred.flatten())}")

    # print_accuracy(y, y_pred_sklrn, "sklearn")
    print_accuracy(y, y_pred_multi_qcqp, "qcqp_multi_l1")
    print_accuracy(y, y_pred_multi_sdp, "sdp_multi_l1")
# %%







# %%

    import pandas as pd
    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.model_selection import train_test_split
    
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.gaussian_process.kernels import ExpSineSquared
    from sklearn.gaussian_process.kernels import RationalQuadratic
    from sklearn.gaussian_process.kernels import WhiteKernel
    
    # from svr_sdp_multi_kernel_l1 import svr_sdp_multi_kernel_l1
    
    co2 = fetch_openml(data_id=41187, as_frame=True)
    co2.frame.head()
    
    co2_data = (
        co2
        .frame
        [["year", "month", "day", "co2"]]
        .assign(
            date = lambda k: pd.to_datetime(k[['year', 'month', 'day']])
        )
        [["date", "co2"]]
    )
    
    co2_data = (
        co2_data.sort_values(by="date")
        .groupby(pd.Grouper(key="date", freq="ME"))
        .agg({"co2": "mean"})
        .dropna()
        .reset_index()
    )
    
    X_ = (co2_data["date"].dt.year + co2_data["date"].dt.month / 12).to_numpy().reshape(-1, 1)
    y_ = co2_data["co2"].to_numpy()
    
    X, y = X_[67:], y_[67:]
    
    long_term_trend_kernel = RBF(length_scale=50.0)
    seasonal_kernel = (
        RBF(length_scale=100.0)
        * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    )
    irregularities_kernel = RationalQuadratic(length_scale=1.0, alpha=1.0)
    noise_kernel = RBF(length_scale=0.1) + WhiteKernel(
        noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
    )
    co2_kernel = (
        long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
    )
    co2_kernel
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=145, shuffle=False)
    
    # long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
    # seasonal_kernel = (
    #     2.0**2
    #     * RBF(length_scale=100.0)
    #     * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    # )
    # irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
    # noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
    #     noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
    # )
    # co2_kernel = (
    #     long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
    # )
    # co2_kernel
    
    # kernel_params = [
    #     (irregularities_kernel, {}),
    #     (seasonal_kernel, {}),
    #     (long_term_trend_kernel, {}),
    #     (noise_kernel, {}),
    # ]
    
    kernel_params = [
        # (co2_kernel, {}),
        ("linear", {}),
        ("rbf", {"gamma":1e-2}),
        ("rbf", {"gamma":1e-1}),
        ("rbf", {"gamma":10}),
        (ExpSineSquared(periodicity=1), {}),
        (ExpSineSquared(periodicity=3), {}),
        (irregularities_kernel, {}),
        (seasonal_kernel, {}),
        (long_term_trend_kernel, {}),
        (noise_kernel, {}),
        (RBF(length_scale=0.1), {}),
    ]
    



    C = 1e2
    epsilon = 1e0
    # model_sklrn = SVR(
    #     C=C,
    #     epsilon=epsilon,
    #     kernel=co2_kernel
    # )

    # model_sklrn.fit(X_train, y_train)
    
    # Create an instance of svr_sdp_multi_kernel
    model_qcqp_multi = SvrQcqpMultiKernelL2Mu(
        # C = C,
        epsilon = epsilon,
        tau = 1e2,
        kernel_params = kernel_params,
        verbose = False,
        compute_C=True
    )
    
    model_qcqp_multi.fit(X_train, y_train)
    model_qcqp_multi.status
# %%

    model_sdp_multi = SvrQcqpMultiKernelL2Trace(
        C = C,
        epsilon = epsilon,
        tau = 1e1,
        kernel_params = kernel_params,
        verbose = False,
        compute_C=False
    )
    
    model_sdp_multi.fit(X_train, y_train)
    model_sdp_multi.status
    
# %%

    y_pred_qcqp_multi = model_qcqp_multi.predict(X_test)
    y_pred_sdp_multi = model_sdp_multi.predict(X_test)
    # y_pred_mape = model_sklrn.predict(X_test)
    y_pred_qcqp_multi_train = model_qcqp_multi.predict(X_train)
    
    support_indices_sklearn = model_sklrn.support_
    dual_coef_full_sklearn = np.zeros_like(y_train)
    dual_coef_full_sklearn[support_indices_sklearn] = model_sklrn.dual_coef_

    d = np.linspace(-2, 2, y_train.shape[0])
    # plt.scatter(d, dual_coef_full_sklearn, label="sklearn", marker="*")
    plt.scatter(d, model_qcqp_multi.beta.flatten(), label="qcqp_multi_l1", marker="+", color="tab:blue")
    plt.scatter(d, model_sdp_multi.beta.flatten(), label="sdp_multi_l1", marker=".", color="orange")
    # plt.ylim(-0.0001, 0.0001)
    plt.legend()
    plt.show()
    
    plt.title("Support vectors")
    plt.scatter(X_train, y_train)
    plt.scatter(model_qcqp_multi.support_vectors, model_qcqp_multi.support_labels, color="red", marker="o", label="QCQP Support Vectors")
    plt.scatter(model_sdp_multi.support_vectors, model_sdp_multi.support_labels, color="orange", marker="+", label="SDP Support Vectors")
    # plt.scatter(X[model_sklrn.support_], y[model_sklrn.support_], color="orange", marker=".", label="Sklearn Support Vectors")
    plt.legend()
    plt.show()

    plt.plot(X, y, color="black", linestyle="dashed", label="Measurements")
    plt.plot(X_test, y_pred_qcqp_multi.flatten(), color="tab:blue", alpha=0.4, label="SVR L1 Multikernel QCQP")
    # plt.plot(X_test, y_pred_sdp_multi.flatten(), color="tab:green", alpha=0.4, label="SVR L1 Multikernel SDP")
    # plt.plot(X_train, y_pred_qcqp_multi_train.flatten(), color="yellow", alpha=0.6, label="SVR L1 Multikernel train")
    # plt.plot(X_test, y_pred_mape.flatten(), color="tab:red", alpha=0.4, label="SVR L1")
    plt.legend()
    


    def print_accuracy(y, y_pred, model):
        print(f"{model} accuracy: {mean_absolute_percentage_error(y, y_pred.flatten())}")

    # print_accuracy(y_test, y_pred_mape, "original_l1")
    print_accuracy(y_test, y_pred_qcqp_multi, "qcqp_multi_l1")
    print_accuracy(y_test, y_pred_sdp_multi, "sdp_multi_l1")
    print_accuracy(y_train, y_pred_qcqp_multi_train, "qcqp_multi_l1_train")
# %%
