# %%
import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y
from sklearn.metrics.pairwise import pairwise_kernels
import warnings
import scipy.linalg

class BaseSVR:

    def __init__(
            self, C: float=1,
            epsilon: float=1,
            tau: float=1,
            kernel_params: list=[("linear", {})],
            c: float=1, # This 'c' seems distinct from 'C', its role might need clarification or could be related to trace_min
            verbose: bool=False,
            trace_min: bool=True,
            trace_min_factor: int=1,
            kronecker_kernel: bool=False,
        ):
        
        self.C = C
        self.epsilon = epsilon
        self.tau = tau
        self.kernel_params = kernel_params
        self.c_param = c # Renamed to avoid confusion with uppercase C regularization param
        self.verbose = verbose
        self.trace_min = trace_min
        self.trace_min_factor = trace_min_factor
        self.kronecker_kernel = kronecker_kernel
        
        # Attributes to be set during fit
        self.kernels_ = [] # Store computed kernel matrices
        self.kernel_R_factors_ = [] # Store R factors for K = R.T @ R
        self.trace_ = 0.0
        self.beta_ = None # Optimized beta coefficients
        self.mu_ = None # Dual variables for kernel constraints
        self.b_ = 0.0 # Bias term
        self.support_vectors_ = np.array([])
        self.support_labels_ = np.array([])
        self.support_beta_ = np.array([])
        self.sup_num_ = 0
        self.status_ = "not_fitted"
        self.objective_value_ = None
        self.problem_ = None


    def check_x_y(self, X, y):
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True) # Ensure y is numeric for SVR
        
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        return X, y

    def delta_kronecker_kernel(self, X, Y=None): # Changed Y: None to Y=None for convention
        """
        Computes the Kronecker delta kernel.
        Returns an identity matrix if X and Y are the same or Y is None.
        Otherwise, compares rows for exact equality.
        """
        if Y is None or X is Y or np.array_equal(X,Y): # More robust check
            return np.eye(X.shape[0])
        
        X_arr = np.asarray(X) # Ensure X is an array
        Y_arr = np.asarray(Y) # Ensure Y is an array

        if X_arr.shape[1] != Y_arr.shape[1]:
            raise ValueError("X and Y must have the same number of features for delta_kronecker_kernel.")

        # Original broadcasting method is generally efficient for moderate dimensions
        X_reshaped = X_arr[:, np.newaxis, :]
        Y_reshaped = Y_arr[np.newaxis, :, :]
        comparison_matrix = (X_reshaped == Y_reshaped)
        kernel_matrix = np.all(comparison_matrix, axis=2).astype(int)
        return kernel_matrix
    
    def _get_kernel_R_factor(self, K):
        """
        Computes R such that K = R.T @ R.
        Uses Cholesky decomposition if K is positive definite,
        otherwise uses eigenvalue decomposition for PSD K.
        """
        try:
            # Try Cholesky for positive definite matrices
            # Add a small epsilon for numerical stability if K is only PSD
            # K_stable = K + np.eye(K.shape[0]) * 1e-9 
            L = scipy.linalg.cholesky(K, lower=True) # K_stable instead of K if regularizing
            return L.T # R = L.T
        except scipy.linalg.LinAlgError:
            # If Cholesky fails (not positive definite), use eigenvalue decomposition
            eigenvalues, eigenvectors = scipy.linalg.eigh(K)
            
            # Clamp small negative eigenvalues to zero (due to numerical precision)
            eigenvalues[eigenvalues < 1e-10] = 0 
            
            D_sqrt = np.diag(np.sqrt(eigenvalues))
            # R such that K = R.T @ R. If K = V D V.T, then R = D_sqrt @ V.T
            R = D_sqrt @ eigenvectors.T 
            return R

    def param(self, X):
        """
        Prepares kernel matrices and their R factors.
        """
        m = X.shape[0]
        onev = np.ones((m, 1))
        
        self.kernels_ = []
        self.kernel_R_factors_ = []
        current_trace = 0 # Local variable for trace sum
        for i, kern_spec in enumerate(self.kernel_params):
            kern_name, kern_args = kern_spec
            if isinstance(kern_name, str):
                kernel = pairwise_kernels(X, metric=kern_name, filter_params=True, **kern_args)
            else: # Callable kernel
                kernel = kern_name(X, **kern_args)
            
            self.kernels_.append(kernel)
            current_trace += np.trace(kernel)
            
            R_factor = self._get_kernel_R_factor(kernel)
            self.kernel_R_factors_.append(R_factor)

        if self.kronecker_kernel:
            kernel_eye = np.eye(m)
            self.kernels_.append(kernel_eye)
            current_trace += m 
            self.kernel_R_factors_.append(np.eye(m)) 
            
        self.trace_ = current_trace
        
        if self.trace_min:
            self.c_param = self.trace_ * self.trace_min_factor
            
        return onev, m
    
    def solve(self, X, y):
        pass

    def _handle_non_optimal(self, y_mean):
        self.sup_num_ = 0
        self.support_vectors_ = np.array([])
        self.support_labels_ = np.array([])
        self.support_beta_ = np.array([])
        self.b_ = y_mean 
        num_kernels = len(self.kernel_params) + (1 if self.kronecker_kernel else 0)
        self.mu_ = np.zeros(num_kernels)
        if hasattr(self, 'X_fit_') and self.X_fit_ is not None:
             self.beta_ = np.zeros((self.X_fit_.shape[0], 1))
        else:
             self.beta_ = np.array([])


    def _compute_support_vectors_and_b(self, X, y):
        if self.beta_ is None or self.beta_.size == 0:
            self._handle_non_optimal(np.mean(y))
            return

        significant_mu_threshold = 1e-7 
        if self.mu_ is not None:
            if np.any(np.abs(self.mu_) >= 1e-4): 
                significant_mu_threshold = 1e-4
            self.mu_ = np.where(np.abs(self.mu_) <= significant_mu_threshold, 0, self.mu_)
        else: 
            num_kernels = len(self.kernels_) # kernels_ now includes kronecker if true
            self.mu_ = np.zeros(num_kernels)

        beta_flat = self.beta_.flatten()
        support_indices = np.abs(beta_flat) > 1e-5 

        if np.sum(support_indices) == 0:
            self._handle_non_optimal(np.mean(y))
            return

        self.sup_num_ = np.sum(support_indices)
        self.support_vectors_ = X[support_indices, :]
        self.support_labels_ = y[support_indices] 
        self.support_beta_ = self.beta_[support_indices]
        
        K_sv_sum = np.zeros((self.sup_num_, self.sup_num_))
        
        # Ensure mu_ has the same length as kernels_
        if len(self.mu_) != len(self.kernels_):
            print(f"Warning: Mismatch in length of mu_ ({len(self.mu_)}) and kernels_ ({len(self.kernels_)}). Adjusting mu_.")
            # This might happen if mu_ wasn't correctly sized after optimization failure or in SOCP dual retrieval
            temp_mu = np.zeros(len(self.kernels_))
            # Attempt to copy over existing values if possible (e.g. if mu_ was shorter)
            copy_len = min(len(self.mu_), len(temp_mu))
            temp_mu[:copy_len] = self.mu_[:copy_len]
            self.mu_ = temp_mu

        for i, K_matrix_full_data in enumerate(self.kernels_): 
            if i < len(self.mu_) and self.mu_[i] != 0: # Check index bounds for mu_
                K_sv_component = K_matrix_full_data[support_indices][:, support_indices]
                K_sv_sum += self.mu_[i] * K_sv_component
        
        epsilon_tolerance = 1e-5 
        unbounded_sv_indices_in_support = (np.abs(self.support_beta_.flatten()) < (self.C - epsilon_tolerance))
        
        if np.any(unbounded_sv_indices_in_support):
            b_sum = 0
            count = 0
            preds_on_sv = K_sv_sum @ self.support_beta_ 
            
            for i in range(self.sup_num_):
                if unbounded_sv_indices_in_support[i]:
                    pred_i = preds_on_sv[i]
                    if self.support_beta_[i] > epsilon_tolerance: 
                        b_sum += self.support_labels_[i] - pred_i - self.epsilon
                    elif self.support_beta_[i] < -epsilon_tolerance: 
                        b_sum += self.support_labels_[i] - pred_i + self.epsilon
                    else: 
                        continue 
                    count += 1
            self.b_ = (b_sum / count)[0] if count > 0 and isinstance(b_sum, np.ndarray) else (b_sum / count if count > 0 else np.mean(y))
        else: 
            self.b_ = np.mean(y) 
        if isinstance(self.b_, np.ndarray): 
            self.b_ = self.b_.item()


    def fit(self, X, y):
        self.X_fit_ = X 
        self.y_fit_ = y 
        X_checked, y_checked = self.check_x_y(X, y) # Use different names to avoid overwriting original y
        
        try:
            self.solve(X_checked, y_checked) 
        except cp.SolverError as e:
            self.status_ = "solver_error"
            print(f"CVXPY Solver failed: {e}")
            if self.problem_ and hasattr(self.problem_, 'solver_stats') and self.problem_.solver_stats: print(f"Solver stats: {self.problem_.solver_stats}")
            self._handle_non_optimal(np.mean(self.y_fit_)) # Use original y_fit_ for mean
            return self
        except Exception as e:
            self.status_ = "unexpected_error"
            print(f"An unexpected error occurred during fitting: {e}")
            import traceback
            traceback.print_exc()
            self._handle_non_optimal(np.mean(self.y_fit_))
            return self
        
        # Use a more robust check for problem status, as string "optimal" might come from older versions or direct assignment
        successful_statuses = [cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal", "optimal_inaccurate"]
        if self.status_ not in successful_statuses:
            print(f"Optimization not successful. Status: {self.status_}. Using default prediction.")
            self._handle_non_optimal(np.mean(self.y_fit_))
            return self
        
        self._compute_support_vectors_and_b(X_checked, y_checked) # Use checked X, y     
        return self
    
    def predict(self, X_pred): 
        return self.decision_function(X_pred)
    
    def decision_function(self, X_pred):
        from sklearn.utils.validation import check_array 
        X_pred_checked = check_array(X_pred) 

        if self.sup_num_ == 0 or self.support_vectors_ is None or self.support_vectors_.shape[0] == 0:
            return np.full(X_pred_checked.shape[0], self.b_)

        K_pred_sum = np.zeros((self.sup_num_, X_pred_checked.shape[0]))
        
        # Use self.kernels_ which is populated correctly in self.param()
        # and self.mu_ which should correspond to self.kernels_
        
        if self.mu_ is None or len(self.mu_) != len(self.kernels_):
            print(f"Warning: Mismatch or uninitialized mu_ in decision_function. Mu length: {len(self.mu_ if self.mu_ is not None else [])}, Kernels length: {len(self.kernels_)}. Predictions may be inaccurate.")
            current_mu = np.zeros(len(self.kernels_)) # Fallback
        else:
            current_mu = self.mu_

        for i, K_matrix_full_data in enumerate(self.kernels_): # Iterate using self.kernels_
            if i < len(current_mu) and current_mu[i] != 0:
                # We need K(support_vectors, X_pred_checked)
                # K_matrix_full_data is K(X_fit, X_fit)
                # Instead, recalculate kernel between SVs and X_pred_checked
                kern_name, kern_args = "placeholder", {} # Need to get original spec
                original_kern_idx = i
                if self.kronecker_kernel and i == len(self.kernels_) -1 : # Last kernel is kronecker
                     kern_name = "kronecker_special" # Special handling
                elif i < len(self.kernel_params):
                     kern_name, kern_args = self.kernel_params[i]


                if kern_name == "kronecker_special":
                    K_component = self.delta_kronecker_kernel(self.support_vectors_, X_pred_checked)
                elif isinstance(kern_name, str):
                    K_component = pairwise_kernels(self.support_vectors_, X_pred_checked, metric=kern_name, filter_params=True, **kern_args)
                elif callable(kern_name): # Callable kernel
                    K_component = kern_name(self.support_vectors_, X_pred_checked, **kern_args)
                else: # Placeholder or issue
                    K_component = np.zeros((self.sup_num_, X_pred_checked.shape[0]))
                    if kern_name != "placeholder":
                         print(f"Warning: Unknown kernel type '{kern_name}' in decision_function.")

                K_pred_sum += current_mu[i] * K_component
        
        prediction = self.support_beta_.T @ K_pred_sum + self.b_
        return prediction.flatten()

    def score(self, X, y, sample_weight=None): 
        from sklearn.metrics import r2_score 
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
    def get_params(self, deep=True): 
        return {
            "C": self.C, 
            "epsilon": self.epsilon,
            "tau": self.tau,
            "kernel_params": self.kernel_params,
            "c": self.c_param, 
            "verbose": self.verbose,
            "trace_min": self.trace_min,
            "trace_min_factor": self.trace_min_factor,
            "kronecker_kernel": self.kronecker_kernel
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            if key == 'c': 
                setattr(self, 'c_param', value)
            else:
                setattr(self, key, value)
        if self.trace_min and hasattr(self, 'trace_') and self.trace_ is not None:
             if 'trace_min_factor' in params or 'trace_min' in params:
                 self.c_param = self.trace_ * self.trace_min_factor
        elif not self.trace_min and 'c' in params: 
            self.c_param = params['c']
        return self
    
# %%  QCQP (Original formulation, likely solved as SOCP by CVXPY+MOSEK)
class SvrQcqpMultiKernelL1Mu(BaseSVR):

    def solve(self, X, y):
            onev, m = self.param(X) 
            
            beta_cvx = cp.Variable((m, 1), name="beta_cvx")
            objective = cp.Maximize(-2*self.epsilon * cp.norm(beta_cvx, 1) + 2 * beta_cvx.T @ y)
            
            kernel_constraints_cvx = []
            for K_matrix in self.kernels_: 
                constraint = cp.quad_form(beta_cvx, cp.psd_wrap(K_matrix)) <= self.tau
                kernel_constraints_cvx.append(constraint)
            
            constraints = [
                *kernel_constraints_cvx,
                beta_cvx.T @ onev == 0,
                cp.abs(beta_cvx) <= self.C 
            ]
            
            self.problem_ = cp.Problem(objective, constraints)
            self.problem_.solve(solver=cp.MOSEK, verbose=self.verbose)
            
            self.status_ = self.problem_.status
            if self.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                self.objective_value_ = self.problem_.value
                self.beta_ = beta_cvx.value.reshape(-1, 1) if beta_cvx.value is not None else np.zeros((m,1))
                try:
                    self.mu_ = np.array([const.dual_value[0] if const.dual_value is not None else 0.0 
                                        for const in kernel_constraints_cvx]).flatten()
                except Exception as e:
                    print(f"Could not retrieve dual values for QCQP: {e}")
                    self.mu_ = np.zeros(len(kernel_constraints_cvx)) # Match length of constraints
            else:
                self.beta_ = np.zeros((m,1)) 
                self.mu_ = np.zeros(len(self.kernels_))
                
            return self
        
# %%  QCQP (Original formulation, likely solved as SOCP by CVXPY+MOSEK)
class SvrQcqpMultiKernelL1Trace(BaseSVR):

    def solve(self, X, y):
            onev, m = self.param(X) 
            
            beta_cvx = cp.Variable((m, 1), name="beta_cvx")
            objective = cp.Maximize(-2*self.epsilon * cp.norm(beta_cvx, 1) + 2 * beta_cvx.T @ y)
            
            kernel_constraints_cvx = []
            for K_matrix in self.kernels_: 
                constraint = cp.quad_form(beta_cvx, cp.psd_wrap(K_matrix)) <= self.tau * np.trace(K_matrix)
                kernel_constraints_cvx.append(constraint)
            
            constraints = [
                *kernel_constraints_cvx,
                beta_cvx.T @ onev == 0,
                cp.abs(beta_cvx) <= self.C 
            ]
            
            self.problem_ = cp.Problem(objective, constraints)
            self.problem_.solve(solver=cp.MOSEK, verbose=self.verbose)
            
            self.status_ = self.problem_.status
            if self.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                self.objective_value_ = self.problem_.value
                self.beta_ = beta_cvx.value.reshape(-1, 1) if beta_cvx.value is not None else np.zeros((m,1))
                try:
                    self.mu_ = np.array([const.dual_value[0] if const.dual_value is not None else 0.0 
                                        for const in kernel_constraints_cvx]).flatten()
                except Exception as e:
                    print(f"Could not retrieve dual values for QCQP: {e}")
                    self.mu_ = np.zeros(len(kernel_constraints_cvx)) # Match length of constraints
            else:
                self.beta_ = np.zeros((m,1)) 
                self.mu_ = np.zeros(len(self.kernels_))
                
            return self

# %% SOCP (Explicit formulation)

class SvrSocpMultiKernelL1MuExplicit(BaseSVR):

    def solve(self, X, y):
        onev, m = self.param(X) 
        
        if self.tau < 0:
            print("Warning: self.tau is negative. SOCP explicit formulation expects tau >= 0.")
            self.status_ = "error_negative_tau"
            self._handle_non_optimal(np.mean(self.y_fit_ if hasattr(self, 'y_fit_') else y)) # Use y_fit_ if available
            return self

        beta_cvx = cp.Variable((m, 1), name="beta_socp")
        objective_term_l1 = -2 * self.epsilon * cp.norm(beta_cvx, 1)
        objective = cp.Maximize(objective_term_l1 + 2 * beta_cvx.T @ y)
        
        socp_kernel_constraints = []
        for R_factor, kernel  in zip(self.kernel_R_factors_, self.kernels_):
            if self.tau < 1e-9: # Effectively zero tau
                constraint = (R_factor @ beta_cvx == 0)
            else:
                norm_arg = R_factor @ beta_cvx
                constraint = cp.norm(norm_arg, 2) <= np.sqrt(self.tau)
            socp_kernel_constraints.append(constraint)
            
        constraints = [
            *socp_kernel_constraints,
            beta_cvx.T @ onev == 0,
            cp.abs(beta_cvx) <= self.C 
        ]
        
        mosek_params = {
            # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1.0e-9,

            # Dual feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1.0e-10,
            
            # Relative optimality gap
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1.0e-9
        }
        
        mosek_params = None
        
        self.problem_ = cp.Problem(objective, constraints)
        self.problem_.solve(solver=cp.MOSEK, verbose=self.verbose)
        
        self.status_ = self.problem_.status
        if self.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.objective_value_ = self.problem_.value
            self.beta_ = beta_cvx.value.reshape(-1, 1) if beta_cvx.value is not None else np.zeros((m,1))
            try:
                raw_dual_values = []
                for const in socp_kernel_constraints:
                    # Attempt to retrieve the scalar dual value.
                    # For cp.norm(X,2) <= t, the dual is typically a non-negative scalar.
                    # If const.dual_value is a list/array (e.g. from some conic dual structures),
                    # we assume the relevant scalar is the first element or the value itself if scalar.
                    val = 0.0
                    if const.dual_value is not None:
                        if isinstance(const.dual_value, (np.ndarray, list)):
                            if len(const.dual_value) > 0:
                                val = const.dual_value[0]
                        elif isinstance(const.dual_value, (int, float)):
                            val = const.dual_value
                    raw_dual_values.append(float(val)) # Ensure float
                
                retrieved_mus = np.array(raw_dual_values).flatten()

                if self.tau > 1e-9: # Apply scaling only if tau is meaningfully positive
                    scaling_factor = 2 * np.sqrt(self.tau)
                    if scaling_factor > 1e-9: # Avoid division by zero / very small number
                        self.mu_ = retrieved_mus / scaling_factor
                    else:
                        # If scaling factor is ~0 but retrieved_mus are not, this indicates an issue
                        # or an extreme scenario. For now, use unscaled if factor is too small.
                        print("Warning: Mu scaling factor in SOCP is near zero. Using unscaled retrieved mus.")
                        self.mu_ = retrieved_mus
                else:
                    # If tau is ~0, the constraint was R @ beta == 0 (equality).
                    # The duals (retrieved_mus) are Lagrange multipliers for these equalities.
                    # These are not directly comparable to the mu_qcqp for beta.T K beta <= 0
                    # via the 2*sqrt(tau) scaling.
                    # For consistency in _compute_support_vectors_and_b, we might need a different
                    # interpretation or accept that mu_ for tau=0 case won't match QCQP's mu.
                    # For now, assign the retrieved duals directly.
                    print("Info: tau is near zero in SOCP. Retrieved 'mu' are duals for equality constraints.")
                    self.mu_ = retrieved_mus

            except Exception as e:
                print(f"Could not retrieve or process/scale dual values for SOCP: {e}")
                self.mu_ = np.zeros(len(socp_kernel_constraints))
        else:
            self.beta_ = np.zeros((m,1))
            self.mu_ = np.zeros(len(self.kernel_R_factors_)) # Ensure mu_ is initialized
            
        return self
    
class SvrSocpMultiKernelL1TraceExplicit(BaseSVR):

    def solve(self, X, y):
        onev, m = self.param(X) 
        
        if self.tau < 0:
            print("Warning: self.tau is negative. SOCP explicit formulation expects tau >= 0.")
            self.status_ = "error_negative_tau"
            self._handle_non_optimal(np.mean(self.y_fit_ if hasattr(self, 'y_fit_') else y)) # Use y_fit_ if available
            return self

        beta_cvx = cp.Variable((m, 1), name="beta_socp")
        objective_term_l1 = -2 * self.epsilon * cp.norm(beta_cvx, 1)
        objective = cp.Maximize(objective_term_l1 + 2 * beta_cvx.T @ y)
        
        socp_kernel_constraints = []
        for R_factor, kernel  in zip(self.kernel_R_factors_, self.kernels_):
            if self.tau < 1e-9: # Effectively zero tau
                constraint = (R_factor @ beta_cvx == 0)
            else:
                norm_arg = R_factor @ beta_cvx
                constraint = cp.norm(norm_arg, 2) <= np.sqrt(self.tau * np.trace(kernel))
            socp_kernel_constraints.append(constraint)
            
        constraints = [
            *socp_kernel_constraints,
            beta_cvx.T @ onev == 0,
            cp.abs(beta_cvx) <= self.C 
        ]
        
        mosek_params = {
            # Primal feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1.0e-9,

            # Dual feasibility tolerance
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1.0e-10,
            
            # Relative optimality gap
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1.0e-9
        }
        
        mosek_params = None
        
        self.problem_ = cp.Problem(objective, constraints)
        self.problem_.solve(solver=cp.MOSEK, verbose=self.verbose)
        
        self.status_ = self.problem_.status
        if self.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            self.objective_value_ = self.problem_.value
            self.beta_ = beta_cvx.value.reshape(-1, 1) if beta_cvx.value is not None else np.zeros((m,1))
            try:
                raw_dual_values = []
                for const in socp_kernel_constraints:
                    # Attempt to retrieve the scalar dual value.
                    # For cp.norm(X,2) <= t, the dual is typically a non-negative scalar.
                    # If const.dual_value is a list/array (e.g. from some conic dual structures),
                    # we assume the relevant scalar is the first element or the value itself if scalar.
                    val = 0.0
                    if const.dual_value is not None:
                        if isinstance(const.dual_value, (np.ndarray, list)):
                            if len(const.dual_value) > 0:
                                val = const.dual_value[0]
                        elif isinstance(const.dual_value, (int, float)):
                            val = const.dual_value
                    raw_dual_values.append(float(val)) # Ensure float
                
                retrieved_mus = np.array(raw_dual_values).flatten()
                self.mu_ = np.zeros(retrieved_mus.size)
                for i, kernel in enumerate(self.kernels_):
                    if self.tau > 1e-9: # Apply scaling only if tau is meaningfully positive
                        scaling_factor = 2 * np.sqrt(self.tau * np.trace(kernel))
                        if scaling_factor > 1e-9: # Avoid division by zero / very small number
                            self.mu_[i] = retrieved_mus[i] / scaling_factor
                        else:
                            # If scaling factor is ~0 but retrieved_mus are not, this indicates an issue
                            # or an extreme scenario. For now, use unscaled if factor is too small.
                            print("Warning: Mu scaling factor in SOCP is near zero. Using unscaled retrieved mus.")
                            self.mu_[i] = retrieved_mus[i]
                    else:
                        # If tau is ~0, the constraint was R @ beta == 0 (equality).
                        # The duals (retrieved_mus) are Lagrange multipliers for these equalities.
                        # These are not directly comparable to the mu_qcqp for beta.T K beta <= 0
                        # via the 2*sqrt(tau) scaling.
                        # For consistency in _compute_support_vectors_and_b, we might need a different
                        # interpretation or accept that mu_ for tau=0 case won't match QCQP's mu.
                        # For now, assign the retrieved duals directly.
                        print("Info: tau is near zero in SOCP. Retrieved 'mu' are duals for equality constraints.")
                        self.mu_[i] = retrieved_mus[i]

            except Exception as e:
                print(f"Could not retrieve or process/scale dual values for SOCP: {e}")
                self.mu_ = np.zeros(len(socp_kernel_constraints))
        else:
            self.beta_ = np.zeros((m,1))
            self.mu_ = np.zeros(len(self.kernel_R_factors_)) # Ensure mu_ is initialized
            
        return self

# %% Example usage (similar to original, but can test SvrSocpMultiKernelL1MuExplicit)
if __name__ == "__main__":
    
    import time # Import time module for profiling
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR as SklearnSVR # Alias
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics.pairwise import laplacian_kernel
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process.kernels import (
        ConstantKernel, Matern, RationalQuadratic, ExpSineSquared
    )
    
    X, y_reg = make_regression( 
        n_samples=100, 
        n_features=1, 
        noise=5,    
        random_state=42 
    )

    kernel_params_list = [
        ("linear", {}),
        ("rbf", {"gamma": 1e-2}),
        ("rbf", {"gamma": 1e-1}),
        ("rbf", {"gamma": 1e0}),
        ("rbf", {"gamma": 1e2}),
        ("poly", {"degree": 2}),
        ("poly", {"degree": 3}),
        ("sigmoid", {}),
        (laplacian_kernel, {"gamma": 1e-1}),
        (laplacian_kernel, {"gamma": 1}),
        (laplacian_kernel, {"gamma": 1e1}),
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
        # (WhiteKernel(), {}),
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

    C_val = 10.0
    epsilon_val = 0.5
    tau_val = 1.0 # Test with tau=1.0 where sqrt(tau)=1
    tau_val = 4.0 # Test with tau=4.0 where sqrt(tau)=2, scaling factor would be 4
    tau_val = 0.25 # Test with tau=0.25 where sqrt(tau)=0.5, scaling factor would be 1
    tau_val = 100


    print("--- Testing SvrQcqpMultiKernelL1Mu (Original Formulation) ---")
    model_qcqp_multi = SvrQcqpMultiKernelL1Mu(
        C=C_val, epsilon=epsilon_val, tau=tau_val,
        kernel_params=kernel_params_list, verbose=False, kronecker_kernel=False
    )
    start_time_qcqp = time.time()
    model_qcqp_multi.fit(X, y_reg)
    end_time_qcqp = time.time()
    print(f"QCQP Fit Time: {end_time_qcqp - start_time_qcqp:.4f} seconds")
    print(f"QCQP Status: {model_qcqp_multi.status_}")
    if model_qcqp_multi.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        if model_qcqp_multi.problem_ and hasattr(model_qcqp_multi.problem_, 'solver_stats') and model_qcqp_multi.problem_.solver_stats:
            print(f"QCQP Compilation Time: {model_qcqp_multi.problem_.compilation_time:.4f} seconds")
            print(f"QCQP Solve Time: {model_qcqp_multi.problem_.solver_stats.solve_time:.4f} seconds")
        y_pred_multi_qcqp = model_qcqp_multi.predict(X)
        print(f"QCQP Objective: {model_qcqp_multi.objective_value_}")
        print(f"QCQP b: {model_qcqp_multi.b_:.6f}")
        print(f"QCQP Support Vectors: {model_qcqp_multi.sup_num_}")
        print(f"QCQP Mu: {np.round(model_qcqp_multi.mu_, 6)}")
        print(f"QCQP MAE: {mean_absolute_error(y_reg, y_pred_multi_qcqp):.6f}")
    else:
        print(f"QCQP Model optimization failed.")



    print("--- Testing SvrQcqpMultiKernelL1Trace (Original Formulation) ---")
    model_qcqp_multi_trace = SvrQcqpMultiKernelL1Trace(
        C=C_val, epsilon=epsilon_val, tau=tau_val,
        kernel_params=kernel_params_list, verbose=False, kronecker_kernel=False
    )
    start_time_qcqp = time.time()
    model_qcqp_multi_trace.fit(X, y_reg)
    end_time_qcqp = time.time()
    print(f"QCQP Fit Time: {end_time_qcqp - start_time_qcqp:.4f} seconds")
    print(f"QCQP Status: {model_qcqp_multi_trace.status_}")
    if model_qcqp_multi_trace.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        if model_qcqp_multi_trace.problem_ and hasattr(model_qcqp_multi_trace.problem_, 'solver_stats') and model_qcqp_multi_trace.problem_.solver_stats:
            print(f"QCQP Compilation Time: {model_qcqp_multi_trace.problem_.compilation_time:.4f} seconds")
            print(f"QCQP Solve Time: {model_qcqp_multi_trace.problem_.solver_stats.solve_time:.4f} seconds")
        y_pred_multi_qcqp = model_qcqp_multi_trace.predict(X)
        print(f"QCQP Objective: {model_qcqp_multi_trace.objective_value_}")
        print(f"QCQP b: {model_qcqp_multi_trace.b_:.6f}")
        print(f"QCQP Support Vectors: {model_qcqp_multi_trace.sup_num_}")
        print(f"QCQP Mu: {np.round(model_qcqp_multi_trace.mu_, 6)}")
        print(f"QCQP MAE: {mean_absolute_error(y_reg, y_pred_multi_qcqp):.6f}")
    else:
        print(f"QCQP Model optimization failed.")


    print("\n--- Testing SvrSocpMultiKernelL1MuExplicit (Explicit SOCP Formulation) ---")
    model_socp_explicit = SvrSocpMultiKernelL1MuExplicit(
        C=C_val, epsilon=epsilon_val, tau=tau_val,
        kernel_params=kernel_params_list, verbose=False, kronecker_kernel=False
    )
    start_time_socp = time.time()
    model_socp_explicit.fit(X, y_reg)
    end_time_socp = time.time()
    print(f"SOCP Fit Time: {end_time_socp - start_time_socp:.4f} seconds")
    print(f"SOCP Status: {model_socp_explicit.status_}")
    if model_socp_explicit.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        if model_socp_explicit.problem_ and hasattr(model_socp_explicit.problem_, 'solver_stats') and model_socp_explicit.problem_.solver_stats:
            print(f"SOCP Compilation Time: {model_socp_explicit.problem_.compilation_time:.4f} seconds")
            print(f"SOCP Solve Time: {model_socp_explicit.problem_.solver_stats.solve_time:.4f} seconds")
        y_pred_multi_socp = model_socp_explicit.predict(X)
        print(f"SOCP Objective: {model_socp_explicit.objective_value_}")
        print(f"SOCP b: {model_socp_explicit.b_:.6f}")
        print(f"SOCP Support Vectors: {model_socp_explicit.sup_num_}")
        print(f"SOCP Mu (scaled): {np.round(model_socp_explicit.mu_, 6)}")
        print(f"SOCP MAE: {mean_absolute_error(y_reg, y_pred_multi_socp):.6f}")

        # Compare betas
        if model_qcqp_multi.beta_ is not None and model_socp_explicit.beta_ is not None:
            beta_diff = np.linalg.norm(model_qcqp_multi.beta_ - model_socp_explicit.beta_)
            print(f"Norm of difference in beta values: {beta_diff:.6e}")
    else:
        print(f"SOCP mu Model optimization failed.")
        

    print("\n--- Testing SvrSocpMultiKernelL1TraceExplicit (Explicit SOCP Formulation) ---")
    model_socp_explicit_trace = SvrSocpMultiKernelL1TraceExplicit(
        C=C_val, epsilon=epsilon_val, tau=tau_val,
        kernel_params=kernel_params_list, verbose=False, kronecker_kernel=False
    )
    start_time_socp = time.time()
    model_socp_explicit_trace.fit(X, y_reg)
    end_time_socp = time.time()
    print(f"SOCP Fit Time: {end_time_socp - start_time_socp:.4f} seconds")
    print(f"SOCP Status: {model_socp_explicit_trace.status_}")
    if model_socp_explicit_trace.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        if model_socp_explicit_trace.problem_ and hasattr(model_socp_explicit_trace.problem_, 'solver_stats') and model_socp_explicit_trace.problem_.solver_stats:
            print(f"SOCP Compilation Time: {model_socp_explicit_trace.problem_.compilation_time:.4f} seconds")
            print(f"SOCP Solve Time: {model_socp_explicit_trace.problem_.solver_stats.solve_time:.4f} seconds")
        y_pred_multi_socp = model_socp_explicit_trace.predict(X)
        print(f"SOCP Objective: {model_socp_explicit_trace.objective_value_}")
        print(f"SOCP b: {model_socp_explicit_trace.b_:.6f}")
        print(f"SOCP Support Vectors: {model_socp_explicit_trace.sup_num_}")
        print(f"SOCP Mu (scaled): {np.round(model_socp_explicit_trace.mu_, 6)}")
        print(f"SOCP MAE: {mean_absolute_error(y_reg, y_pred_multi_socp):.6f}")

        # Compare betas
        if model_qcqp_multi.beta_ is not None and model_socp_explicit_trace.beta_ is not None:
            beta_diff = np.linalg.norm(model_qcqp_multi.beta_ - model_socp_explicit_trace.beta_)
            print(f"Norm of difference in beta values: {beta_diff:.6e}")
    else:
        print(f"SOCP trace Model optimization failed.")

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y_reg, color='black', label='Data', s=20, alpha=0.7)
    if model_qcqp_multi.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        plt.plot(np.sort(X.flatten()), model_qcqp_multi.predict(np.sort(X, axis=0)), label=f'QCQP mu (b={model_qcqp_multi.b_:.2f})', linestyle='--')
    if model_qcqp_multi_trace.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        plt.plot(np.sort(X.flatten()), model_qcqp_multi_trace.predict(np.sort(X, axis=0)), label=f'QCQP trace (b={model_qcqp_multi_trace.b_:.2f})', linestyle='--')
    if model_socp_explicit.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        plt.plot(np.sort(X.flatten()), model_socp_explicit.predict(np.sort(X, axis=0)), label=f'SOCP mu Explicit (b={model_socp_explicit.b_:.2f})', linestyle=':')
    if model_socp_explicit_trace.status_ in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE, "optimal"):
        plt.plot(np.sort(X.flatten()), model_socp_explicit_trace.predict(np.sort(X, axis=0)), label=f'SOCP trace Explicit (b={model_socp_explicit_trace.b_:.2f})', linestyle=':')


    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    plt.title(f"SVR Model Predictions (tau={tau_val})")
    plt.show()


# %%
