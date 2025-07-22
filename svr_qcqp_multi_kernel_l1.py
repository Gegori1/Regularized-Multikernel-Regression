# %%
import numpy as np
import cvxpy as cp
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics.pairwise import pairwise_kernels
import warnings

class BaseSVR:
    """
    Base class for Support Vector Regression models.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C. Must be strictly positive.
    epsilon : float, default=1.0
        Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
        within which no penalty is associated in the training loss function
        with points predicted within a distance epsilon from the actual value.
    tau : float, default=1.0
        Parameter for the multi-kernel learning formulation.
    kernel_params : list of tuple, default=[("linear", {})]
        List of kernel specifications. Each tuple should contain:
        - kernel_name (str or callable): Name of the kernel (e.g., 'rbf', 'linear')
          or a callable function that computes the kernel.
        - kernel_args (dict): Parameters for the kernel.
    c : float, default=1.0
        Deprecated or unused parameter (appears to be overwritten if trace_min is True).
    verbose : bool, default=False
        Enable verbose output during optimization.
    trace_min : bool, default=True
        Whether to set the 'c' parameter based on the trace of kernel matrices.
    trace_min_factor : int, default=1
        Factor to multiply with the sum of traces if trace_min is True.
    kronecker_kernel : bool, default=False
        Whether to add a Kronecker delta kernel (identity matrix) to the list of kernels.
    """
    def __init__(
            self, C: float=1.0,
            epsilon: float=1.0,
            tau: float=1.0,
            kernel_params: list=[("linear", {})],
            c: float=1.0, # This parameter seems to be potentially overwritten or its role needs clarification.
            verbose: bool=False,
            trace_min: bool=True,
            trace_min_factor: int=1,
            kronecker_kernel: bool=False,
        ):
        
        self.C = C
        self.epsilon = epsilon
        self.tau = tau
        self.kernel_params = kernel_params
        self.c_init = c # Store initial c, as self.c might be changed
        self.verbose = verbose
        self.trace_min = trace_min
        self.trace_min_factor = trace_min_factor
        self.kronecker_kernel = kronecker_kernel
        
        # Attributes to be set during fit
        self.kernels = []
        self.trace = 0.0
        self.beta = None
        self.mu = None
        self.b = 0.0
        self.support_vectors = np.array([])
        self.support_labels = np.array([])
        self.support_beta = np.array([])
        self.sup_num = 0
        self.status = "not_fitted"
        self.objective_value = None
        self.problem = None # To store the CVXPY problem object

    def _check_x_y(self, X, y):
        """Validate and reshape X and y."""
        X, y = check_X_y(X, y, multi_output=False, y_numeric=True)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return X, y

    def _calculate_kernel_matrix(self, X1, X2, kernel_name, kernel_args):
        """Helper function to compute a single kernel matrix."""
        if X2 is None:
            X2 = X1
        if isinstance(kernel_name, str):
            return pairwise_kernels(X1, X2, metric=kernel_name, filter_params=True, **kernel_args)
        elif callable(kernel_name):
            return kernel_name(X1, X2, **kernel_args)
        else:
            raise ValueError(f"Invalid kernel_name type: {type(kernel_name)}. Must be str or callable.")

    def delta_kronecker_kernel(self, X, Y=None):
        """
        Computes the Kronecker delta kernel.
        Returns an identity matrix if X and Y are the same, otherwise compares rows.
        """
        X = np.asarray(X)
        if Y is None or X is Y or np.array_equal(X,Y): # Added X is Y for efficiency
            return np.eye(X.shape[0])
        
        Y = np.asarray(Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features for delta_kronecker_kernel.")

        # Efficient way to compare all pairs of rows for exact matches
        # This can be slow for large X, Y if they are not identical.
        # The original broadcasting method is fine if feature dimensions are not too large.
        kernel_matrix = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                if np.array_equal(X[i], Y[j]):
                    kernel_matrix[i, j] = 1
        return kernel_matrix


    def _prepare_kernels(self, X):
        """
        Prepares kernel matrices and calculates their combined trace.
        Stores kernel matrices in self.kernels and sum of traces in self.trace.
        """
        m = X.shape[0]
        self.kernels = []
        # self.kernels_scaled = []
        current_trace = 0.0 # Use a local variable for accumulation

        for kern_name, kern_args in self.kernel_params:
            kernel_matrix = self._calculate_kernel_matrix(X, X, kern_name, kern_args)
            self.kernels.append(kernel_matrix)
            # self.kernels_scaled.append(kernel_matrix / np.trace(kernel_matrix))
            current_trace += np.trace(kernel_matrix)
        
        if self.kronecker_kernel:
            # The Kronecker kernel here is essentially an identity matrix when X is compared with itself.
            # It's like adding a penalty for each sample individually.
            self.kernels.append(np.eye(m)) 
            # self.kernels_scaled.append(np.eye(m) / m)
            current_trace += m
            
        self.trace = current_trace
        
        # Update self.c based on trace if trace_min is True
        if self.trace_min:
            self.c = self.trace * self.trace_min_factor
        else:
            self.c = self.c_init # Use the initial c if not trace_min

        return np.ones((m, 1)), m
    
    def _solve_cvxpy_problem(self, objective, constraints, mosek_params=None):
        """
        Helper method to define and solve the CVXPY problem.
        """
        self.problem = cp.Problem(objective, constraints)
        
        solver_kwargs = {'solver': cp.MOSEK, 'verbose': self.verbose}
        if mosek_params:
            solver_kwargs['mosek_params'] = mosek_params
        
        # Suppress specific CVXPY warnings if necessary, though it's better to ensure convexity
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Forming a nonconvex expression quad_form\\(x, indefinite\\)\\.")
            # Note: cp.psd_wrap should ideally make the matrix PSD, 
            # but if K is numerically indefinite, this warning might still appear.
            # Ensure kernels are truly PSD or handle non-convexity appropriately.
            self.problem.solve(**solver_kwargs)
        
        self.status = self.problem.status
        if self.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE): # Consider OPTIMAL_INACCURATE as a success
            self.objective_value = self.problem.value
            # Ensure beta is a NumPy array
            if hasattr(self.beta, 'value'):
                 self.beta = self.beta.value.reshape(-1, 1) if self.beta.value is not None else np.zeros((self.X_fit_.shape[0], 1))
            else: # if self.beta was already a numpy array (e.g. in non_optimal)
                 self.beta = np.zeros((self.X_fit_.shape[0], 1))

            return True
        else:
            print(f"Optimization failed or was not optimal. Status: {self.status}")
            if self.problem.solver_stats:
                 print(f"Solver stats: {self.problem.solver_stats}")
            return False

    def solve(self, X, y):
        """Placeholder for the main optimization logic in derived classes."""
        raise NotImplementedError("The solve method must be implemented by subclasses.")

    def _handle_non_optimal(self, y_mean):
        """Sets default values when optimization is not successful."""
        self.sup_num = 0
        self.support_vectors = np.array([]) # Ensure correct empty shape
        self.support_labels = np.array([])
        self.support_beta = np.array([])
        self.b = y_mean # A common heuristic for non-optimal SVR
        self.mu = np.zeros(len(self.kernel_params) + (1 if self.kronecker_kernel else 0)) # Initialize mu correctly
        self.beta = np.zeros((self.X_fit_.shape[0], 1)) # Initialize beta correctly

    def _compute_support_vectors_and_b(self, X, y):
        """Computes support vectors and the bias term 'b' after successful optimization."""
        if self.beta is None: # Should not happen if solve was successful
            self._handle_non_optimal(np.mean(y))
            return

        # Filter mus (dual variables for kernel constraints)
        # This heuristic might need adjustment based on problem scale and solver precision
        if self.mu is not None:
            significant_mu_threshold = 1e-7 
            if np.any(self.mu >= 1e-4): # If some mus are large, use a slightly larger threshold
                significant_mu_threshold = 1e-4
            self.mu = np.where(np.abs(self.mu) <= significant_mu_threshold, 0, self.mu)
        else: # Should be set by solver
            self.mu = np.zeros(len(self.kernel_params) + (1 if self.kronecker_kernel else 0))


        # Identify support vectors based on beta values
        # Beta values close to zero mean the corresponding sample is not a support vector.
        beta_flat = self.beta.flatten()
        support_indices = np.abs(beta_flat) > 1e-5 # Threshold for identifying SVs

        if np.sum(support_indices) == 0:
            self._handle_non_optimal(np.mean(y))
            return

        self.sup_num = np.sum(support_indices)
        self.support_vectors = X[support_indices, :]
        self.support_labels = y[support_indices] # y is already (n_samples, 1)
        self.support_beta = self.beta[support_indices]

        # Compute b (bias term)
        # b is typically computed using unbounded support vectors
        # (those for which |beta_i| < C)
        
        # Calculate combined kernel matrix for support vectors
        # K_sv = sum(mu_k * Kernel_k(SV, SV))
        K_sv_sum = np.zeros((self.sup_num, self.sup_num))
        
        # Iterate through original kernels
        for i, (kern_name, kern_args) in enumerate(self.kernel_params):
            if self.mu[i] != 0: # Only consider kernels with non-zero dual variables
                kern = self._calculate_kernel_matrix(self.support_vectors, self.support_vectors, kern_name, kern_args)
                K_sv_sum += self.mu[i] * (kern / np.trace(kern))
        
        # Add Kronecker kernel contribution if applicable
        if self.kronecker_kernel and self.mu[-1] != 0:
            # For SVs with themselves, delta_kronecker_kernel is an identity matrix of size sup_num
            K_sv_sum += self.mu[-1] * (np.eye(self.sup_num) / X.shape[0])

        # Determine unbounded support vectors
        # These are SVs where beta_i is not at the C or -C boundary.
        # Add a small tolerance for floating point comparisons.
        epsilon_tolerance = 1e-5 
        unbounded_sv_indices = (np.abs(self.support_beta.flatten()) < (self.C - epsilon_tolerance))
        
        # Compute b
        # b = y_i - sum(beta_j * K_ij) - epsilon (for beta_i > 0)
        # b = y_i - sum(beta_j * K_ij) + epsilon (for beta_i < 0)
        # This can be written as: b = y_i - sum(beta_j * K_ij) - sign(beta_i) * epsilon
        # Averaging over unbounded SVs is more robust.

        if np.any(unbounded_sv_indices):
            # Use only unbounded SVs for b calculation
            # K_sv_sum_unbounded_rows relates to K(SV_all, SV_unbounded)
            # We need K(SV_unbounded, SV_all) @ support_beta_all
            # Or, more simply, (support_labels_unbounded - (K_sv_sum @ support_beta)_unbounded - sign(support_beta_unbounded) * epsilon)

            # f_i = sum_j beta_j K(x_i, x_j)
            # For an unbounded SV i: y_i - f_i = sign(beta_i) * epsilon
            # So, b = y_i - f_i - sign(beta_i) * epsilon = y_i - (K_sv_sum @ self.support_beta)_i - np.sign(self.support_beta_i) * self.epsilon
            
            f_unbounded = (K_sv_sum @ self.support_beta)[unbounded_sv_indices]
            y_unbounded = self.support_labels[unbounded_sv_indices]
            beta_unbounded = self.support_beta[unbounded_sv_indices]
            
            # sign_beta_epsilon = np.sign(beta_unbounded) * self.epsilon
            # A more direct way from primal-dual relations for SVR:
            # For unbounded SVs (0 < alpha_i < C or 0 < alpha_i* < C),
            # w^T phi(x_i) + b = y_i +/- epsilon
            # Here, beta = alpha - alpha*
            # xi_upper_i = f(x_i) - y_i - epsilon  (where beta_i > 0, so alpha_i > 0)
            # xi_lower_i = y_i - f(x_i) - epsilon  (where beta_i < 0, so alpha_i* > 0)
            # For these SVs, the constraints are active.
            # If beta_i > 0 (and not C), then f(x_i) - y_i = epsilon. So b = epsilon - (f(x_i) - b) + y_i
            # If beta_i < 0 (and not -C), then y_i - f(x_i) = epsilon. So b = -epsilon - (f(x_i) -b) + y_i

            # Let pred_on_sv_unbounded = (K_sv_sum @ self.support_beta)[unbounded_sv_indices]
            # b_values = y_unbounded.flatten() - pred_on_sv_unbounded.flatten() - np.sign(beta_unbounded.flatten()) * self.epsilon
            
            # Simplified: For an unbounded SV_i, y_i - sum_j beta_j K(x_i, x_j) - b = sign(beta_i) * epsilon
            # So, b = y_i - sum_j beta_j K(x_i, x_j) - sign(beta_i) * epsilon
            
            # The original way:
            # epsilon_beta = np.where(self.support_beta >= 0, -self.epsilon, self.epsilon) # This seems reversed.
            # If beta_i > 0, error is y_i - f_i >= epsilon. If beta_i < 0, error is f_i - y_i >= epsilon.
            # For active constraints (unbounded SVs):
            # y_i - f_i = epsilon  if beta_i > 0
            # f_i - y_i = epsilon  if beta_i < 0
            # So, b = y_i - (f_i - b) - epsilon  for beta_i > 0
            # b = y_i - (K_sv_sum @ self.support_beta)[i] - epsilon
            # And b = y_i - (K_sv_sum @ self.support_beta)[i] + epsilon for beta_i < 0
            
            # Let's use the standard formulation: b = y_i - sum_j alpha_j y_j K(x_i, x_j)
            # Or for epsilon-SVR, for an SV_i not at bound C:
            # b = y_i - sum_{sv_j} beta_j K(x_i, x_j) - epsilon  (if beta_i > 0, i.e. alpha_i active)
            # b = y_i - sum_{sv_j} beta_j K(x_i, x_j) + epsilon  (if beta_i < 0, i.e. alpha_star_i active)
            
            b_sum = 0
            count = 0
            preds_on_sv = K_sv_sum @ self.support_beta
            for i in range(self.sup_num):
                if unbounded_sv_indices[i]:
                    pred_i = preds_on_sv[i]
                    if self.support_beta[i] > epsilon_tolerance: # Effectively alpha_i is active
                        b_sum += self.support_labels[i] - pred_i - self.epsilon
                    elif self.support_beta[i] < -epsilon_tolerance: # Effectively alpha_star_i is active
                        b_sum += self.support_labels[i] - pred_i + self.epsilon
                    else: # beta_i is very close to 0, should ideally not be an unbounded SV if C is large
                          # Or it means both alpha and alpha_star are zero, which means it's not an SV.
                          # This case should be rare for true unbounded SVs.
                        continue 
                    count += 1
            self.b = (b_sum / count) if count > 0 else np.mean(y) # Fallback if no clear unbounded SVs

        else: # All SVs are on the boundary |beta_i| = C
              # This is less common, implies data might be perfectly separable or C is too small.
              # In this case, b is a range. A common choice is to average over all SVs.
            preds_on_sv = K_sv_sum @ self.support_beta
            b_values_lower = self.support_labels - preds_on_sv - self.epsilon
            b_values_upper = self.support_labels - preds_on_sv + self.epsilon
            self.b = (np.max(b_values_lower[self.support_beta.flatten() > epsilon_tolerance]) + 
                      np.min(b_values_upper[self.support_beta.flatten() < -epsilon_tolerance])) / 2
            if not np.isfinite(self.b): # Fallback if no SVs satisfy conditions
                 self.b = np.mean(y)


    def fit(self, X, y):
        """
        Fit the SVR model according to the given training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = self._check_x_y(X, y)
        self.X_fit_ = X # Store X for use in _handle_non_optimal if needed
        self.y_fit_ = y # Store y for use in _handle_non_optimal if needed
        self.m = X.shape[0]
        # Prepare kernels and other parameters
        self._prepare_kernels(X) # This now also sets self.c if trace_min is True
        
        # Solve the optimization problem
        # The actual CVXPY variable for beta should be defined in the subclass's solve method
        # and then assigned to self.beta.value
        try:
            self.solve(X, y) # This method should call _solve_cvxpy_problem
        except cp.SolverError as e:
            self.status = "solver_error"
            print(f"CVXPY Solver failed during problem setup or solve call: {e}")
            self._handle_non_optimal(np.mean(y))
            return self
        except Exception as e:
            self.status = "unexpected_error"
            print(f"An unexpected error occurred during fitting: {e}")
            self._handle_non_optimal(np.mean(y))
            return self
        
        if self.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            print(f"Optimization was not successful. Status: {self.status}. Using default prediction.")
            self.beta = np.zeros((X.shape[0],1)) # Ensure beta is initialized
            self._handle_non_optimal(np.mean(y))
            return self
        
        # Compute support vectors and bias term b
        self._compute_support_vectors_and_b(X, y)      

        return self
    
    def decision_function(self, X):
        """
        Calculate the decision function (predicted values before thresholding for classification).
        For SVR, this is the predicted regression value.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The decision function of the input samples.
        """
        X = check_array(X) # Validate X
        if self.sup_num == 0 or self.support_vectors is None or self.support_vectors.shape[0] == 0:
            # If no support vectors, predict the mean or a constant b
            return np.full(X.shape[0], self.b)

        # K_pred = sum(mu_k * Kernel_k(SV, X_new))
        K_pred_sum = np.zeros((self.sup_num, X.shape[0]))
        
        active_kernels_params = []
        if self.kronecker_kernel: # kronecker kernel params are not in self.kernel_params
            active_kernels_params = list(self.kernel_params) + [('kronecker', {})] # Placeholder
        else:
            active_kernels_params = list(self.kernel_params)

        # Ensure mu matches the number of effective kernels
        num_effective_kernels = len(self.kernel_params) + (1 if self.kronecker_kernel else 0)
        if self.mu is None or len(self.mu) != num_effective_kernels:
             # This can happen if fit wasn't successful or mu wasn't set correctly
            # print("Warning: self.mu is not properly initialized. Assuming equal weights or re-evaluating.")
            # Fallback: if mu is not set, perhaps assume all original kernels are used without weighting by mu
            # Or, more safely, if mu is missing, it implies an issue in `solve` or SV computation.
            # For now, if mu is problematic, we might not be able to use the multi-kernel aspect correctly.
            # A simple sum without mu might be one interpretation if mu is meant for learning kernel weights.
            # However, in the QCQP formulation, mu are duals. If they are zero, kernel is not used.
            # If mu is None, it's an error state.
            if self.mu is None: # Default to no kernel contribution if mu is missing
                # This means only 'b' will be returned if K_pred_sum remains zero
                pass


        # Iterate through original kernels
        for kk, (i, (kern_name, kern_args)) in zip(self.kernels, enumerate(self.kernel_params)):
            if self.mu is not None and i < len(self.mu) and self.mu[i] != 0:
                kern = self._calculate_kernel_matrix(self.support_vectors, X, kern_name, kern_args)
                K_pred_sum += self.mu[i] * (kern / np.trace(kk))
        
        # Add Kronecker kernel contribution if applicable
        if self.kronecker_kernel and self.mu is not None and len(self.mu) == num_effective_kernels and self.mu[-1] != 0:
            kern = self.delta_kronecker_kernel(self.support_vectors, X)
            K_pred_sum += self.mu[-1] * (kern / self.m)
        
        if np.all(self.mu == 0) and self.sup_num > 0 : # All mu are zero, but SVs exist
             # This case is unusual. If all mu are zero, it implies the kernel constraints
             # were not active or had zero dual values.
             # Standard SVR prediction would be sum beta_i K(sv_i, x) + b.
             # If mu's are weights for a *single* combined kernel, and all mu are zero,
             # then the kernel term is zero.
             # If mu's are duals for *constraints* on individual kernels, then this is different.
             # The provided formulation seems to use mu as duals for quadratic constraints.
             # The objective doesn't directly use a sum of mu_k * K_k.
             # The prediction f(x) = sum_i beta_i * (sum_k mu_k K_k(x_i, x)) + b is not standard.
             # Standard SVR: f(x) = sum_i beta_i K_combined(x_i, x) + b
             # If K_combined = sum_k mu_k K_k, then the prediction is correct.
             # Let's assume the decision function is sum_beta_sv * K_combined(SV, X) + b
             # where K_combined is implicitly defined by the weighted sum using mu.
             pass # K_pred_sum would be zero, so prediction is just b.

        prediction = self.support_beta.T @ K_pred_sum + self.b
        return prediction.flatten() # Ensure 1D array

    def predict(self, X):
        """
        Perform regression on samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        return self.decision_function(X)
    
    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score # Local import
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            "C": self.C, 
            "epsilon": self.epsilon,
            "tau": self.tau,
            "kernel_params": self.kernel_params,
            "c": self.c_init, # Return initial c
            "verbose": self.verbose,
            "trace_min": self.trace_min,
            "trace_min_factor": self.trace_min_factor,
            "kronecker_kernel": self.kronecker_kernel
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if key == 'c':
                self.c_init = value # Update initial c
            setattr(self, key, value)
        # Re-evaluate self.c if trace_min is True and related params changed
        if self.trace_min and ('trace_min_factor' in params or 'trace_min' in params):
            if hasattr(self, 'trace') and self.trace is not None: # if trace is already computed
                 self.c = self.trace * self.trace_min_factor
            # else, it will be computed during fit in _prepare_kernels
        elif not self.trace_min and 'c' in params:
            self.c = self.c_init

        return self

# %%  

class SvrQcqpMultiKernelL1Mu(BaseSVR):
    """
    SVR with multiple kernels using a Quadratically Constrained Quadratic Program (QCQP)
    formulation (L1 epsilon-insensitive loss).
    Each kernel K_i contributes a constraint: beta' * K_i * beta <= tau_i (here, a global tau).
    The dual variables 'mu' correspond to these quadratic constraints.
    """
    def solve(self, X, y):
        """
        Set up and solve the SVR optimization problem using CVXPY.
        Objective: Maximize -2*epsilon * ||beta||_1 + 2 * beta^T * y
        Constraints:
            beta^T * K_j * beta <= tau  (for each kernel K_j)
            beta^T * 1 = 0
            -C <= beta_i <= C
        """
        onev, m = self._prepare_kernels(X) # self.kernels is now populated
            
        # CVXPY Variables
        # self.beta is already an attribute, but we need a CVXPY variable for optimization
        beta_cvx = cp.Variable((m, 1), name="beta_cvx")

        # Objective Function
        # Using cp.norm(beta_cvx, 1) for the L1 norm term
        objective = cp.Maximize(-2*self.epsilon * cp.norm(beta_cvx, 1) + 2 * beta_cvx.T @ y)
        
        # Constraints
        self.constraints_kernels_cvx_ = [] # Store CVXPY constraint objects to access duals
        for K_matrix in self.kernels:
            # Ensure K_matrix is symmetric PSD for cp.quad_form
            # cp.psd_wrap attempts to make it PSD for the solver.
            constraint = cp.quad_form(beta_cvx, cp.psd_wrap(K_matrix)) <= self.tau
            self.constraints_kernels_cvx_.append(constraint)
            
        constraints = [
            *self.constraints_kernels_cvx_,
            beta_cvx.T @ onev == 0,
            cp.abs(beta_cvx) <= self.C # Compact constraint for -C <= beta <= C
        ]

        # Assign CVXPY variable to self.beta so _solve_cvxpy_problem can access its .value
        self.beta = beta_cvx 

        if self._solve_cvxpy_problem(objective, constraints):
            # Extract dual variables (mu) for the kernel constraints
            if self.constraints_kernels_cvx_:
                try:
                    self.mu = np.array([const.dual_value[0] if const.dual_value is not None else 0.0 
                                        for const in self.constraints_kernels_cvx_]).flatten()
                except Exception as e:
                    print(f"Could not retrieve dual values: {e}")
                    self.mu = np.zeros(len(self.constraints_kernels_cvx_))
            else:
                self.mu = np.array([])
        else:
            # If solve fails, _solve_cvxpy_problem handles setting status
            # and self.beta will be a zero array from _handle_non_optimal via fit method
            self.mu = np.zeros(len(self.kernels)) # Ensure mu is initialized even on failure

        return self # self.status and self.beta (as np array) are set by _solve_cvxpy_problem
    

# %%  

class SvrQcqpMultiKernelL1Trace(BaseSVR):
    """
    SVR with multiple kernels using a Quadratically Constrained Quadratic Program (QCQP)
    formulation (L1 epsilon-insensitive loss).
    Each kernel K_i contributes a constraint: beta' * K_i * beta <= tau_i (here, a global tau).
    The dual variables 'mu' correspond to these quadratic constraints.
    """
    def solve(self, X, y):
        """
        Set up and solve the SVR optimization problem using CVXPY.
        Objective: Maximize -2*epsilon * ||beta||_1 + 2 * beta^T * y
        Constraints:
            beta^T * K_j * beta <= tau  (for each kernel K_j)
            beta^T * 1 = 0
            -C <= beta_i <= C
        """
        onev, m = self._prepare_kernels(X) # self.kernels is now populated
            
        # CVXPY Variables
        # self.beta is already an attribute, but we need a CVXPY variable for optimization
        beta_cvx = cp.Variable((m, 1), name="beta_cvx")

        # Objective Function
        # Using cp.norm(beta_cvx, 1) for the L1 norm term
        objective = cp.Maximize(-2*self.epsilon * cp.norm(beta_cvx, 1) + 2 * beta_cvx.T @ y)
        
        # Constraints
        self.constraints_kernels_cvx_ = [] # Store CVXPY constraint objects to access duals
        for K_matrix in self.kernels:
            # Ensure K_matrix is symmetric PSD for cp.quad_form
            # cp.psd_wrap attempts to make it PSD for the solver.
            constraint = cp.quad_form(beta_cvx, cp.psd_wrap(K_matrix)) <= self.tau * np.trace(K_matrix)
            self.constraints_kernels_cvx_.append(constraint)
            
        constraints = [
            *self.constraints_kernels_cvx_,
            beta_cvx.T @ onev == 0,
            cp.abs(beta_cvx) <= self.C # Compact constraint for -C <= beta <= C
        ]

        # Assign CVXPY variable to self.beta so _solve_cvxpy_problem can access its .value
        self.beta = beta_cvx 

        if self._solve_cvxpy_problem(objective, constraints):
            # Extract dual variables (mu) for the kernel constraints
            if self.constraints_kernels_cvx_:
                try:
                    self.mu = np.array([const.dual_value[0] if const.dual_value is not None else 0.0 
                                        for const in self.constraints_kernels_cvx_]).flatten()
                except Exception as e:
                    print(f"Could not retrieve dual values: {e}")
                    self.mu = np.zeros(len(self.constraints_kernels_cvx_))
            else:
                self.mu = np.array([])
        else:
            # If solve fails, _solve_cvxpy_problem handles setting status
            # and self.beta will be a zero array from _handle_non_optimal via fit method
            self.mu = np.zeros(len(self.kernels)) # Ensure mu is initialized even on failure

        return self # self.status and self.beta (as np array) are set by _solve_cvxpy_problem
    
# %%

class SvrQcqpMultiKernelL1EpigraphTrace(BaseSVR):
    """
    SVR with multiple kernels using an epigraph formulation for trace normalization.
    Objective: Maximize -2*eps*||beta||_1 + 2*beta^T*y - tau*t
    Constraints:
        t >= (beta^T * K_j * beta) / trace(K_j) (for each kernel K_j)
        beta^T * 1 = 0
        -C <= beta_i <= C
    """
    def solve(self, X, y):
        onev, m = self._prepare_kernels(X)
        
        # CVXPY Variables
        beta_cvx = cp.Variable((m, 1), name="beta_cvx")
        t_epi = cp.Variable(1, name="t_epigraph") # Epigraph variable

        # Objective Function
        objective = cp.Maximize(-2*self.epsilon * cp.norm(beta_cvx, 1) + 2 * beta_cvx.T @ y - self.tau * t_epi)
        
        # Constraints
        self.constraints_kernels_cvx_ = []
        for i, K_matrix in enumerate(self.kernels):
            trace_K = np.trace(K_matrix)
            if trace_K <= 1e-9: # Avoid division by zero or very small trace
                # If trace is effectively zero, this kernel might be problematic or zero itself.
                # One option: constraint beta^T K beta <= 0 (if K is PSD, means K beta = 0)
                # Or simply skip if it implies K is null.
                # For now, if trace is zero, we cannot normalize.
                # This constraint might become infeasible or ill-defined.
                # Consider adding a small regularization to K or handling this case.
                print(f"Warning: Kernel {i} has trace ~0. Constraint might be ill-defined.")
                # Fallback: If trace is zero, and K is PSD, beta^T K beta must be >=0.
                # If beta^T K beta is also to be small (related to t_epi), this is complex.
                # For simplicity, if trace is zero, perhaps this kernel shouldn't constrain t_epi.
                # Or, if K is not the zero matrix, beta must be in its null space for quad_form to be 0.
                # This formulation requires trace_K > 0.
                if np.allclose(K_matrix, 0): continue # Skip null kernels
                # Add constraint that quad_form must be zero if trace is zero.
                # constraint = cp.quad_form(beta_cvx, cp.psd_wrap(K_matrix)) <= 1e-9 # Effectively zero
            else:
                constraint = (cp.quad_form(beta_cvx, cp.psd_wrap(K_matrix))) <= t_epi
            self.constraints_kernels_cvx_.append(constraint)
            
        constraints = [
            *self.constraints_kernels_cvx_,
            beta_cvx.T @ onev == 0,
            cp.abs(beta_cvx) <= self.C
        ]

        # MOSEK parameters (optional, as an example)
        desired_tol = 1e-14
        mosek_params = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': desired_tol, # Primal feasibility for conic
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': desired_tol, # Dual feasibility for conic
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': desired_tol, # Relative duality gap for conic
            'MSK_DPAR_INTPNT_QO_TOL_PFEAS': desired_tol, # Primal feasibility for QCQP
            'MSK_DPAR_INTPNT_QO_TOL_DFEAS': desired_tol, # Dual feasibility for QCQP
            'MSK_DPAR_INTPNT_QO_TOL_REL_GAP': desired_tol, # Relative duality gap for QCQP
            'MSK_IPAR_LOG': 1, # Enable MOSEK log
        }
        # Use None if default MOSEK params are fine.
        mosek_params = None 

        self.beta = beta_cvx # Assign CVXPY variable

        if self._solve_cvxpy_problem(objective, constraints):
            if self.constraints_kernels_cvx_:
                try:
                    self.mu = np.array([const.dual_value[0] if const.dual_value is not None else 0.0
                                        for const in self.constraints_kernels_cvx_]).flatten()
                except Exception as e:
                    print(f"Could not retrieve dual values for epigraph formulation: {e}")
                    self.mu = np.zeros(len(self.constraints_kernels_cvx_))

            else:
                 self.mu = np.array([])
            self.t_epi_value = t_epi.value # Store optimal t
        else:
            self.mu = np.zeros(len(self.kernels))
            self.t_epi_value = None
        return self

# %%
if __name__ == "__main__":
    
    from sklearn.datasets import make_regression
    from sklearn.svm import SVR as SklearnSVR # Alias to avoid confusion
    from sklearn.metrics import mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    # from svr_sdp_multi_kernel_l1 import svr_sdp_multi_kernel_l1 # Assuming this file exists if uncommented

    # Create a regression dataset
    X_data, y_data = make_regression(
        n_samples=100,
        n_features=1, 
        noise=10, # Increased noise for a bit more challenge
        random_state=42 # Use a fixed random state for reproducibility
    )
    # y_data = y_data / np.std(y_data) # Optional: scale target

    kernel_params_list = [
        ("linear", {}),
        ("poly", {"degree": 2, "coef0": 1}), # Added coef0 for poly
        ("poly", {"degree": 3, "coef0": 1}),
        ("rbf", {"gamma": 0.01}), # Gamma values often need tuning
        ("rbf", {"gamma": 0.1}),
        ("rbf", {"gamma": 1.0}),
        ("rbf", {"gamma": 10.0}),
        ("sigmoid", {"gamma": 0.1, "coef0": 0}), # Sigmoid often needs coef0
    ]

    C_val = 10.0 # Regularization
    epsilon_val = 0.8 # Epsilon tube width
    tau_val = 1.0 # Tau for kernel constraints

    print("--- Testing SvrQcqpMultiKernelL1Mu ---")
    model_qcqp_multi = SvrQcqpMultiKernelL1Mu(
        C=C_val,
        epsilon=epsilon_val,
        tau=tau_val,
        kernel_params=kernel_params_list,
        verbose=False, # Set to True for detailed solver output
        kronecker_kernel=False, # Test with kronecker kernel
        trace_min=False # Test with trace_min disabled initially
    )
    
    model_qcqp_multi.fit(X_data, y_data)
    
    if model_qcqp_multi.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        y_pred_multi_qcqp = model_qcqp_multi.predict(X_data)
        print(f"QCQP Model Objective: {model_qcqp_multi.objective_value}")
        print(f"QCQP Model b: {model_qcqp_multi.b}")
        print(f"QCQP Model Support Vectors: {model_qcqp_multi.sup_num}")
        print(f"QCQP Model Mu (Duals for kernel constraints): \n{model_qcqp_multi.mu}")
        print(f"QCQP Model R2 score: {model_qcqp_multi.score(X_data, y_data)}")
        print(f"QCQP Model MAE: {mean_absolute_error(y_data, y_pred_multi_qcqp)}")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("QCQP Support Vectors")
        plt.scatter(X_data, y_data, label="Data", alpha=0.6)
        if model_qcqp_multi.sup_num > 0:
            plt.scatter(model_qcqp_multi.support_vectors, model_qcqp_multi.support_labels, 
                        color="red", marker="o", s=100, label="QCQP SVs", facecolors='none')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("QCQP Predictions")
        plt.scatter(X_data, y_data, label="Data", alpha=0.6)
        plt.scatter(X_data, y_pred_multi_qcqp, marker="+", label="QCQP Predictions", color="green")
        # Plot epsilon tube
        plt.plot(np.sort(X_data.flatten()), model_qcqp_multi.predict(np.sort(X_data, axis=0)) + model_qcqp_multi.epsilon, 'k--', lw=0.8, label='epsilon tube')
        plt.plot(np.sort(X_data.flatten()), model_qcqp_multi.predict(np.sort(X_data, axis=0)) - model_qcqp_multi.epsilon, 'k--', lw=0.8)

        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        print(f"QCQP Model optimization failed. Status: {model_qcqp_multi.status}")

    print("\n--- Testing SvrQcqpMultiKernelL1EpigraphTrace ---")
    model_epigraph = SvrQcqpMultiKernelL1EpigraphTrace(
        C=C_val,
        epsilon=epsilon_val,
        tau=tau_val, # Tau for epigraph term
        kernel_params=kernel_params_list,
        verbose=False,
        kronecker_kernel=True,
        trace_min=False
    )
    model_epigraph.fit(X_data, y_data)

    if model_epigraph.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        y_pred_epigraph = model_epigraph.predict(X_data)
        print(f"Epigraph Model Objective: {model_epigraph.objective_value}")
        print(f"Epigraph Model b: {model_epigraph.b}")
        print(f"Epigraph Model Support Vectors: {model_epigraph.sup_num}")
        print(f"Epigraph Model Mu (Duals for t_epi constraints): \n{model_epigraph.mu}")
        print(f"Epigraph Model t_epi value: {model_epigraph.t_epi_value}")
        print(f"Epigraph Model R2 score: {model_epigraph.score(X_data, y_data)}")
        print(f"Epigraph Model MAE: {mean_absolute_error(y_data, y_pred_epigraph)}")
        
        plt.figure(figsize=(10, 5))
        plt.title("Epigraph Model Predictions")
        plt.scatter(X_data, y_data, label="Data", alpha=0.6)
        plt.scatter(X_data, y_pred_epigraph, marker="x", label="Epigraph Predictions", color="purple")
        if model_epigraph.sup_num > 0:
            plt.scatter(model_epigraph.support_vectors, model_epigraph.support_labels, 
                        color="orange", marker="s", s=100, label="Epigraph SVs", facecolors='none')
        plt.legend()
        plt.show()
    else:
        print(f"Epigraph Model optimization failed. Status: {model_epigraph.status}")


    # Comparison with Scikit-learn SVR (using a single kernel for simplicity of comparison)
    print("\n--- Testing Scikit-learn SVR (RBF kernel) ---")
    # Find an RBF kernel from the list for a somewhat fair comparison
    rbf_params_for_sklearn = next((p['gamma'] for k, p in kernel_params_list if k == 'rbf'), 0.1)

    model_sklrn = SklearnSVR(
        C=C_val, 
        epsilon=epsilon_val,
        kernel="rbf", # Example: use RBF
        gamma=rbf_params_for_sklearn 
    )
    model_sklrn.fit(X_data, y_data.ravel()) # y needs to be 1D for sklearn SVR
    y_pred_sklrn = model_sklrn.predict(X_data)
    
    print(f"Sklearn SVR Intercept (b): {model_sklrn.intercept_[0]}")
    print(f"Sklearn SVR Support Vectors: {len(model_sklrn.support_)}")
    print(f"Sklearn SVR R2 score: {model_sklrn.score(X_data, y_data)}")
    print(f"Sklearn SVR MAE: {mean_absolute_error(y_data, y_pred_sklrn)}")

    plt.figure(figsize=(8, 6))
    plt.title("All Model Predictions")
    plt.scatter(X_data, y_data, label="True Data", alpha=0.5, s=30)
    if model_qcqp_multi.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        plt.plot(np.sort(X_data.flatten()), model_qcqp_multi.predict(np.sort(X_data, axis=0)), '--', label="QCQP Multi-Kernel", color="green")
    if model_epigraph.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        plt.plot(np.sort(X_data.flatten()), model_epigraph.predict(np.sort(X_data, axis=0)), ':', label="Epigraph Multi-Kernel", color="purple")
    plt.plot(np.sort(X_data.flatten()), model_sklrn.predict(np.sort(X_data, axis=0)), '-.', label=f"Sklearn SVR (RBF gamma={rbf_params_for_sklearn})", color="red")
    plt.legend()
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.show()

    # Example showing dual coefficients (beta for our model, dual_coef_ for sklearn)
    # Note: Sklearn's dual_coef_ are alpha_i - alpha_i^* and are only for SVs.
    # Our beta are defined for all samples.
    if model_qcqp_multi.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and model_qcqp_multi.beta is not None:
        plt.figure(figsize=(10,4))
        plt.stem(model_qcqp_multi.beta.flatten(), label="QCQP beta values", linefmt='grey', markerfmt='D')
        
        # For sklearn, dual_coef_ are only for support vectors. We need to map them back.
        dual_coef_full_sklearn = np.zeros_like(y_data.flatten())
        if hasattr(model_sklrn, 'support_') and model_sklrn.support_ is not None:
             dual_coef_full_sklearn[model_sklrn.support_] = model_sklrn.dual_coef_.flatten()
        plt.stem(dual_coef_full_sklearn, label="Sklearn dual_coef (on SVs)", linefmt='blue', markerfmt='o')
        plt.title("Dual Coefficients (Beta values)")
        plt.xlabel("Sample Index")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.show()


# %%
