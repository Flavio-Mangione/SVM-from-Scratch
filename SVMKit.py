import numpy as np
from cvxopt import matrix, solvers
from joblib import Parallel, delayed
from typing import Optional, Literal, Tuple, List, Union
from time import time

class SVM:
    """
    Support Vector Machine classifier using the dual formulation with kernel support.
    
    Supports Gaussian (RBF) and Polynomial kernels, and binary or multiclass classification 
    via One-vs-All (OvA) or One-vs-One (OvO) strategies.

    Parameters:
    -----------
    kernel : {'gaussian', 'polynomial'} 
        Type of kernel to use for the SVM.
    C : float
        Regularization parameter.
    gamma : float
        Kernel coefficient for RBF.
    degree : int
        Degree for polynomial kernel.
    solver : {'cvxopt', 'mvp'}
        Solver for dual optimization.
    decision_function_shape : {'ova', 'ovo', None}
        Strategy for multiclass classification.
    """
    
    def __init__(self, kernel: Optional[Literal["gaussian", "polynomial"]] ='gaussian', C:float=1.0, gamma:float=1.0, 
                 degree:int=2, solver: Optional[Literal["cvxopt", "mvp"]] = 'cvxopt', decision_function_shape: Optional[Literal["ova", "ovo"]] = None,
                 tol: Optional[float] = 1e-5):
        
        self.kernel_name = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.solver = solver
        self.decision_function_shape = decision_function_shape  
        self.tolerance = tol
        
        # Model parameters to be learned
        self.alphas = None
        self.bias = None
        self.X_train = None
        self.y_train = None
        self.classes = None
        self.K_train = None
        
        # Training metrics
        self.n_iter = None 
        self.duality_gap = None
        self.CPU_time = None
    # -----------------------------------------------------
    # Kernel Methods
    
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        # Compute kernel matrix between X1 and X2 
        # For Gaussian kernel: K(x, x') = exp(-γ ||x - x'||²)
        # Efficiently computed using the identity: ||x - x'||² = ||x||² + ||x'||² - 2xᵀx′
        if self.kernel_name == 'gaussian':
            X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            return np.exp(-self.gamma * (X1_sq + X2_sq - 2 * np.dot(X1, X2.T)))
        
        elif self.kernel_name == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** self.degree
        
        else:
            raise ValueError("Unsupported kernel type")

    # -----------------------------------------------------
    # Training Methods
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Fit the SVM model to the training data
        self.classes = np.unique(y)
        
        # Multiclass One-vs-All strategy
        if self.decision_function_shape == "ova" and len(self.classes) > 2:
            start = time()
            self._fit_ova(X, y, self.classes)
            self.CPU_time = time() - start
            
            self.duality_gap = np.mean([model.duality_gap for _, model in self.models])
            self.n_iter = np.mean([model.n_iter for _, model in self.models])
            return
        
        # Multiclass One-vs-One strategy
        elif self.decision_function_shape == "ovo":
            start = time()
            self._fit_ovo(X, y, self.classes)
            self.CPU_time = time() - start

            self.duality_gap = np.mean([model.duality_gap for _, model in self.models])
            self.n_iter = np.mean([model.n_iter for _, model in self.models])
            return
        else:
            if self.decision_function_shape is not None:
                raise ValueError("Invalid decision_function_shape. Use 'ova' or 'ovo' for multiclass classification.")
        
        # Binary classification case
        self.X_train = X
        self.y_train = y
        self.K_train = self._kernel(X, X)
        
        if self.solver == 'cvxopt':
            start = time()
            self._fit_cvxopt()
            self.CPU_time = time() - start
        elif self.solver == 'mvp':
            start = time()
            self._fit_mvp()
            self.CPU_time = time() - start 
        else:
            raise ValueError("Unsupported solver. Use 'cvxopt' or 'mvp'.")
    # -----------------------------------------------------
    # Define each fit method

    def _train_one_vs_all(self, X, y, label):
            # Helper method for One-vs-All training
            y_bin = np.where(y == label, 1, -1)
            svm = SVM(kernel=self.kernel_name, C=self.C, gamma=self.gamma,
                      degree=self.degree, solver=self.solver, decision_function_shape=None,
                      tol=self.tolerance)
            svm.fit(X, y_bin)
            return label, svm
        
    def _fit_ova(self, X: np.ndarray, y: np.ndarray, labels: np.ndarray):
            # One-vs-All multiclass training strategy in parallel 
            self.models = Parallel(n_jobs=-1)(delayed(self._train_one_vs_all)(X, y, label) for label in labels)

    def _fit_ovo(self, X: np.ndarray, y: np.ndarray, labels):

        # One-vs-One multiclass training strategy
        self.models = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                y_i = labels[i]
                y_j = labels[j]
                # Only filter indexes for class i and j
                idx = np.where((y == y_i) | (y == y_j))[0]
                X_ij = X[idx]
                y_ij = y[idx]
                y_ij = np.where(y_ij == y_i, 1, -1)

                svm = SVM(kernel=self.kernel_name, C=self.C, gamma=self.gamma,
                      degree=self.degree, solver=self.solver, decision_function_shape=None,
                      tol=self.tolerance)
                
                svm.fit(X_ij, y_ij)
                self.models.append(((y_i, y_j), svm)) 
 
    def _fit_cvxopt(self):
        """
        Solve the dual SVM problem using the CVXOPT quadratic programming solver.
        
        Formulates the dual optimization problem as:
            minimize: (1/2) αᵀ P α + qᵀ α
            subject to:
                G α ≤ h   (inequality constraints: 0 ≤ αᵢ ≤ C);
                A α = b   (equality constraint: ∑αᵢ yᵢ = 0).
        
        This is a convex quadratic optimization problem, guaranteeing a global optimum.
        """
        # Solve the dual SVM problem using CVXOPT QP solver
        n_samples = len(self.y_train)
        y= self.y_train
    
        # Quadratic term: P = (yyT)K   
        epsilon = 1e-3 # add noise for numerical stability 
        P = matrix(np.outer(y, y) * self.K_train)
        
        # Linear term: q = -1
        q = matrix(-np.ones(n_samples))
        
        # Inequality: 0 ≤ α ≤ C
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))  # Stack -I and I
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))  
        
        # Equality: ∑αᵢyᵢ = 0
        A = matrix(y.reshape(1, -1).astype(np.float64))  # yT 
        b = matrix(0.0)

        # Suppress solver output
        solvers.options['show_progress'] = False

        # Solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)
        self.n_iter = solution.get('iterations', None)
        self.alphas = np.ravel(solution['x'])
        self._compute_bias(self.K_train, self.y_train, self.alphas)
        self._compute_duality_gap(self.K_train, self.y_train, self.alphas, self.bias)

    # Train using the Most Violating Pair algorithm with fixed tolerance and max iterations
    def _fit_mvp(self):
        """
        Train the SVM using the Most Violating Pair (MVP) algorithm.

        Iteratively selects and updates the pair of Lagrange multipliers (αᵢ, αⱼ) 
        that most violates the KKT conditions.
        The procedure continues until no more violating pairs are found or the maximum 
        number of iterations is reached.
        """

        self.alphas = self.most_violating_pair_solver(tol = self.tolerance, max_iter = 1000)
        self._compute_bias(self.K_train, self.y_train, self.alphas)
        self._compute_duality_gap(self.K_train, self.y_train, self.alphas, self.bias)
        
    # -----------------------------------------------------
    # Define The Most Violating Pair Algorithm 

    @staticmethod
    def _compute_kkt_violation(i: int, gradient: np.ndarray, alpha: np.ndarray, 
                              y: np.ndarray, C: float, tol: float) -> float:
        """Compute KKT violation for sample i"""

        E_i = gradient[i] * y[i]  # KKT violation component for i
        
        # Depending on αᵢ , determine whether it violates lower, upper or margin bounds
        if alpha[i] <= tol and E_i < -tol:
            return abs(E_i + tol)
        elif alpha[i] >= C - tol and E_i > tol:
            return abs(E_i - tol)
        elif tol < alpha[i] < C - tol and abs(E_i) > tol:
            return abs(E_i) - tol
        return 0

    @staticmethod
    def _select_first_variable(gradient: np.ndarray, alpha: np.ndarray, y: np.ndarray, 
                              C: float, tol: float) -> int:
        """Select i with largest KKT violation (most promising to update)"""

        max_violation, best_i = 0, -1
        
        for i in range(len(y)):
            violation = SVM._compute_kkt_violation(i, gradient, alpha, y, C, tol)
            if violation > max_violation:
                max_violation, best_i = violation, i
        
        return best_i if max_violation > tol else -1

    @staticmethod
    def _select_second_variable(best_i: int, gradient: np.ndarray, alpha: np.ndarray, 
                               y: np.ndarray, K: np.ndarray, C: float) -> int:
        """Find j that maximizes |gradient[i] - gradient[j]|"""
        
        best_j, max_progress = -1, 0
        candidates = np.where((alpha > 0) & (alpha < C))[0]
        if candidates.size == 0:
            candidates = np.arange(len(y))

        for j in candidates:
            if j == best_i:
                continue
            
            # Compute box constraints L and H depending on labels
            if y[best_i] != y[j]:
                L, H = max(0, alpha[j] - alpha[best_i]), min(C, C + alpha[j] - alpha[best_i])
            else:
                L, H = max(0, alpha[best_i] + alpha[j] - C), min(C, alpha[best_i] + alpha[j])
            
            # Compute η = K_ii + K_jj - 2*K_ij (second derivative in QP problem)
            eta = K[best_i, best_i] + K[j, j] - 2 * K[best_i, j]
            progress = abs(gradient[best_i] - gradient[j])
            
            # Select j only if feasible (L < H, eta > 0)
            if L < H and eta > 0 and progress > max_progress:
                max_progress, best_j = progress, j
        
        return best_j

    @staticmethod
    def _select_working_set(gradient: np.ndarray, alpha: np.ndarray, y: np.ndarray, 
                            K: np.ndarray, C: float, tol: float) -> Tuple[int, int]:
        """Selects the most violating pair (i, j) for optimization"""

        # Step 1: Select i with largest KKT violation (most promising to update)
        best_i = SVM._select_first_variable(gradient, alpha, y, C, tol)
        if best_i == -1:
            return -1, -1
             
        # Step 2: Find j that maximizes ∣∇i−∇j∣
        best_j = SVM._select_second_variable(best_i, gradient, alpha, y, K, C)
        
        return best_i, best_j if best_j != -1 else -1

    def most_violating_pair_solver(self, tol: float = 1e-3, max_iter: int = 1000) -> np.ndarray:
        """
        Most Violating Pair (MVP) algorithm for SVM.
        
        Iteratively selects the pair of variables that most violates 
        the KKT conditions to maximize progress toward optimality.
        """
        n = len(self.y_train)
        np.random.seed(42) # Set a seed for the repoducibility
        
        # Initialize alpha with small random values to avoid starting from a flat gradient
        # This helps break symmetry and improves early convergence, especially when alpha is zero
        alpha = np.random.uniform(0, min(tol, self.C/10), n)
        
        # Precompute initial gradient
        gradient = self.K_train.dot(alpha * self.y_train) - self.y_train
        
        iterations = 0
        while iterations < max_iter:
            # Select the working set (i, j)
            i, j = self._select_working_set(gradient, alpha, self.y_train, self.K_train, self.C, tol)

            # If no suitable pair is found, optimization is complete
            if i == -1 or j == -1:
                break
                
            # Optimize the pair
            elif not self._optimize_pair(i, j, alpha, gradient, self.y_train, self.K_train, self.C, tol):
                # If optimization fails to make progress, interrupt the iterations
                break
            
            iterations += 1
        
        self.n_iter = iterations
        return alpha

    @staticmethod
    def _optimize_pair(i: int, j: int, alpha: np.ndarray, gradient: np.ndarray,
                    y: np.ndarray,K: np.ndarray, C: float, tol: float) -> bool:
        """
        Analytically optimize the pair (αᵢ, αⱼ) while maintaining constraints
        """
        
        # Compute bounds (L, H) for αⱼ based on box constraints and equality constraint
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - C)
            H = min(C, alpha[i] + alpha[j])
        
        if L >= H:
            return False
        
        # Calculate the curvature
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        if eta <= tol:
            return False
        
        # Calculate new αⱼ
        alpha_j_new = alpha[j] + y[j] * (gradient[i] - gradient[j]) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)
        
        # Check for significant change
        if np.isclose(alpha_j_new, alpha[j], rtol=1e-12, atol=1e-12):
            return False
        
        # Calculate new αᵢ (constraint: yᵢαᵢ + yⱼαⱼ = constant)
        alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)
        
        # Incremental gradient update
        delta_i = alpha_i_new - alpha[i]
        delta_j = alpha_j_new - alpha[j]
        gradient += delta_i * y[i] * K[:, i] + delta_j * y[j] * K[:, j]
        
        # Update αᵢ and αj
        alpha[i], alpha[j] = alpha_i_new, alpha_j_new
        
        return True
    # -----------------------------------------------------
    # Bias, Primal and Dual Problem

    def _compute_bias(self, K: np.ndarray, y: np.ndarray, alphas: np.ndarray):
        
        # Compute the bias term using support vectors
            
        support = (alphas > 0) & (alphas < self.C)
        if not np.any(support):
            self.bias = 0
            return

        b_values = y[support] - K[support].dot(alphas * y)
        self.bias = np.mean(b_values)

    def _dual_objective(self, alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:

        # Compute the dual objective value
        return alpha.sum() - 0.5 * (alpha * y).dot(K.dot(alpha * y))
    
    def _primal_objective(self, alpha: np.ndarray, y: np.ndarray, K: np.ndarray, bias: float) -> float:
        # Decision function f(xᵢ) for all i
        decision = K.dot(alpha * y) + bias
        
        # Hinge loss: max(0, 1 - yᵢ f(xᵢ))
        hinge_losses = np.maximum(0, 1 - y * decision)
        
        # Regularization term: (1/2) * ||w||^2
        regularizer = 0.5 * np.sum(np.outer(alpha * y, alpha * y) * K)
        
        return regularizer + self.C * np.sum(hinge_losses)
    
    def _compute_duality_gap(self, K: np.ndarray, y: np.ndarray, alpha: np.ndarray, bias: float) -> float:
   
        primal_value = self._primal_objective(alpha, y, K, bias)
        dual_value = self._dual_objective(alpha, y, K)
        
        self.duality_gap = primal_value - dual_value
    
    # -----------------------------------------------------
    # Prediction Methods
    
    def _decision_function_ova(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores for One-vs-All strategy"""
        n_sample = X.shape[0]
        scores = np.zeros((n_sample, len(self.models)))
        for idx, (_, model) in enumerate(self.models):
            scores[:, idx] = model.decision_function(X)
        return scores
    
    def _decision_function_ovo(self, X: np.ndarray) -> np.ndarray:
        """Compute voting scores for One-vs-One strategy"""
        n_sample = X.shape[0]
        votes = np.zeros((n_sample, len(self.classes)))

        for (y_i, y_j), model in self.models:
            decisions = model.decision_function(X)
            predictions = np.sign(decisions)
            idx_yi = np.where(predictions > 0)[0]
            idx_yj = np.where(predictions <= 0)[0]

            votes[idx_yi, np.where(self.classes == y_i)[0][0]] += 1
            votes[idx_yj, np.where(self.classes == y_j)[0][0]] += 1
                    
        return votes

    def _decision_function_binary(self, X: np.ndarray) -> np.ndarray:
        """Compute decision scores for binary classification"""
        K = self._kernel(X, self.X_train)
        return np.dot(K, self.alphas * self.y_train) + self.bias

    def decision_function(self, X: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute decision scores for the samples"""
        if self.decision_function_shape == "ova":
            return self._decision_function_ova(X)
        elif self.decision_function_shape == "ovo":
            return self._decision_function_ovo(X)
        else:
            return self._decision_function_binary(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        
        # Predict class labels for samples in X
        if self.decision_function_shape == "ova":
            scores = self.decision_function(X)
            labels = [label for label, _ in self.models]
            max_indices = np.argmax(scores, axis=1)
            return np.array([labels[i] for i in max_indices])
        
        elif self.decision_function_shape == "ovo":
            votes = self.decision_function(X)
            max_indices = np.argmax(votes, axis=1)
            return self.classes[max_indices]
        
        else:
            return np.sign(self.decision_function(X))

    # -----------------------------------------------------
    # Utility Methods
    
    def score(self,y_pred: np.ndarray, y: np.ndarray) -> float:
        return np.mean(y_pred == y)

    def get_support_vectors(self) -> np.ndarray:
        support_indices = np.where(self.alphas > self.tolerance)[0]
        return self.X_train[support_indices]

    def report_metrics(self):

        """Binary case"""
        alpha0 = np.zeros_like(self.y_train)
        if self.decision_function_shape is None:  
            initial_obj = self._dual_objective(alpha0, self.y_train, self.K_train)
            final_obj = self._dual_objective(self.alphas, self.y_train, self.K_train)

            print(f"Dual objective (initial): {initial_obj:.2f}")
            print(f"Dual objective (final): {final_obj:.2f}")
            print(f"Number of Iterations: {self.n_iter}")
            print(f"Bias: {self.bias:.2f}")
            print(f"Number of Support Vectors: {len(self.get_support_vectors())}")
            print(f"Max alpha value: {round(np.max(self.alphas), 3)}")
            print(f"Min non-zero alpha: {round(np.min(self.alphas[self.alphas > 0]), 3)}")
            print(f"CPU Time: {self.CPU_time:.2f} seconds")
        else:   
            """Multiclass case"""
            initial_obj = np.mean([model._dual_objective(alpha0, model.y_train, model.K_train)for _, model in self.models])
            self.dual_objective = np.mean([model._dual_objective(model.alphas, model.y_train, model.K_train)for _, model in self.models])
            self.bias = np.mean([model.bias for _, model in self.models])
            self.support_vector = np.mean([len(model.get_support_vectors()) for _, model in self.models])

            all_alphas = np.concatenate([model.alphas for _, model in self.models])
            
            print(f"Dual objective (initial): {initial_obj:.2f}")
            print(f"Dual objective (final): {self.dual_objective:.2f}")
            print(f"Number of Iterations: {self.n_iter:.0f}")
            print(f"Bias: {self.bias:.2f}")
            print(f"Number of Support Vectors: {self.support_vector:.0f}")
            print(f"Max alpha value: {round(np.max(all_alphas),3)}")
            print(f"Min non-zero alpha: {round(np.min(all_alphas[all_alphas > 0]),3)}")
            print(f"CPU Time: {self.CPU_time:.2f} seconds")
