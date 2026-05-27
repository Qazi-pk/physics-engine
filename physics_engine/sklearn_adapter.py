"""scikit-learn compatible wrapper around PIR's discover_law, for SRBench.

discover_law(csv_path, target_var, ...) reads a CSV, treats every non-target
column as a candidate variable, and returns (best_expr, val_error, significant)
where best_expr is a sympy expression in terms of the CSV's column names.
This wrapper bridges SRBench's ndarray fit/predict interface to that.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.base import BaseEstimator, RegressorMixin

from physics_engine.discovery.symbolic_search import discover_law

# Internal target column name, unlikely to collide with real feature names.
_TARGET_COL = "pir_target__"


class PIRRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        enforce_dimensions=True,
        use_ransac=True,
        use_residual=True,
        use_sparse=True,
        use_ot_loss=False,
        alpha=0.7,
        beta=0.3,
        allowed_powers=None,
        unary_functions=None,
        include_pairwise_products=True,
        add_physics_features=False,
        max_basis_terms=5,
        lambda_penalty=0.01,
        random_state=42,
        max_train_rows=800,
    ):
    
        self.enforce_dimensions = enforce_dimensions
        self.use_ransac = use_ransac
        self.use_residual = use_residual
        self.use_sparse = use_sparse
        self.use_ot_loss = use_ot_loss
        self.alpha = alpha
        self.beta = beta
        self.allowed_powers = allowed_powers
        self.unary_functions = unary_functions
        self.include_pairwise_products = include_pairwise_products
        self.add_physics_features = add_physics_features
        self.max_basis_terms = max_basis_terms
        self.lambda_penalty = lambda_penalty
        self.random_state = random_state
        self.max_train_rows = max_train_rows

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_df = X.copy()
        else:
            X = np.asarray(X)
            self.feature_names_ = [f"x_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=self.feature_names_)

        X_df[_TARGET_COL] = np.asarray(y, dtype=float)

        if self.max_train_rows and len(X_df) > self.max_train_rows:
            X_df = X_df.sample(
                n=self.max_train_rows,
                random_state=self.random_state,
            ).reset_index(drop=True)

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        )
        try:
            X_df.to_csv(tmp.name, index=False)
            tmp.close()
            result = discover_law(
                tmp.name,
                _TARGET_COL,
                enforce_dimensions=self.enforce_dimensions,
                use_ransac=self.use_ransac,
                use_residual=self.use_residual,
                use_sparse=self.use_sparse,
                use_ot_loss=self.use_ot_loss,
                alpha=self.alpha,
                beta=self.beta,
                allowed_powers=self.allowed_powers,
                unary_functions=self.unary_functions,
                include_pairwise_products=self.include_pairwise_products,
                add_physics_features=self.add_physics_features,
                max_basis_terms=self.max_basis_terms,
                lambda_penalty=self.lambda_penalty,
                random_state=self.random_state,
                return_metadata=False,
            )
        finally:
            os.unlink(tmp.name)

        best_expr = result[0] if isinstance(result, tuple) else result
        if best_expr is None:
            best_expr = sp.Integer(0)

        self.expr_ = sp.sympify(best_expr)
        self.symbols_ = [sp.Symbol(n) for n in self.feature_names_]
        self._fn = sp.lambdify(self.symbols_, self.expr_, modules="numpy")
        return self

    def model(self):
        return str(self.expr_)

    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        cols = [X[:, i] for i in range(X.shape[1])]
        out = np.asarray(self._fn(*cols), dtype=float)
        if out.ndim == 0:
            out = np.full(X.shape[0], float(out))
        return out