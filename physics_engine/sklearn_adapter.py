"""scikit-learn compatible wrapper around PIR's discover_law, for SRBench.

discover_law(csv_path, target_var, ...) reads a CSV, treats every non-target
column as a candidate variable, and returns (best_expr, val_error, significant)
where best_expr is a sympy expression in terms of the CSV's column names.
This wrapper bridges SRBench's ndarray fit/predict interface to that.

Optional fallback gates (new in 3.6, OFF by default, so the default behaviour
is unchanged from 3.5). With gates="all" (or a tuple of names), after the
engine's fit the regressor tries, in order, stopping at the first accepted:

  "rational"  y = P(x)/Q(x), low-degree polynomials   -- accepted only if EXACT
  "exp"       y = C*exp(sum of monomials)              -- accepted only if EXACT
  "noisy"     y = C*prod x_j**(p/q), noise-tolerant     -- accepted only if its
              training MSE is <= the engine's training MSE

"EXACT" means max|y_hat - y| / max|y| < 1e-6 for the model string parsed back
from text. A gate model replaces the engine's model only when it passes; any
error inside a gate keeps the engine's model. `gate_info_` records what each
gate did. Thresholds are fixed a priori (see each gate module's docstring).
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
_GATE_ORDER = ("rational", "exp", "noisy")


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
        max_time=None,
        gates=None,
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
        self.max_time = max_time
        self.gates = gates

    def _gate_list(self):
        g = self.gates
        if g is None or g is False or g == ():
            return ()
        if g == "all" or g is True:
            return _GATE_ORDER
        g = (g,) if isinstance(g, str) else tuple(g)
        bad = [x for x in g if x not in _GATE_ORDER]
        if bad:
            raise ValueError(f"unknown gates {bad}; choose from {_GATE_ORDER} or 'all'")
        return tuple(x for x in _GATE_ORDER if x in g)  # always in the fixed order

    def fit(self, X, y):
        gates = self._gate_list()  # validate before the (slow) engine fit
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
        self.final_source_ = "engine"
        self.gate_info_ = {"gates": list(gates), "decision": "gates_off" if not gates else None}
        if gates:
            try:
                self._run_gates(X, y, gates)
            except Exception as e:  # a gate must never break the engine's result
                self.gate_info_["error"] = f"{type(e).__name__}: {e}"
            self.gate_info_["final_source"] = self.final_source_
        return self

    # ------------------------------------------------------------------ gates
    def _parse(self, expr_str):
        syms = {n: sp.Symbol(n) for n in self.feature_names_}
        expr = sp.sympify(expr_str, locals=syms)
        fn = sp.lambdify([syms[n] for n in self.feature_names_], expr, modules="numpy")
        return expr, fn

    @staticmethod
    def _eval(fn, Xv, n):
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(fn(*Xv.T), dtype=float), (n,))

    def _adopt(self, expr, fn, source):
        self.expr_, self._fn, self.final_source_ = expr, fn, source

    def _run_gates(self, X, y, gates):
        from physics_engine.discovery.rational_gate import EXACT_TOL

        info = self.gate_info_
        names = self.feature_names_
        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float).ravel()
        n = len(yv)
        scale = float(max(np.max(np.abs(yv)), 1e-300))
        base = self._eval(self._fn, Xv, n)
        base_err = float(np.max(np.abs(base - yv)) / scale)
        base_mse = float(np.mean((base - yv) ** 2))
        if not np.isfinite(base_err):
            base_err, base_mse = float("inf"), float("inf")
        info["engine_err_scaled"] = base_err
        if base_err < EXACT_TOL:
            info["decision"] = "engine_exact"
            return

        def exact_ok(res, tag):
            info[f"{tag}_expr"] = res["expr"]
            if not res.get("exact"):
                info[tag] = "not_exact"
                return False
            expr, fn = self._parse(res["expr"])
            err = float(np.max(np.abs(self._eval(fn, Xv, n) - yv)) / scale)
            info[f"{tag}_string_err_scaled"] = err
            if np.isfinite(err) and err < EXACT_TOL and err < base_err:
                self._adopt(expr, fn, tag)
                return True
            info[tag] = "rejected_not_better"
            return False

        if "rational" in gates:
            from physics_engine.discovery.rational_gate import rational_gate
            res = rational_gate(Xv, yv, names)
            if res is None:
                info["rational"] = "no_form"
            elif exact_ok(res, "rational"):
                info["decision"] = "rational_accepted"
                return

        if "exp" in gates:
            from physics_engine.discovery.exp_gate import exp_gate
            res = exp_gate(Xv, yv, names)
            if res is None:
                info["exp"] = "no_form"
            elif exact_ok(res, "exp"):
                info["decision"] = "exp_accepted"
                return

        if "noisy" in gates:
            from physics_engine.discovery.noisy_gate import noisy_power_gate
            res = noisy_power_gate(Xv, yv, names)
            if res is None:
                info["noisy"] = "no_form"
            else:
                info["noisy_expr"] = res["expr"]
                expr, fn = self._parse(res["expr"])
                mse = float(np.mean((self._eval(fn, Xv, n) - yv) ** 2))
                info["noisy_train_mse"], info["engine_train_mse"] = mse, base_mse
                if np.isfinite(mse) and (not np.isfinite(base_mse) or mse <= base_mse):
                    self._adopt(expr, fn, "noisy")
                    info["decision"] = "noisy_accepted"
                    return
                info["noisy"] = "rejected_not_better"
        info["decision"] = "engine_kept"

    # -------------------------------------------------------------- interface
    def model(self):
        return str(self.expr_)

    def predict(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        cols = [X[:, i] for i in range(X.shape[1])]
        out = np.asarray(self._fn(*cols), dtype=float)
        if out.ndim == 0:
            out = np.full(X.shape[0], float(out))
        return out
