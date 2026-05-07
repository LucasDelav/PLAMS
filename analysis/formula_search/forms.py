"""Form library for the formula search.

A "form" is a parametric expression f(d_1,...,d_k; theta) that is fit jointly
on (red, ox) targets sharing the same algebraic structure but with separate
coefficients.

Two flavors are supported:
  - polynomial subset: chosen monomials over k descriptors, linear in theta
    (closed-form least-squares fit)
  - Padé(num=1, den=1): rational form with k descriptors (nonlinear, curve_fit)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Monomial:
    """A monomial: tuple of exponents over the descriptor subset.

    e.g. Monomial((2, 1)) over descriptors (mu, eta) means mu^2 * eta."""
    exps: tuple[int, ...]

    @property
    def degree(self) -> int:
        return sum(self.exps)

    def label(self, var_names: tuple[str, ...]) -> str:
        if self.degree == 0:
            return "1"
        terms = []
        for v, e in zip(var_names, self.exps):
            if e == 0:
                continue
            terms.append(v if e == 1 else f"{v}^{e}")
        return "*".join(terms)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        out = np.ones(X.shape[0])
        for j, e in enumerate(self.exps):
            if e == 0:
                continue
            out = out * (X[:, j] ** e if e != 1 else X[:, j])
        return out


def all_monomials(k: int, max_total_degree: int) -> list[Monomial]:
    """All monomials in k variables with 0 < total degree <= max_total_degree, plus the constant."""
    out = [Monomial((0,) * k)]  # constant
    for d in range(1, max_total_degree + 1):
        for exps in product(range(d + 1), repeat=k):
            if sum(exps) == d:
                out.append(Monomial(exps))
    return out


def build_design(X: np.ndarray, monomials: Iterable[Monomial]) -> np.ndarray:
    cols = [m.evaluate(X) for m in monomials]
    return np.column_stack(cols) if cols else np.empty((X.shape[0], 0))


@dataclass(frozen=True)
class PolynomialForm:
    """Polynomial form: a chosen subset of monomials over a chosen subset of descriptors."""
    descriptor_idx: tuple[int, ...]
    monomials: tuple[Monomial, ...]

    @property
    def k(self) -> int:
        return len(self.descriptor_idx)

    @property
    def n_params(self) -> int:
        return len(self.monomials)

    def label(self, all_descriptor_names: list[str]) -> str:
        var_names = tuple(all_descriptor_names[i] for i in self.descriptor_idx)
        return " + ".join(m.label(var_names) for m in self.monomials)

    def design(self, X_full: np.ndarray) -> np.ndarray:
        X_sub = X_full[:, list(self.descriptor_idx)]
        return build_design(X_sub, self.monomials)


@dataclass(frozen=True)
class PadeForm:
    """Padé(1,1) form: y = (a0 + sum a_i d_i) / (1 + sum b_i d_i)."""
    descriptor_idx: tuple[int, ...]

    @property
    def k(self) -> int:
        return len(self.descriptor_idx)

    @property
    def n_params(self) -> int:
        return 1 + 2 * self.k

    def label(self, all_descriptor_names: list[str]) -> str:
        var_names = [all_descriptor_names[i] for i in self.descriptor_idx]
        num = " + ".join(["a0"] + [f"a_{v}*{v}" for v in var_names])
        den = " + ".join(["1"] + [f"b_{v}*{v}" for v in var_names])
        return f"({num}) / ({den})"

    def model_fn(self):
        k = self.k

        def fn(X_flat: np.ndarray, *theta: float) -> np.ndarray:
            X = X_flat.reshape(-1, k)
            a = np.asarray(theta[: 1 + k])
            b = np.asarray(theta[1 + k:])
            num = a[0] + X @ a[1:]
            den = 1.0 + X @ b
            return num / den

        return fn

    def x_data(self, X_full: np.ndarray) -> np.ndarray:
        X_sub = X_full[:, list(self.descriptor_idx)]
        return X_sub.reshape(-1)


def enumerate_polynomial_forms(
    n_descriptors: int,
    *,
    k_values: tuple[int, ...] = (1, 2, 3),
    max_total_degree: int = 3,
    full_only: bool = False,
    max_subset_count: int = 12,
) -> list[PolynomialForm]:
    """Enumerate (descriptor_subset, monomial_subset) forms.

    Args:
        n_descriptors: total number of candidate descriptors.
        k_values: which subset sizes to enumerate.
        max_total_degree: highest total polynomial degree per monomial.
        full_only: if True, only emit ONE form per (subset, degree) using all
            monomials up to that degree (no monomial-subset enumeration).
        max_subset_count: skip exhaustive monomial enumeration when the polynomial
            space has more than this many monomials (those forms get only the
            full polynomial and forward-stepwise variants from the search step).

    Returns: list of PolynomialForm.
    """
    seen: set[PolynomialForm] = set()
    for k in k_values:
        for desc_idx in combinations(range(n_descriptors), k):
            for deg in range(1, max_total_degree + 1):
                monos_full = all_monomials(k, deg)
                if full_only:
                    seen.add(PolynomialForm(desc_idx, tuple(monos_full)))
                    continue
                if len(monos_full) - 1 > max_subset_count:
                    # Too many monomials at this degree to enumerate every subset:
                    # emit only the full polynomial. (Forward stepwise can be run
                    # downstream to populate sparser variants for these forms.)
                    seen.add(PolynomialForm(desc_idx, tuple(monos_full)))
                    continue
                const = monos_full[0]
                others = monos_full[1:]
                for mask in range(1, 1 << len(others)):
                    chosen = [const] + [others[i] for i in range(len(others)) if mask >> i & 1]
                    seen.add(PolynomialForm(desc_idx, tuple(chosen)))
    return sorted(seen, key=lambda f: (f.k, f.n_params, f.descriptor_idx))


def enumerate_pade_forms(
    n_descriptors: int,
    *,
    k_values: tuple[int, ...] = (1, 2, 3),
) -> list[PadeForm]:
    forms: list[PadeForm] = []
    for k in k_values:
        for desc_idx in combinations(range(n_descriptors), k):
            forms.append(PadeForm(desc_idx))
    return forms
