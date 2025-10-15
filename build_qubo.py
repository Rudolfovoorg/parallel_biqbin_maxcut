#!/usr/bin/env python3
"""
build_qubo.py

Builds QUBOs for portfolio optimization from a configuration file.

Usage:
    python build_qubo.py -f config.json
or
    python build_qubo.py -f config.yaml

Config fields control whether a 'selection' or 'quantities' QUBO is built,
and specify all other parameters.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import yaml

from utils import qubo_to_biqbin_representation

TRADING_DAYS_PER_YEAR = 252

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# === Core computation functions (same as before) =====================

def compute_returns_and_cov(df_prices: pd.DataFrame, annualize: bool = True) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute daily simple returns, then return annualized mean return vector and covariance matrix.
    Annualization: mean returns * TRADING_DAYS_PER_YEAR; covariance * TRADING_DAYS_PER_YEAR.
    """
    # percent change (simple returns)
    daily_ret = df_prices.pct_change().dropna(how='all')
    if daily_ret.shape[0] == 0:
        raise RuntimeError("No returns could be computed (check price DataFrame).")

    mu_daily = daily_ret.mean()
    cov_daily = daily_ret.cov()

    if annualize:
        mu = mu_daily * TRADING_DAYS_PER_YEAR
        cov = cov_daily * TRADING_DAYS_PER_YEAR
    else:
        mu = mu_daily
        cov = cov_daily

    return mu.values, cov.values


def build_selection_qubo(mu: np.ndarray, cov: np.ndarray, q: float = 1.0,
                         cardinality_k: int = None, penalty: float = 10.0) -> np.ndarray:
    """
    Build Q matrix for selection problem.

    Minimize: F(x) = q * x^T cov x - mu^T x
    where x_i in {0,1}.

    If cardinality_K is given, add soft constraint penalty * (sum_i x_i - K)^2.

    Returns:
        Q: numpy array (N x N)
    """

    mu = mu.squeeze()
    N = len(mu)

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if N != cov.shape[0]:
        raise ValueError("Inconsistent dimensions between mu and covariance.")
    if not np.allclose(cov, cov.T, atol=1e-6):
        raise ValueError("Covariance matrix must be symmetric.")
    if cardinality_k is not None and cardinality_k > N:
        raise ValueError(f"Cardinality {cardinality_k} cannot be greater than {N}.")
    if q < 0:
        raise ValueError("Risk aversion q must be non-negative.")
    if penalty <= 0:
        raise ValueError("Penalty must be positive.")

    # Start with risk term
    Q = q * cov.copy()

    # We place mu on the diagonal (Q_ii)
    Q -= np.diag(mu)

    # Add cardinality penalty if required: P * (sum x_i - K)^2
    if cardinality_k is not None:
        P = penalty
        # Diagonal contribution -2*P*k
        Q += np.diag([-2*P*cardinality_k for _ in range(N)])

        # Off-diagonal contributions from P * x_i x_j.
        Q += np.ones([N,N]) * P

        # constant term P*K^2 omitted

    return Q


def build_quantities_qubo(mu: np.ndarray, cov: np.ndarray, prices: np.ndarray, 
                          budget: float, q: float = 1.0, penalty: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    
    """
    Build QUBO for integer-encoded holdings.
    Returns: Q matrix (M x M) and index_map (list of (asset_i, bit_k) for each bit index)
    """
    
    N = len(mu)

    if cov.shape[0] != cov.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if N != cov.shape[0]:
        raise ValueError("Inconsistent dimensions between mu, cov, and prices.")
    if not np.allclose(cov, cov.T, atol=1e-6):
        raise ValueError("Covariance matrix must be symmetric.")
    if len(prices) != N:
        raise ValueError("Prices vector length must match mu and covariance.")
    if np.any(prices <= 0):
        raise ValueError("All prices must be positive.")
    if budget <= 0:
        raise ValueError("Budget B must be positive.")
    if q < 0:
        raise ValueError("Risk aversion q must be non-negative.")
    if penalty <= 0:
        raise ValueError("Penalty must be positive.")
    
    # compute max number of units per asset given budget (floor)
    N_i = np.floor(budget / prices).astype(int)
    # bits per asset
    K_i = np.maximum(1, np.ceil(np.log2(N_i + 1)).astype(int))  # ensure at least 1 bit
    index_map = []
    for i in range(N):
        for k in range(K_i[i]):
            index_map.append((i, k))
    M = len(index_map)

    Q = np.zeros((M, M), dtype=float)

    
    #normalization factor:
    s_lin = 1 / budget          # scale for linear return terms
    s_quad = 1 / (budget * budget)   # scale for quadratic terms (risk and penalty)
    pen_lin_factor = -2 * penalty / budget


    # fill Q
    for p, (i, k) in enumerate(index_map):
        pow_k = 2 ** k
        # diagonal linear return term: - (P_i * mu_i * 2^k) / B
        Q[p, p] += - (prices[i] * mu[i] * pow_k) * s_lin

        # diagonal: risk self-term q * Sigma_ii * P_i^2 * 2^{2k} scaled by 1/B^2
        Q[p, p] += (q * cov[i, i] * (prices[i] ** 2) * (pow_k ** 2)) * s_quad

        # diagonal: penalty self-term lambda * P_i^2 * 2^{2k} scaled by 1/B^2
        Q[p, p] += (penalty * (prices[i] ** 2) * (pow_k ** 2)) * s_quad

        # diagonal: linear contribution from penalty -2*lam*B*P_i*2^k  -> after normalization becomes pen_lin_factor * P_i * 2^k
        Q[p, p] += pen_lin_factor * prices[i] * pow_k

    # off-diagonal: pairwise contributions
    for p in range(M):
        i, k = index_map[p]
        pow_k = 2 ** k
        for q_idx in range(p + 1, M):
            j, l = index_map[q_idx]
            pow_l = 2 ** l

            # risk contribution: q * Sigma_ij * P_i * P_j * 2^{k+l} scaled by 1/B^2
            risk_coeff = (q * cov[i, j] * prices[i] * prices[j] * (pow_k * pow_l)) * s_quad

            # penalty contribution: lam * P_i * P_j * 2^{k+l} scaled by 1/B^2
            pen_coeff = (penalty * prices[i] * prices[j] * (pow_k * pow_l)) * s_quad

            Q_val = risk_coeff + pen_coeff
            Q[p, q_idx] += Q_val
            Q[q_idx, p] += Q_val

    return Q, index_map


def save_qubo_json(qubo_dict: Dict[str, Any], filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(qubo_dict, f, indent=2)
    logging.info("Saved QUBO JSON to %s", filename)


# === Config-driven builder ==========================================

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")

    if path.suffix.lower() in {".json"}:
        with open(path, "r") as f:
            return json.load(f)
    elif path.suffix.lower() in {".yml", ".yaml"}:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format (use .json or .yaml)")


def build_qubo_from_config(cfg: Dict[str, Any]) -> None:
    """
    Main function to build QUBO using configuration dictionary.
    """

    # required fields
    qubo_type = cfg.get("type", "").lower()
    data_path = Path(cfg["data_path"])
    tickers = cfg["tickers"]
    output = cfg.get("output", "qubo.json")
    start_date = cfg.get("start_date")
    end_date = cfg.get("end_date")
    q = float(cfg.get("q", 1.0))
    penalty = float(cfg.get("penalty", 1.0))
    annualize = bool(cfg.get("annualize", True))
    scale = float(cfg.get("scale", 100))

    if not data_path.exists():
        raise FileNotFoundError(f"Data file {data_path} not found.")

    df = pd.read_excel(data_path).set_index("Date")
    if not all(t in df.columns for t in tickers):
        missing = [t for t in tickers if t not in df.columns]
        raise ValueError(f"Missing tickers in dataset: {missing}")

    if start_date:
        if start_date not in df.index:
            raise ValueError(f"start date '{start_date}' not found in dataset")
        df = df[start_date:]
    else:
        farthest = df.index.min()
        logging.info("No start date specified; using first available date %s", farthest)
        df = df[farthest:]

    if end_date:
        if end_date not in df.index:
            raise ValueError(f"end date '{end_date}' not found in dataset.")
        df_prices = df[:end_date][tickers]
        prices = df.loc[end_date][tickers].values
    else:
        latest = df.index.max()
        df_prices = df[:latest][tickers]
        prices = df.loc[latest][tickers].values
        logging.info("No end date specified; using last available date %s", latest)

    mu, sigma = compute_returns_and_cov(df_prices, annualize=annualize)

    if qubo_type == "selection":
        cardinality = int(cfg.get("cardinality", len(tickers)//2))
        Q = build_selection_qubo(mu, sigma, q=q, cardinality_k=cardinality, penalty=penalty)
        Q_scaled = np.round(Q * scale)
        qubo_dict = qubo_to_biqbin_representation(Q_scaled)
        qubo_dict.update({
            "tickers": tickers,
            "mu": mu.tolist(),
            "cov": sigma.tolist(),
            "prices": prices.tolist(),
            "q": q,
            "penalty": penalty,
            "cardinality": cardinality
        })

    elif qubo_type == "quantities":
        budget = float(cfg["budget"])
        Q, index_map = build_quantities_qubo(mu, sigma, prices, budget, q, penalty)
        Q_scaled = np.round(Q * scale)
        qubo_dict = qubo_to_biqbin_representation(Q_scaled)
        qubo_dict.update({
            "tickers": tickers,
            "mu": mu.tolist(),
            "cov": sigma.tolist(),
            "prices": prices.tolist(),
            "budget": budget,
            "index_map": index_map,
            "q": q,
            "penalty": penalty
        })

    else:
        raise ValueError("Config 'type' must be either 'selection' or 'quantities'.")

    save_qubo_json(qubo_dict, output)


# === Entry point ====================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build QUBOs for portfolio optimization from a config file."
    )
    parser.add_argument("config", help="Path to config JSON/YAML file.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    build_qubo_from_config(cfg)


if __name__ == "__main__":
    main()
