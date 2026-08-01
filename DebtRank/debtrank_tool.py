#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================================================
# Copyright (c) 2026 Jorge A. Chan-Lau.
# SPDX-License-Identifier: CC-BY-NC-ND-4.0
#
# The DebtRank Systemic Risk Tool -- including its HTML and Python
# implementations -- and its accompanying documentation are licensed under
# the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
# International License (CC BY-NC-ND 4.0):
# https://creativecommons.org/licenses/by-nc-nd/4.0/
#
# Disclaimer: This analytical calculator is provided "as is" solely for
# educational and research purposes. It does not constitute financial,
# investment, legal, regulatory, or other professional advice. The author
# makes no warranties, express or implied, regarding its accuracy,
# completeness, reliability, or fitness for any particular purpose. To the
# fullest extent permitted by applicable law, the author shall not be liable
# for any direct, indirect, incidental, special, consequential, or other
# damages or financial losses arising from its use or reliance on its outputs.
# Users are responsible for independently validating all calculations and
# results.
#
# The views and interpretations reflected in this tool are solely those of the
# author. Use of the tool in a course, workshop, training program, or other
# institutional setting does not imply endorsement by, affiliation with, or
# responsibility on the part of any employer, host, sponsor, course provider,
# or other organization.
# ==========================================================================
"""
DebtRank Systemic Risk Tool (Python version)
============================================
Replicates the standalone HTML tool (DebtRank_Systemic_Risk_Tool_v03.html).

Methodology:
  * Battiston, Puliga, Kaushik, Tasca & Caldarelli (2012), "DebtRank: Too
    Central to Fail?", Sci. Rep. 2:541  -> original DebtRank (propagate once)
  * Bardoscia, Battiston, Caccioli & Caldarelli (2015), "DebtRank: A
    Microscopic Foundation for Shock Propagation", PLoS ONE 10(6)
    -> linear DebtRank, leverage-matrix eigenvalue stability
  * Battiston, Caldarelli, D'Errico & Gurciullo (2016), "Leveraging the
    network", SSRN 2571218 -> default-based impact vs scenario vulnerability,
    1st/2nd/3rd round decomposition, loss distributions, VaR/CVaR.  The
    optional fire-sale calculation here is a simplified heuristic inspired by
    that paper, not its target-leverage restoration equation.

Input (auto-detected, same CSVs as the HTML tool):
  1. Bilateral-exposure matrix: first column = bank name; then one column per
     bank with A_ij (lender i row, borrower j column); plus columns
     'external_assets' and 'equity'.
  2. Aggregate balance sheets: columns bank, interbank_lending,
     interbank_borrowing, external_assets, equity. The bilateral network is
     then reconstructed as one deterministic sparse directed-fitness topology
     fitted with RAS.  This is a reproducible approximation inspired by
     Bardoscia et al. (2015), not the paper's stochastic network ensemble.

Usage:
  python debtrank_tool.py DebtRank_example_network.csv
  python debtrank_tool.py data.csv --algo original --shock external --alpha 1.0
  python debtrank_tool.py data.csv --shock default --bank "Citigroup"
  python debtrank_tool.py data.csv --fire-sales --phi 0.3
  python debtrank_tool.py data.csv --mc 2000 --mc-mean 1.0 --var-level 0.95
  python debtrank_tool.py data.csv --layout spring     # kk|spring|circular|shell|spiral
  python debtrank_tool.py data.csv --max-edges 80      # draw 80 largest exposures (0 = all)
  python debtrank_tool.py data.csv --no-plots          # console report only

Outputs: console report + (unless --no-plots) a multi-panel figure
'debtrank_report.png' replicating the HTML tool's charts.

Dependencies: numpy, matplotlib (pip install numpy matplotlib)
"""

import argparse
import csv
import sys

import numpy as np


def select_edges(edges, max_edges=90):
    """Choose which exposures to draw in the network diagram: the `max_edges`
    largest exposures (max_edges = 0 draws all of them). `edges` must be a list
    of (lender, borrower, weight) tuples ALREADY sorted by descending weight.

    This limit is purely cosmetic (keeping the diagram legible) and never
    affects the DebtRank or contagion results.
    """
    if max_edges and max_edges > 0:
        return edges[:max_edges]
    return edges

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

class Network:
    """Interbank network: names, A (NxN exposures, lender i -> borrower j),
    AE (external assets), E (equity)."""

    def __init__(self, names, A, AE, E, reconstructed=False,
                 reconstruction_info=None):
        self.names = list(names)
        self.A = np.asarray(A, dtype=float)
        self.AE = np.asarray(AE, dtype=float)
        self.E = np.asarray(E, dtype=float)
        self.reconstructed = reconstructed
        self.reconstruction_info = reconstruction_info or {}
        if np.any(self.E <= 0):
            raise ValueError("Equity values must be positive.")

    @property
    def n(self):
        return len(self.E)

    @property
    def ib_lending(self):
        return self.A.sum(axis=1)

    @property
    def ib_borrowing(self):
        return self.A.sum(axis=0)

    @property
    def leverage(self):
        """Interbank leverage matrix Lambda_ij = A_ij / E_i (Bardoscia 2015)."""
        return self.A / self.E[:, None]

    @property
    def spectral_radius(self):
        """|lambda_max| of the leverage matrix: <1 stable, >1 amplifying."""
        return float(np.abs(np.linalg.eigvals(self.leverage)).max())


def load_network(path):
    """Auto-detect CSV layout (bilateral matrix vs aggregate balance sheets)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
    header = [c.strip() for c in rows[0]]
    body = rows[1:]
    names = [r[0].strip() for r in body]
    n = len(names)
    low = [c.lower() for c in header]

    def col(name):
        return low.index(name) if name in low else -1

    c_e, c_ae = col("equity"), col("external_assets")
    if c_e < 0:
        raise ValueError("No 'equity' column found.")
    E = np.array([float(r[c_e]) for r in body])

    is_matrix = len(header) >= n + 1 and all(
        header[j + 1] == names[j] for j in range(n))

    if is_matrix:
        if c_ae < 0:
            raise ValueError("Matrix layout needs 'external_assets'.")
        A = np.array([[max(0.0, float(r[j + 1])) for j in range(n)]
                      for r in body])
        np.fill_diagonal(A, 0.0)
        AE = np.array([float(r[c_ae]) for r in body])
        return Network(names, A, AE, E, reconstructed=False)

    c_l, c_b = col("interbank_lending"), col("interbank_borrowing")
    if min(c_l, c_b, c_ae) < 0:
        raise ValueError("Aggregate layout needs columns: interbank_lending, "
                         "interbank_borrowing, external_assets, equity.")
    lend = np.array([float(r[c_l]) for r in body])
    borr = np.array([float(r[c_b]) for r in body])
    AE = np.array([float(r[c_ae]) for r in body])
    A, reconstruction_info = reconstruct(lend, borr, return_info=True)
    return Network(names, A, AE, E, reconstructed=True,
                   reconstruction_info=reconstruction_info)


def reconstruct(lend, borrow, base_density=0.05, min_degree=3,
                iters=10000, tol=1e-9, return_info=False):
    """Reconstruct one deterministic bilateral matrix from aggregate totals.

    The support is a sparse binary topology selected from Bardoscia-style
    directed fitness probabilities

        p_ij = z x_out_i x_in_j / (1 + z x_out_i x_in_j),

    using separate lending and borrowing fitnesses.  The base target density
    is 5%; for small systems a three-link average-degree floor is used.  The
    support is deterministically densified only when needed for RAS to match
    the observed margins.  As in Bardoscia et al. (2015), total borrowing is
    rescaled when necessary so that it equals total lending.

    This deliberately returns one reproducible network rather than the paper's
    ensemble of random networks.  Set ``return_info`` to receive reconstruction
    diagnostics together with the matrix.
    """
    lend = np.asarray(lend, dtype=float)
    borrow = np.asarray(borrow, dtype=float)
    if lend.ndim != 1 or borrow.ndim != 1 or len(lend) != len(borrow):
        raise ValueError("Lending and borrowing totals must be equal-length vectors.")
    if np.any(~np.isfinite(lend)) or np.any(~np.isfinite(borrow)):
        raise ValueError("Interbank lending and borrowing must be finite numbers.")
    if np.any(lend < 0) or np.any(borrow < 0):
        raise ValueError("Interbank lending and borrowing must be non-negative.")

    n = len(lend)
    total_lend = float(lend.sum())
    total_borrow = float(borrow.sum())
    if total_lend == 0 and total_borrow == 0:
        A = np.zeros((n, n), dtype=float)
        info = {"borrow_scale": 1.0, "base_density": base_density,
                "target_density": 0.0, "initial_links": 0, "links": 0,
                "density": 0.0, "ras_iterations": 0,
                "margin_error": 0.0}
        return (A, info) if return_info else A
    if total_lend <= 0 or total_borrow <= 0:
        raise ValueError("Total interbank lending and borrowing must both be positive.")

    borrow_scale = total_lend / total_borrow
    borrow_adj = borrow * borrow_scale
    total = total_lend
    feasibility_tol = tol * max(1.0, total)
    for i in range(n):
        if lend[i] > total - borrow_adj[i] + feasibility_tol:
            raise ValueError(
                f"No zero-diagonal network can allocate lending for bank {i}: "
                "its lending exceeds all other banks' adjusted borrowing.")
        if borrow_adj[i] > total - lend[i] + feasibility_tol:
            raise ValueError(
                f"No zero-diagonal network can allocate borrowing for bank {i}: "
                "its adjusted borrowing exceeds all other banks' lending.")

    x_out = lend / total
    x_in = borrow_adj / total
    candidates = [(i, j, float(x_out[i] * x_in[j]))
                  for i in range(n) for j in range(n)
                  if i != j and lend[i] > 0 and borrow_adj[j] > 0]
    if not candidates:
        raise ValueError("No feasible lender-borrower pairs remain after removing self-links.")

    possible = max(1, n * (n - 1))
    target_density = max(float(base_density),
                         min(1.0, float(min_degree) / max(1, n - 1)))
    active_floor = max(int(np.count_nonzero(lend)),
                       int(np.count_nonzero(borrow_adj)))
    target_links = min(len(candidates), max(
        active_floor, int(round(target_density * possible))))

    # Choose z so that the sum of directed fitness probabilities equals the
    # desired number of links.  The topology itself is the deterministic set
    # of highest-probability links.
    if target_links >= len(candidates):
        scored = [(1.0, i, j) for i, j, _ in candidates]
    else:
        def expected_links(z):
            return sum((z * p) / (1.0 + z * p)
                       for _, _, p in candidates)

        lo, hi = 0.0, 1.0
        while expected_links(hi) < target_links and hi < 1e18:
            hi *= 2.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if expected_links(mid) < target_links:
                lo = mid
            else:
                hi = mid
        z = (lo + hi) / 2.0
        scored = [((z * p) / (1.0 + z * p), i, j)
                  for i, j, p in candidates]

    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    chosen = {(i, j) for _, i, j in scored[:target_links]}

    # Make every positive-margin row and column represented in the support.
    for i in np.flatnonzero(lend > 0):
        if not any(ii == i for ii, _ in chosen):
            _, ii, jj = next(item for item in scored if item[1] == i)
            chosen.add((ii, jj))
    for j in np.flatnonzero(borrow_adj > 0):
        if not any(jj == j for _, jj in chosen):
            _, ii, jj = next(item for item in scored if item[2] == j)
            chosen.add((ii, jj))

    score_map = {(i, j): max(score, 1e-12) for score, i, j in scored}

    def ras_fit(support):
        A = np.zeros((n, n), dtype=float)
        for i, j in support:
            A[i, j] = score_map[i, j]
        last_error = float("inf")
        for iteration in range(1, iters + 1):
            row_sums = A.sum(axis=1)
            positive_rows = lend > 0
            if np.any(row_sums[positive_rows] <= 0):
                return None, last_error, iteration
            A[positive_rows] *= (
                lend[positive_rows] / row_sums[positive_rows])[:, None]

            col_sums = A.sum(axis=0)
            positive_cols = borrow_adj > 0
            if np.any(col_sums[positive_cols] <= 0):
                return None, last_error, iteration
            A[:, positive_cols] *= (
                borrow_adj[positive_cols] / col_sums[positive_cols])[None, :]

            if iteration % 25 == 0 or iteration == iters:
                row_error = float(np.max(np.abs(A.sum(axis=1) - lend)))
                col_error = float(np.max(np.abs(A.sum(axis=0) - borrow_adj)))
                last_error = max(row_error, col_error) / max(1.0, total)
                if last_error < tol:
                    return A, last_error, iteration
        return A, last_error, iters

    initial_links = len(chosen)
    while True:
        A, margin_error, ras_iterations = ras_fit(chosen)
        if A is not None and margin_error < tol:
            break
        if len(chosen) >= len(candidates):
            raise ValueError("RAS did not converge even with all feasible links.")
        add_count = max(1, int(np.ceil(0.05 * len(candidates))))
        for _, i, j in scored:
            if (i, j) not in chosen:
                chosen.add((i, j))
                add_count -= 1
                if add_count == 0:
                    break

    info = {
        "borrow_scale": borrow_scale,
        "base_density": float(base_density),
        "target_density": target_density,
        "initial_links": initial_links,
        "links": len(chosen),
        "density": len(chosen) / possible,
        "ras_iterations": ras_iterations,
        "margin_error": margin_error,
    }
    return (A, info) if return_info else A


# --------------------------------------------------------------------------
# Shocks & propagation dynamics
# --------------------------------------------------------------------------

def initial_shock(net, kind="external", alpha=0.01, bank=None):
    """h(1): 'external' (all banks lose alpha of external assets),
    'externalSel' (one bank), or 'default' (one bank, h=1)."""
    h = np.zeros(net.n)
    if kind == "default":
        h[bank] = 1.0
    elif kind == "externalSel":
        h[bank] = min(1.0, alpha * net.AE[bank] / net.E[bank])
    else:
        h = np.minimum(1.0, alpha * net.AE / net.E)
    return h


def run_linear(net, h0, maxit=1000, tol=1e-11):
    """Linear DebtRank (Bardoscia 2015):
       h_i(t+1) = min{1, h_i(t) + sum_j Lambda_ij [h_j(t) - h_j(t-1)]}.
    Banks keep propagating as long as they keep receiving distress.
    Returns (h_final, H_history, rounds)."""
    Lam = net.leverage
    Esum = net.E.sum()
    h, h_prev = h0.copy(), np.zeros(net.n)
    H_hist = [float(net.E @ h) / Esum]
    rounds = 0
    for t in range(maxit):
        h_new = np.minimum(1.0, h + Lam @ (h - h_prev))
        chg = np.abs(h_new - h).max()
        h_prev, h = h, h_new
        rounds = t + 1
        H_hist.append(float(net.E @ h) / Esum)
        if chg < tol:
            break
    return h, H_hist, rounds


def run_original(net, h0, maxit=1000):
    """Original DebtRank (Battiston 2012): states U/D/I; a bank propagates
    distress only the first time it is distressed (lower bound on linear)."""
    Lam = net.leverage
    Esum = net.E.sum()
    h = h0.copy()
    state = np.where(h > 0, "D", "U")
    H_hist = [float(net.E @ h) / Esum]
    rounds = 0
    for t in range(maxit):
        d = state == "D"
        h_new = np.minimum(1.0, h + Lam[:, d] @ h[d])
        state[d] = "I"
        state[(state == "U") & (h_new > 0)] = "D"
        h = h_new
        rounds = t + 1
        H_hist.append(float(net.E @ h) / Esum)
        if not (state == "D").any():
            break
    return h, H_hist, rounds


def run_dynamics(net, h0, algo="linear", **kw):
    return (run_original if algo == "original" else run_linear)(net, h0, **kw)


def fire_sale(net, h, phi=0.3):
    """Heuristic third-round effect inspired by Battiston et al. (2016).

    Distressed survivors sell external assets in proportion to their equity
    loss, and linear market impact ``phi`` depresses the common asset price.
    This does not solve the paper's target-leverage restoration equation.
    Returns (h_after, price_drop, third_round_H)."""
    Esum = net.E.sum()
    alive = h < 0.999
    sold = float((h * net.AE)[alive].sum())
    price_drop = min(0.99, phi * sold / max(net.AE.sum(), 1e-9))
    add = price_drop * net.AE / net.E
    h2 = np.minimum(1.0, h + add)
    third = float(net.E @ (h2 - h)) / Esum
    return h2, price_drop, third


# --------------------------------------------------------------------------
# Systemic risk measures (Battiston 2016)
# --------------------------------------------------------------------------

def global_vulnerability(net, h):
    """H(t) = equity-weighted average relative loss."""
    return float(net.E @ h) / net.E.sum()


def all_impacts(net, algo="linear"):
    """Default-based DebtRank impact of each bank (Battiston-style).

    Default one bank, then measure the induced relative equity loss in the
    rest of the system, excluding the seed's equity.  This is not Bardoscia
    et al.'s 2015 single-bank external-asset-shock impact experiment.
    """
    Esum = net.E.sum()
    imp = np.zeros(net.n)
    for i in range(net.n):
        h0 = np.zeros(net.n)
        h0[i] = 1.0
        h, _, _ = run_dynamics(net, h0, algo)
        imp[i] = max(0.0, float(net.E @ h) / Esum - net.E[i] / Esum)
    return imp


def monte_carlo(net, algo="linear", mean_shock=0.01, trials=2000,
                level=0.95, seed=12345):
    """Loss distribution under random systemic shocks alpha ~ Beta with the
    given mean; compares first-round-only vs full propagation. Returns dict
    with sorted loss arrays and mean/VaR/CVaR for both."""
    rng = np.random.default_rng(seed)
    a = 2.0
    b = a * (1.0 - mean_shock) / mean_shock
    Esum = net.E.sum()
    first, full = np.empty(trials), np.empty(trials)
    for t in range(trials):
        alpha = rng.beta(a, b)
        h0 = np.minimum(1.0, alpha * net.AE / net.E)
        first[t] = float(net.E @ h0) / Esum
        h, _, _ = run_dynamics(net, h0, algo, maxit=600)
        full[t] = float(net.E @ h) / Esum

    def stats(arr):
        s = np.sort(arr)
        qi = min(len(s) - 1, int(level * len(s)))
        return {"arr": s, "mean": float(s.mean()), "VaR": float(s[qi]),
                "CVaR": float(s[qi:].mean())}

    return {"first": stats(first), "full": stats(full), "level": level}


# --------------------------------------------------------------------------
# Network-diagram layouts (pure numpy; coordinates in [-1, 1]^2)
# --------------------------------------------------------------------------

def _hop_distances(adj):
    """All-pairs shortest hop counts (Floyd-Warshall on the undirected link
    graph); disconnected pairs get max finite distance + 1."""
    n = adj.shape[0]
    D = np.where(adj, 1.0, np.inf)
    np.fill_diagonal(D, 0.0)
    for k in range(n):
        D = np.minimum(D, D[:, k:k + 1] + D[k:k + 1, :])
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = (finite.max() if finite.size else 1.0) + 1.0
    return D


def layout_kk(adj, iters=250):
    """Kamada-Kawai via stress majorization (SMACOF). Default layout."""
    n = adj.shape[0]
    D = _hop_distances(adj)
    L = D / D.max()                                  # ideal lengths
    W = np.where(D > 0, 1.0 / np.maximum(D, 1e-9) ** 2, 0.0)
    ang = 2 * np.pi * np.arange(n) / n
    P = np.column_stack([np.cos(ang), np.sin(ang)])  # deterministic init
    for _ in range(iters):
        for i in range(n):
            d = P[i] - P
            dist = np.maximum(1e-9, np.hypot(d[:, 0], d[:, 1]))
            w = W[i].copy()
            w[i] = 0.0
            sw = w.sum()
            if sw > 0:
                tgt = P + (L[i] / dist)[:, None] * d
                P[i] = (w[:, None] * tgt).sum(axis=0) / sw
    return P


def layout_spring(adj, iters=300, seed=20121212):
    """Fruchterman-Reingold force-directed layout (deterministic seed)."""
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    P = rng.uniform(-1, 1, size=(n, 2))
    k = np.sqrt(4.0 / n)
    temp = 0.25
    A = adj.astype(float)
    for _ in range(iters):
        d = P[:, None, :] - P[None, :, :]
        dist = np.maximum(1e-6, np.hypot(d[..., 0], d[..., 1]))
        rep = (k * k / dist ** 2)[..., None] * d           # repulsion
        att = (A * dist / k)[..., None] * d / dist[..., None]  # attraction
        np.fill_diagonal(rep[..., 0], 0); np.fill_diagonal(rep[..., 1], 0)
        disp = rep.sum(axis=1) - att.sum(axis=1)
        norm = np.maximum(1e-9, np.hypot(disp[:, 0], disp[:, 1]))
        P += disp / norm[:, None] * np.minimum(norm, temp)[:, None]
        temp *= 0.985
    return P


def layout_circular(order):
    """One ring, ordered by DebtRank impact (descending) from the top."""
    n = len(order)
    P = np.zeros((n, 2))
    for rank, node in enumerate(order):
        a = 2 * np.pi * rank / n - np.pi / 2
        P[node] = [np.cos(a), np.sin(a)]
    return P


def layout_shell(order):
    """Concentric rings by DebtRank impact; inner ring = most systemic."""
    n = len(order)
    n1, n2 = max(1, round(n * 0.2)), max(1, round(n * 0.3))
    shells = [order[:n1], order[n1:n1 + n2], order[n1 + n2:]]
    radii = [0.35, 0.68, 1.0]
    P = np.zeros((n, 2))
    for s, sh in enumerate(shells):
        for rank, node in enumerate(sh):
            a = 2 * np.pi * rank / max(1, len(sh)) - np.pi / 2 + s * 0.35
            P[node] = [radii[s] * np.cos(a), radii[s] * np.sin(a)]
    return P


def layout_spiral(order):
    """Battiston et al. (2012) golden-angle spiral; most systemic at centre."""
    n = len(order)
    P = np.zeros((n, 2))
    for rank, node in enumerate(order):
        ang = rank * 2.399963
        rad = 0.12 + 0.88 * rank / max(1, n - 1)
        P[node] = [rad * np.cos(ang), rad * np.sin(ang)]
    return P


LAYOUTS = ("kk", "spring", "circular", "shell", "spiral")


def network_positions(net, impacts, layout="kk"):
    """Dispatch to the chosen layout. Returns n x 2 coordinates."""
    adj = (net.A > 0) | (net.A.T > 0)
    np.fill_diagonal(adj, False)
    order = list(np.argsort(-impacts))
    if layout == "spring":
        return layout_spring(adj)
    if layout == "circular":
        return layout_circular(order)
    if layout == "shell":
        return layout_shell(order)
    if layout == "spiral":
        return layout_spiral(order)
    return layout_kk(adj)


# --------------------------------------------------------------------------
# Report & plots
# --------------------------------------------------------------------------

def console_report(net, scenario_desc, algo, h0, h, H_hist, rounds,
                   third, price_drop, impacts, mc=None):
    E, Esum, n = net.E, net.E.sum(), net.n
    first = global_vulnerability(net, h0)
    H = global_vulnerability(net, h)
    second = H_hist[-1] - first if third == 0 else H - first - third
    ampl = H / first if first > 0 else float("nan")
    defaults = int((h >= 0.999).sum())
    sr = net.spectral_radius

    bar = "=" * 72
    print(bar)
    print("DEBTRANK SYSTEMIC RISK REPORT")
    print(bar)
    print(f"Institutions:            {n}")
    print(f"Interbank links:         {int((net.A > 0).sum())}")
    print(f"Bilateral matrix:        "
          f"{'reconstructed (directed fitness + RAS)' if net.reconstructed else 'read directly'}")
    if net.reconstructed:
        ri = net.reconstruction_info
        print(f"Reconstructed density:  {ri['density']:.1%} "
              f"({ri['links']} links)")
        if abs(ri["borrow_scale"] - 1.0) > 1e-10:
            print(f"Borrowing-total scale:  {ri['borrow_scale']:.6f} "
                  "(rescaled to total lending)")
    print(f"Spectral radius |l_max|: {sr:.4f}  "
          f"({'STABLE (<1): shocks damped' if sr < 1 else 'UNSTABLE (>1): shocks amplified'})")
    print(f"Scenario:                {scenario_desc}")
    print(f"Algorithm:               {'original DebtRank (Battiston 2012)' if algo=='original' else 'linear DebtRank (Bardoscia 2015)'}")
    print("-" * 72)
    print("SYSTEM-LEVEL RESULTS")
    print(f"  Global vulnerability H(T):    {H:8.4f}  ({H*100:.1f}% of system equity)")
    print(f"  First-round (direct) loss:    {first:8.4f}")
    print(f"  Second-round (interbank):     {second:8.4f}")
    if third > 0:
        print(f"  Third-round (simplified fire sales): {third:8.4f}  "
              f"(price drop {price_drop*100:.1f}%)")
    print(f"  Network amplification:        {ampl:8.2f}x")
    print(f"  Equity wiped out:             {H*Esum:10.1f}  of {Esum:.1f}")
    print(f"  Defaults (h=1):               {defaults} / {n}")
    print(f"  Propagation rounds:           {rounds}")
    print("-" * 72)
    print("FIRM-LEVEL RESULTS")
    print(f"  {'Institution':<20s} {'DefImpact':>10s} {'Vulnerability':>14s} "
          f"{'Equity':>10s} {'IB-lev':>7s} {'Default':>8s}")
    order = np.argsort(-impacts)
    iblev = net.ib_lending / E
    for i in order:
        print(f"  {net.names[i]:<20s} {impacts[i]:>10.4f} {h[i]:>14.4f} "
              f"{E[i]:>10.1f} {iblev[i]:>7.2f} {'  yes' if h[i] >= 0.999 else '   no':>8s}")
    if mc:
        lv = int(mc["level"] * 100)
        print("-" * 72)
        print(f"MONTE-CARLO LOSS DISTRIBUTION ({len(mc['full']['arr'])} trials)")
        print(f"  {'':<24s}{'first round only':>18s}{'full propagation':>18s}")
        print(f"  {'Mean loss':<24s}{mc['first']['mean']:>18.4f}{mc['full']['mean']:>18.4f}")
        print(f"  {'VaR ' + str(lv) + '%':<24s}{mc['first']['VaR']:>18.4f}{mc['full']['VaR']:>18.4f}")
        print(f"  {'CVaR ' + str(lv) + '%':<24s}{mc['first']['CVaR']:>18.4f}{mc['full']['CVaR']:>18.4f}")
        print("  -> network effects shift VaR by "
              f"{(mc['full']['VaR'] - mc['first']['VaR'])*100:.1f} pp")
    print(bar)


def make_figure(net, h0, h, H_hist, first, second, third, impacts,
                mc=None, outfile="debtrank_report.png", layout="kk",
                max_edges=90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = net.n
    E = net.E
    ta = net.AE + net.ib_lending
    imax = impacts.max() or 1e-9
    cmap = plt.get_cmap("RdYlBu_r")
    colors = [cmap(0.15 + 0.7 * impacts[i] / imax) for i in range(n)]

    nrow = 3
    fig = plt.figure(figsize=(14, 5 * nrow))
    fig.suptitle("DebtRank Systemic Risk Report", fontsize=15, y=0.995)
    gs = fig.add_gridspec(nrow, 4)

    # (1) convergence of H(t)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.plot(range(len(H_hist)), H_hist, "o-", color="#1f5fa6", ms=4)
    ax.set_xlabel("propagation round t")
    ax.set_ylabel("global vulnerability H(t)")
    ax.set_title("Distress propagation over rounds")
    ax.grid(alpha=0.3)

    # (2) round decomposition
    ax = fig.add_subplot(gs[0, 2:4])
    segs = [("1st round\n(external)", first, "#2471a3"),
            ("2nd round\n(interbank)", second, "#1f5fa6"),
            ("3rd round\n(simplified fire sales)", third, "#b9770e")]
    bottom = 0.0
    for lab, v, c in segs:
        if v > 1e-9:
            ax.bar(["loss decomposition"], [v], bottom=bottom, color=c, label=lab)
            ax.text(0, bottom + v / 2, f"{v*100:.1f}%", ha="center",
                    va="center", color="w", fontweight="bold")
            bottom += v
    ax.set_ylabel("share of system equity lost")
    ax.set_title(f"Loss decomposition: H(T) = {bottom*100:.1f}%")
    ax.legend(loc="upper right", fontsize=9)

    # (3) composite firm-level rankings: impact + vulnerability side by side
    ax = fig.add_subplot(gs[1, 0])
    order = np.argsort(impacts)
    ax.barh([net.names[i] for i in order], impacts[order] * 100,
            color=[colors[i] for i in order])
    ax.set_xlabel("default impact (% system equity)")
    ax.set_title("Systemic default-impact ranking\n(Battiston-style DebtRank)", fontsize=10)
    ax.tick_params(axis="y", labelsize=7.5)

    ax = fig.add_subplot(gs[1, 1])
    vorder = np.argsort(h)
    vcols = ["#c0392b" if h[i] >= 0.999 else "#1f5fa6" for i in vorder]
    ax.barh([net.names[i] for i in vorder], h[vorder] * 100, color=vcols)
    ax.set_xlabel("loss h(T) (% own equity)")
    ax.set_title("Vulnerability ranking\n(this scenario; red = default)",
                 fontsize=10)
    ax.tick_params(axis="y", labelsize=7.5)

    # (4) impact vs vulnerability scatter
    ax = fig.add_subplot(gs[1, 2:4])
    sizes = 60 + 500 * ta / ta.max()
    ax.scatter(impacts, h, s=sizes, c=colors, edgecolors="k",
               linewidths=0.5, alpha=0.85)
    mx, my = np.median(impacts), np.median(h)
    ax.axvline(mx, ls="--", color="#b9770e", alpha=0.5)
    ax.axhline(my, ls="--", color="#b9770e", alpha=0.5)
    for i in range(n):
        if impacts[i] > mx or h[i] > my or ta[i] > 0.6 * ta.max():
            ax.annotate(net.names[i], (impacts[i], h[i]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("default-based systemic impact")
    ax.set_ylabel("vulnerability h(T) under scenario")
    ax.set_title("Impact vs vulnerability (upper-right = too-central-to-fail)")
    ax.grid(alpha=0.3)

    # (5) network & DebtRank diagram (selectable layout)
    ax = fig.add_subplot(gs[2, 0:2])
    pos = network_positions(net, impacts, layout)
    edges = [(i, j, net.A[i, j]) for i in range(n) for j in range(n)
             if net.A[i, j] > 0]
    edges.sort(key=lambda e: -e[2])
    emax = edges[0][2] if edges else 1.0
    drawn = select_edges(edges, max_edges)
    for i, j, w in drawn:
        ax.plot(*zip(pos[i], pos[j]), color=colors[i],
                lw=0.3 + 2.0 * w / emax, alpha=0.25, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=80 + 900 * ta / ta.max(),
               c=colors, edgecolors="w", zorder=2)
    for i in range(n):
        ax.annotate(net.names[i], pos[i], fontsize=6.5, ha="center",
                    xytext=(0, 9), textcoords="offset points", zorder=3)
    titles = {"kk": "Kamada-Kawai layout",
              "spring": "Spring (Fruchterman-Reingold) layout",
              "circular": "Circular layout (ordered by DebtRank)",
              "shell": "Shell layout (inner ring = most systemic)",
              "spiral": "DebtRank spiral: centre = most systemic"}
    shown = (f"{len(drawn)} of {len(edges)} exposures shown"
             if edges else "no exposures")
    ax.set_title(f"Network & DebtRank diagram - {titles.get(layout, layout)}"
                 f"\n({shown})", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

    # (6) Monte-Carlo loss distribution (optional)
    ax = fig.add_subplot(gs[2, 2:4])
    if mc:
        bins = np.linspace(0, max(mc["full"]["arr"].max(),
                                  mc["first"]["arr"].max()) * 1.02, 41)
        ax.hist(mc["first"]["arr"], bins=bins, color="#2471a3", alpha=0.5,
                label="first round only")
        ax.hist(mc["full"]["arr"], bins=bins, color="#c0392b", alpha=0.5,
                label="full propagation")
        lv = int(mc["level"] * 100)
        ax.axvline(mc["first"]["VaR"], color="#2471a3", ls="--",
                   label=f"VaR{lv} first = {mc['first']['VaR']*100:.0f}%")
        ax.axvline(mc["full"]["VaR"], color="#c0392b", ls="--",
                   label=f"VaR{lv} full = {mc['full']['VaR']*100:.0f}%")
        ax.set_xlabel("global equity loss H(T)")
        ax.set_ylabel("frequency")
        ax.set_title("Loss distribution: network effects shift VaR right")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Monte-Carlo panel\n(run with --mc TRIALS)",
                ha="center", va="center", color="gray")
        ax.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outfile, dpi=140)
    print(f"\nFigure saved to: {outfile}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="DebtRank systemic-risk stress test (Python version of "
                    "DebtRank_Systemic_Risk_Tool_v03.html)")
    p.add_argument("csv", help="input CSV (bilateral matrix or aggregate)")
    p.add_argument("--algo", choices=["linear", "original"], default="linear",
                   help="linear DebtRank (Bardoscia 2015, default) or "
                        "original DebtRank (Battiston 2012)")
    p.add_argument("--shock", choices=["external", "externalSel", "default"],
                   default="external",
                   help="external-asset shock to all banks (default), to one "
                        "bank, or single-bank default")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="shock size in %% of external assets (default 1.0)")
    p.add_argument("--bank", default=None,
                   help="bank name (or index) for externalSel/default shocks")
    p.add_argument("--fire-sales", action="store_true",
                   help="add simplified heuristic third-round fire-sale losses")
    p.add_argument("--phi", type=float, default=0.30,
                   help="fire-sale market impact (default 0.30)")
    p.add_argument("--mc", type=int, default=0, metavar="TRIALS",
                   help="run Monte-Carlo loss distribution with TRIALS draws")
    p.add_argument("--mc-mean", type=float, default=1.0,
                   help="mean Monte-Carlo shock size in %% (default 1.0)")
    p.add_argument("--var-level", type=float, default=0.95,
                   help="VaR/CVaR confidence level (default 0.95)")
    p.add_argument("--seed", type=int, default=12345, help="MC random seed")
    p.add_argument("--layout", choices=list(LAYOUTS), default="kk",
                   help="network-diagram layout: kk = Kamada-Kawai (default), "
                        "spring = Fruchterman-Reingold, circular, shell, "
                        "spiral = Battiston 2012")
    p.add_argument("--max-edges", type=int, default=90, metavar="N",
                   help="draw the N largest exposures in the network diagram "
                        "(default 90; e.g. 80, 50; 0 = draw all). Cosmetic "
                        "only -- does not change the DebtRank/contagion results.")
    p.add_argument("--no-plots", action="store_true",
                   help="console report only (no matplotlib needed)")
    p.add_argument("--out", default="debtrank_report.png",
                   help="output figure filename")
    args = p.parse_args(argv)

    net = load_network(args.csv)

    bank_idx = 0
    if args.bank is not None:
        if args.bank.isdigit():
            bank_idx = int(args.bank)
        else:
            try:
                bank_idx = net.names.index(args.bank)
            except ValueError:
                sys.exit(f"Bank '{args.bank}' not found. Banks: {net.names}")
    elif args.shock in ("externalSel", "default"):
        print(f"(no --bank given; using '{net.names[0]}')")

    alpha = args.alpha / 100.0
    h0 = initial_shock(net, args.shock, alpha, bank_idx)
    desc = {"external": f"{args.alpha}% external-asset shock, all banks",
            "externalSel": f"{args.alpha}% external-asset shock on "
                           f"{net.names[bank_idx]}",
            "default": f"default of {net.names[bank_idx]}"}[args.shock]

    h, H_hist, rounds = run_dynamics(net, h0, args.algo)
    third, price_drop = 0.0, 0.0
    if args.fire_sales:
        h, price_drop, third = fire_sale(net, h, args.phi)
        H_hist = H_hist + [global_vulnerability(net, h)]

    impacts = all_impacts(net, args.algo)
    mc = (monte_carlo(net, args.algo, args.mc_mean / 100.0, args.mc,
                      args.var_level, args.seed) if args.mc > 0 else None)

    first = global_vulnerability(net, h0)
    second = global_vulnerability(net, h) - first - third
    console_report(net, desc, args.algo, h0, h, H_hist, rounds,
                   third, price_drop, impacts, mc)

    if not args.no_plots:
        try:
            make_figure(net, h0, h, H_hist, first, second, third,
                        impacts, mc, args.out, args.layout, args.max_edges)
        except ImportError:
            print("matplotlib not installed -> skipping figure "
                  "(pip install matplotlib)")


if __name__ == "__main__":
    main()
