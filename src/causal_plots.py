"""Visualization helpers for causal discovery results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_dag(
    edges: list[tuple[str, str, ...]],
    out_path: Path,
    title: str = "Causal DAG",
) -> None:
    """Draw a directed graph from a list of (cause, effect, ...) tuples."""
    G = nx.DiGraph()
    for e in edges:
        cause, effect = e[0], e[1]
        weight = e[2] if len(e) > 2 else 1.0
        G.add_edge(cause, effect, weight=round(float(weight), 3))

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2.0)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, node_color="lightblue",
                           edgecolors="black", linewidths=1.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray",
                           arrows=True, arrowsize=20, width=1.5,
                           connectionstyle="arc3,rad=0.1")
    edge_labels = nx.get_edge_attributes(G, "weight")
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=8)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_adjacency_heatmap(
    adj: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str = "LiNGAM Adjacency Matrix",
) -> None:
    """Heatmap of a weighted adjacency matrix."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(adj, cmap="RdBu_r", vmin=-np.abs(adj).max(), vmax=np.abs(adj).max())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    # Annotate cells
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            val = adj[i, j]
            if abs(val) >= 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > np.abs(adj).max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Cause (j)")
    ax.set_ylabel("Effect (i)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_edge_bootstrap(
    boot_df: pd.DataFrame,
    out_path: Path,
    top_n: int = 25,
    title: str = "Bootstrap Edge Stability",
) -> None:
    """Horizontal bar chart of bootstrap edge frequencies."""
    df = boot_df.head(top_n).copy()
    df["label"] = df["cause"] + " → " + df["effect"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
    colors = ["steelblue" if f >= 0.5 else "lightcoral" for f in df["frequency"]]
    ax.barh(range(len(df)), df["frequency"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlabel("Frequency (fraction of bootstrap runs)")
    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
