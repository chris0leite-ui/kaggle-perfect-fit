"""Run all diagnostic analyses on train.csv and generate plots."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import split_holdout
from src.clusters import assign_clusters, replace_sentinels
from src.diagnostics import (
    sentinel_cluster_crosstab,
    sentinel_chi2_test,
    sentinel_target_regression,
    fit_city_x4_ols,
    residual_normality_test,
    residual_heteroscedasticity_test,
    residual_stats,
    fit_gam_per_cluster,
    pooled_vs_cluster_gam_test,
    fit_r2_ceiling,
    feature_group_r2_breakdown,
    x4_bimodality_test,
    x4_city_distribution_test,
    x4_gap_analysis,
)
from src.diagnostics_plots import (
    plot_sentinel_distribution,
    plot_residual_analysis,
    plot_gam_per_cluster,
    plot_r2_breakdown,
    plot_ceiling_residuals,
    plot_x4_bimodality,
)

DATA_PATH = Path("data/dataset.csv")
PLOT_DIR = Path("plots/diagnostics")


def main():
    # Load train data
    train, _ = split_holdout(DATA_PATH)
    train["cluster"] = assign_clusters(train)

    print("=" * 60)
    print("TODO 1: x5 Sentinel Missingness")
    print("=" * 60)

    ct = sentinel_cluster_crosstab(train, "cluster")
    print("\nCross-tab:\n", ct)

    chi2 = sentinel_chi2_test(train, "cluster")
    print(f"\nChi-squared: {chi2['chi2']:.3f}, p-value: {chi2['p_value']:.4f}, dof: {chi2['dof']}")

    reg = sentinel_target_regression(train)
    print(f"\nOLS: target ~ x5_imputed + x5_is_sentinel")
    print(f"  x5_imputed coef: {reg['coef_x5_imputed']:.4f} (p={reg['pvalue_x5_imputed']:.4f})")
    print(f"  x5_sentinel coef: {reg['coef_x5_sentinel']:.4f} (p={reg['pvalue_x5_sentinel']:.4f})")
    print(f"  R-squared: {reg['r_squared']:.4f}")

    plot_sentinel_distribution(ct, PLOT_DIR / "sentinel_by_cluster.png")
    print("  -> Plot saved")

    print("\n" + "=" * 60)
    print("TODO 2: Residual Analysis after City + x4")
    print("=" * 60)

    ols = fit_city_x4_ols(train)
    stats = residual_stats(ols)
    print(f"\nOLS: target ~ City + x4")
    print(f"  R-squared: {stats['r_squared']:.4f}")
    print(f"  Adj R-squared: {stats['adj_r_squared']:.4f}")
    print(f"  Residual mean: {stats['residual_mean']:.6f}")
    print(f"  Residual std: {stats['residual_std']:.3f}")
    print(f"  Residual skew: {stats['residual_skew']:.3f}")
    print(f"  Residual kurtosis: {stats['residual_kurtosis']:.3f}")

    norm = residual_normality_test(ols.resid)
    print(f"\nShapiro-Wilk: stat={norm['statistic']:.4f}, p={norm['p_value']:.6f}")

    het = residual_heteroscedasticity_test(ols.resid, train["cluster"])
    print(f"Levene's test: stat={het['statistic']:.4f}, p={het['p_value']:.4f}")

    plot_residual_analysis(ols, train["cluster"], PLOT_DIR / "residual_analysis.png")
    print("  -> Plot saved")

    print("\n" + "=" * 60)
    print("TODO 3: x1/x2 Nonlinear Shapes Within Clusters")
    print("=" * 60)

    for feat in ["x1", "x2"]:
        gams = fit_gam_per_cluster(train, feat, "cluster")
        print(f"\n{feat} GAM per cluster:")
        for label, info in sorted(gams.items()):
            print(f"  {label}: R-sq={info['r_squared']:.4f}, n={info['n']}")

        consistency = pooled_vs_cluster_gam_test(train, feat, "cluster")
        print(f"  F-test: F={consistency['f_stat']:.3f}, p={consistency['p_value']:.4f}")
        print(f"  RSS pooled: {consistency['rss_pooled']:.1f}, RSS cluster: {consistency['rss_cluster']:.1f}")

        plot_gam_per_cluster(gams, feat, PLOT_DIR / f"gam_{feat}_per_cluster.png")
        print(f"  -> Plot saved")

    print("\n" + "=" * 60)
    print("TODO 4: Total R-squared Ceiling")
    print("=" * 60)

    ceiling = fit_r2_ceiling(train)
    print(f"\nCeiling model (City + x4 + GAM(x1) + GAM(x2) + x5 + x8 + x10 + x11):")
    print(f"  R-squared: {ceiling['r_squared']:.4f}")
    print(f"  Adj R-squared: {ceiling['adj_r_squared']:.4f}")
    print(f"  Residual std: {ceiling['residual_std']:.3f}")
    print(f"  n: {ceiling['n']}")

    breakdown = feature_group_r2_breakdown(train)
    print("\nR-squared breakdown:")
    print(breakdown.to_string(index=False))

    plot_r2_breakdown(breakdown, PLOT_DIR / "r2_breakdown.png")

    # Compute residuals for histogram
    from src.diagnostics import _prepare_ceiling_data
    from pygam import LinearGAM, s as spline, l as linear_term
    X_df, y = _prepare_ceiling_data(train)
    X = X_df.values
    terms = (linear_term(0) + linear_term(1) + spline(2, n_splines=20)
             + spline(3, n_splines=20) + linear_term(4) + linear_term(5)
             + linear_term(6) + linear_term(7))
    gam_full = LinearGAM(terms)
    gam_full.fit(X, y)
    residuals = y - gam_full.predict(X)
    plot_ceiling_residuals(residuals, PLOT_DIR / "ceiling_residuals.png")
    print("  -> Plots saved")

    print("\n" + "=" * 60)
    print("TODO 5: x4 Bimodal Origin")
    print("=" * 60)

    bimod = x4_bimodality_test(train["x4"])
    print(f"\nBimodality test ({bimod['method']}):")
    print(f"  Statistic (gap ratio): {bimod['statistic']:.2f}")
    print(f"  Gap: [{bimod['gap_start']:.4f}, {bimod['gap_end']:.4f}], width={bimod['gap_width']:.4f}")

    city_dist = x4_city_distribution_test(train)
    print(f"\nKS test (Albacete vs Zaragoza):")
    print(f"  KS stat: {city_dist['ks_statistic']:.4f}, p={city_dist['p_value']:.4f}")
    print(f"  Mean Albacete: {city_dist['mean_albacete']:.4f}, Zaragoza: {city_dist['mean_zaragoza']:.4f}")

    gap = x4_gap_analysis(train["x4"])
    print(f"\nGap analysis:")
    print(f"  Gap: [{gap['gap_start']:.4f}, {gap['gap_end']:.4f}]")
    print(f"  n_below: {gap['n_below']}, n_above: {gap['n_above']}")
    print(f"  Nearest below: {gap['nearest_below']:.6f}")
    print(f"  Nearest above: {gap['nearest_above']:.6f}")
    print(f"  Exact gap: {gap['exact_gap']}")

    plot_x4_bimodality(train, gap, PLOT_DIR / "x4_bimodality.png")
    print("  -> Plot saved")

    print("\n" + "=" * 60)
    print("All diagnostics complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
