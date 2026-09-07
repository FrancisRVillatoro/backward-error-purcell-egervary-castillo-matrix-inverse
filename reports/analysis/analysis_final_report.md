# Recovered Castillo campaign: statistical analysis

## Data basis

The analysis uses the recovered, matrix-complete dataset. The complete core contains 20,000 replicas per stochastic parameter cell for n=8,...,181. A deterministic hash-based sample of exactly 2,000 matrices per stochastic cell is used for every dimension through n=512. All five methods and both precisions remain paired within every selected matrix.

## Global accounting

- Balanced matrices: `1664000`
- Exact core inverse rows: `128000000`
- Recovered inverse records counted: `146867630`
- Recovered inverse failures counted: `327222`
- Overall recovered inverse-failure rate: `2.22800627e-03`

## Scaling of q95(r_R/u)

| method | precision | cells | median s | q25 | q75 |
|---|---|---:|---:|---:|---:|
| R0_C0 | float32 | 64 | 0.494519 | 0.0545094 | 0.952784 |
| R0_C0 | float64 | 64 | 0.548741 | 0.0390121 | 1.43146 |
| R0_C1 | float32 | 64 | 0.25575 | 0.120284 | 0.445507 |
| R0_C1 | float64 | 64 | 0.290432 | 0.110041 | 0.451558 |
| R0_C2 | float32 | 64 | 0.282845 | 0.107488 | 0.477864 |
| R0_C2 | float64 | 64 | 0.345077 | 0.0849495 | 0.485894 |
| R1_C1 | float32 | 64 | 0.224856 | 0.085281 | 0.585522 |
| R1_C1 | float64 | 64 | 0.234271 | 0.0875327 | 0.590985 |
| R2_C2 | float32 | 64 | 0.279802 | 0.0863899 | 0.584257 |
| R2_C2 | float64 | 64 | 0.297281 | 0.0888355 | 0.58415 |

## Robustness checks

- Maximum absolute standardized censoring difference in numerical metrics (runtime excluded): `0.546676`.
- Maximum standardized runtime difference between low- and high-progress task quartiles: `0.482083`.
- Median paired forward-error ratio after two refinement steps versus no refinement: `0.435951`. Values below one indicate improvement.
- Bootstrap tasks completed: `640`; percentile cluster bootstrap with `2000` replicates per task.

## Files

- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/inverse_core_exact_quantiles.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/inverse_balanced_quantiles.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/solution_balanced_quantiles.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/inverse_failure_rates_all_recovered.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/deterministic_sequences.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/refinement_effects.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/core_vs_balanced_sensitivity.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/inverse_power_fits_by_cell.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/inverse_power_fits_by_family.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/precision_consistency.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/censoring_audit.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/bootstrap_power_slopes.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/q95_slope_summary.csv`
- `/mnt/home/users/tic_118_uma/frv/castillo_stability_campaign/reports/analysis/castillo_recovered_analysis.png`

## Scope limitations

- The complete 20000-replica core ends at n=181.
- The exact balanced extension uses 2000 matrices per stochastic cell through n=512.
- n=724 is not included in the balanced inferential analysis.
- n=1024 is absent from the recovered campaign.
- Bootstrap intervals are percentile cluster-bootstrap intervals; BCa intervals were not used because a 2000-cluster jackknife for every cell/method/precision combination would add disproportionate cost without changing the primary inferential target.
