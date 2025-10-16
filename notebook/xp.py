import mlflow
import hashlib
import json
import tempfile
import pandas as pd
import numpy as np
from IrisAnalysis import IrisAnalysis
import pathlib
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("bayes_iris_sensitivity")

def hash_text(txt: str) -> str:
    import hashlib
    return hashlib.sha1(txt.encode("utf-8")).hexdigest()

trait = "sepal length (cm)"
ia = IrisAnalysis(trait=trait)

path = pathlib.Path("IrisAnalysis.py")
iris_analysis_hash = hash_text(path.read_text())

k = 5
seed_folds = 123
chains = 4
warmup = 500
samples = 1000
thin = 1
adapt_delta = 0.8

for s_int in [.01, 1, 100]:
    for s_sp in [.01, 1, 100]:
        for s_res in  [.01, 1, 100]:
            
            stan_code = ia.set_priors(s_int, s_sp, s_res)
            stan_hash = hash_text(stan_code)

            with mlflow.start_run(run_name=f"priors(si={s_int},ss={s_sp},sr={s_res})") as run:

                mlflow.log_params({
                    "trait": trait,
                    "k": k,
                    "seed_folds": seed_folds,
                    "chains": chains,
                    "warmup": warmup,
                    "samples": samples,
                    "thin": thin,
                    "adapt_delta": adapt_delta,
                    "s_intercept": s_int,
                    "s_species_effect": s_sp,
                    "s_residual": s_res,
                    "stan_hash": stan_hash,
                    "iris_analysis_hash": iris_analysis_hash
                })

                results = ia.cross_val(
                    k = k,
                    s_intercept = s_int,
                    s_species_effect = s_sp,
                    s_residual = s_res,
                    warmup = warmup,
                    samples = samples,
                    chains = chains,
                    adapt_delta = adapt_delta,
                )

                for fold_idx, res in enumerate(results):
                    with mlflow.start_run(run_name=f"fold_{fold_idx+1}", nested=True):
                        mlflow.log_metrics({
                            "R2_fit": res["R2_fit"],
                            "R2_pred": res["R2_pred"]
                        })

                        csv_path = tempfile.mktemp(suffix=f"_fold{fold_idx+1}.csv")
                        res["summary"].to_csv(csv_path)
                        mlflow.log_artifact(csv_path, artifact_path=f"fold_{fold_idx+1}")

                        nsp = res["posteriors"]["species_effects"].shape[1]
                        cmap = plt.get_cmap("jet")
                        colors = [cmap(x / (nsp + 1)) for x in range(nsp + 1)]
                        fig, ax = plt.subplots()
                        ax.plot(res["posteriors"]["intercept"], color=colors[0], linewidth=0.5)
                        for j in range(nsp):
                            ax.plot(res["posteriors"]["species_effects"][:, j], color=colors[1 + j], linewidth=0.5)
                        
                        ax.set_title("linear predictor")
                        fig_path1 = tempfile.mktemp(suffix=f"_fold{fold_idx+1}_linear.png")
                        plt.savefig(fig_path1, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        mlflow.log_artifact(fig_path1, artifact_path=f"fold_{fold_idx+1}/plots")
                        
                        fig, ax = plt.subplots()
                        ax.plot(res["posteriors"]["residual"], color="black", linewidth=0.5)
                        ax.set_title("residual")
                        fig_path2 = tempfile.mktemp(suffix=f"_fold{fold_idx+1}_residual.png")
                        plt.savefig(fig_path2, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                        mlflow.log_artifact(fig_path2, artifact_path=f"fold_{fold_idx+1}/plots")

                R2_fit_values = [r["R2_fit"] for r in results]
                R2_pred_values = [r["R2_pred"] for r in results]
                mlflow.log_metrics({
                    "R2_fit_mean": np.mean(R2_fit_values),
                    "R2_pred_mean": np.mean(R2_pred_values),
                    "R2_fit_median": np.median(R2_fit_values),
                    "R2_pred_median": np.median(R2_pred_values),
                    "R2_fit_min": np.min(R2_fit_values),
                    "R2_fit_max": np.max(R2_fit_values),
                    "R2_pred_min": np.min(R2_pred_values),
                    "R2_pred_max": np.max(R2_pred_values),
                    "R2_fit_std": np.std(R2_fit_values, ddof=1),
                    "R2_pred_std": np.std(R2_pred_values, ddof=1)
                })




