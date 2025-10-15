from cmdstanpy import CmdStanModel
import tempfile
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

class IrisAnalysis:
    def __init__(self, trait = "sepal length (cm)"):
        
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df["species"] = iris.target
        self.trait = trait
        self.key = iris.target_names
        self.K = len(np.unique(iris.target))
        
        self.stan_data = {
            "N": len(self.df),
            "K": self.K,
            "species": (self.df["species"].to_numpy() + 1).tolist(),
            "y": self.df[self.trait].values.tolist(),
        }
                     
    def set_priors(self, s_intercept, s_species_effect, s_residual):

        return f"""
data {{
    int<lower=1> N;
    int<lower=1> K;
    array[N] int<lower=1, upper=K> species;
    vector[N] y;
}}

parameters {{
    real intercept;
    vector[K-1] beta;
    real<lower=0> residual;
}}

transformed parameters {{
    vector[N] mu;
    vector[K] species_effects;
    species_effects[1] = 0;
    species_effects[2:K] = beta;
    for (i in 1:N) {{
        mu[i] = intercept + species_effects[species[i]];
    }}
}}

model {{
    // priors
    intercept ~ normal(0, {s_intercept});
    beta ~ normal(0, {s_species_effect});
    residual ~ normal(0, {s_residual});

    // model likelihood
    y ~ normal(mu, residual);
}}
"""
    def set_initval(self, s_intercept, s_species_effect, s_residual, chains, initval_seed):

        np.random.seed(initval_seed)
        return [
            {
                "intercept": np.random.normal(0, s_intercept),
                "beta": abs(np.random.normal(0, s_species_effect, size = self.K - 1)),
                "residual": abs(np.random.normal(0, s_residual)),
            }
            for _ in range(chains)
        ]

    def compile(self, s_intercept, s_species_effect, s_residual):
        prog = self.set_priors(s_intercept, s_species_effect, s_residual)
        tmp = tempfile.NamedTemporaryFile(suffix='.stan', delete=False)
        tmp.write(prog.encode('utf-8'))
        tmp.close()
        mod = CmdStanModel(stan_file = tmp.name)
        self.__exe__ = mod.exe_file
    
    def sample(
        self,
        data,
        s_intercept, s_species_effect, s_residual,
        warmup: int = 500,
        samples: int = 500,
        thinning: int = 1,
        initval_seed: int = 123,
        stan_seed: int = 456,
        adapt_delta: float = 0.8,
        chains: int = 4,
    ):
        initval = self.set_initval(s_intercept, s_species_effect, s_residual, chains, initval_seed)

        mod = CmdStanModel(exe_file=self.__exe__)
        stan_fit = mod.sample(
            data = data,
            seed = stan_seed,
            chains = chains,
            parallel_chains = chains,
            iter_warmup = warmup,
            iter_sampling = samples,
            thin = thinning,
            inits = initval,
            adapt_delta = adapt_delta,
            show_console = False,
        )
        
        summary = stan_fit.summary()
        posteriors = {}
        for val in ["intercept","species_effects","residual"]:
            posteriors[val] = stan_fit.stan_variable(val)

        R2 = self.R2(summary, data["y"], species_idx = np.array(data["species"]) - 1)

        return summary, posteriors, R2

    def make_folds(self, k: int = 5, seed_folds: int = 123):

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed_folds)
        folds = []

        x = self.df[self.trait].values
        y = self.df["species"].values
    
        for train_idx, test_idx in skf.split(x, y):                
            train_subset = self.df.iloc[train_idx]
            test_subset = self.df.iloc[test_idx]

            train_stan_data = {
                "N": len(train_subset),
                "K": self.K,
                "species": (train_subset["species"].to_numpy() + 1).tolist(),
                "y": train_subset[self.trait].values.tolist(),
            }
    
            test_stan_data = {
                "N": len(test_subset),
                "K": self.K,
                "species": (test_subset["species"].to_numpy() + 1).tolist(),
                "y": test_subset[self.trait].values.tolist(),
            }
    
            folds.append({
                "train": train_stan_data,
                "test": test_stan_data,
            })
        return folds

    def R2(
        self,
        summary,
        y,
        species_idx
    ):
        intercept_median = summary.loc["intercept", "50%"]
        species_effects_median = summary.loc[
            summary.index.str.startswith("species_effects"), "50%"
        ].values
        y_hat = intercept_median + species_effects_median[species_idx]
        y_true = np.array(y)
        ss_res = np.sum((y_true - y_hat) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot

    def cross_val(
        self,
        k,
        s_intercept,
        s_species_effect,
        s_residual,
        **sample_kwargs,
    ):
        folds = self.make_folds(k=k)
        self.compile(s_intercept, s_species_effect, s_residual)

        results = []

        for i, fold in enumerate(folds):
            test_data = fold["test"]
            train_data = fold["train"]
    
            summary_i, posteriors_i, R2_fit_i = self.sample(
                train_data,
                s_intercept = s_intercept,
                s_species_effect = s_species_effect,
                s_residual = s_residual,
                **sample_kwargs,
            )

            R2_pred_i = self.R2(
                summary_i, test_data["y"],
                species_idx = np.array(test_data["species"]) - 1
            )
            
            results.append({
                "fold": i,
                "R2_fit": R2_fit_i,
                "R2_pred": R2_pred_i,
                "summary": summary_i,
                "posteriors": posteriors_i,
            })

        return results
        
