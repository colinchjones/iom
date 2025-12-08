import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from scipy import stats
from scipy.special import kv, gamma
from scipy.integrate import quad 
from scipy.optimize import brentq
from pingouin import multivariate_normality

import torch
import normflows as nf
from tqdm import tqdm


class InfluentialOutlierMetric:
    """
    Influential Outlier Metric (IOM)

    This class implements a method to detect influential outliers in regression 
    or machine learning models by combining:

    1. **SHAP values**: proxy for leverage/influence.
    2. **Residuals**: proxy for error magnitude.

    A normalizing flow is applied to both SHAP values and residuals to transform 
    them toward Gaussianity while controlling the transformation via a penalty λ.  
    The Mahalanobis distance (for SHAP) and squared residuals are then combined 
    multiplicatively to form an *Influential Outlier Metric* (IOM).

    Parameters
    ----------
    shap_values : np.ndarray
        Precomputed SHAP values of shape (n_samples, n_features).
    residuals : np.ndarray
        Residuals from the model of shape (n_samples,).
    K, hidden, layers : int
        Architecture parameters for the SHAP normalizing flow.
    K_resid, hidden_resid, layers_resid : int
        Architecture parameters for the residual normalizing flow.
    lambdas, lambdas_resid : array-like
        Candidate λ values to control the penalty on Jacobian regularization.
    epoch, epoch_resid : int
        Number of training epochs for SHAP and residual flows respectively.

    Attributes
    ----------
    shap_values_z : np.ndarray
        Transformed SHAP values in latent space.
    resid_z : np.ndarray
        Transformed residuals in latent space.
    pval : float
        p-value for Gaussianity test on transformed SHAP values.
    pval_resid : float
        p-value for Gaussianity test on transformed residuals.
    lam_final : float
        Final λ for SHAP values.
    lam_final_resid : float
        Final λ for residuals.
    IOM_ : pd.Series
        Final Influential Outlier Metric values.
    """

    def __init__(self, shap_values, residuals,
                 K, layers, hidden,
                 K_resid, layers_resid, hidden_resid,
                 lambdas=np.concatenate([[0], np.exp(np.linspace(-5, 5, 50))]),
                 lambdas_resid=np.concatenate([[0], np.exp(np.linspace(-5, 5, 50))]),
                 epoch=500, epoch_resid=500,
                 seed=True):

        self.shap_values_ = shap_values
        self.resid = residuals.reshape(-1, 1)
        self.n, self.p = self.shap_values_.shape
        self.seed = seed

        self.K = K
        self.hidden = hidden
        self.layers = layers
        self.epoch = epoch
        self.lambdas = lambdas

        self.K_resid = K_resid
        self.hidden_resid = hidden_resid
        self.layers_resid = layers_resid
        self.epoch_resid = epoch_resid
        self.lambdas_resid = lambdas_resid
        
        self.shap_values_z = None
        self.pval = None
        self.lam_final = None
        self.lam_final_resid = None
        self.resid_z = None
        self.pval_resid = None
        self.IOM_ = None


    def norm_flow(self, lam, resid=False):
        """
        Train a normalizing flow and transform data.

        Maps:
        - Raw SHAP values → latent Gaussian representation
        - Residuals → latent Gaussian representation

        Parameters
        ----------
        lam : float
            Regularization parameter penalizing deviation from identity flow.
        resid : bool, default=False
            If True, use residuals instead of SHAP values.

        Returns
        -------
        z : np.ndarray
            Latent-space transformed data.
        pval : float
            p-value from Gaussianity test (Shapiro if 1D, 
            Henze Zirkler otherwise).
        """
        if resid:
            K = self.K_resid
            data = self.resid
            latent_size = 1
            hidden = self.hidden_resid
            layers = self.layers_resid
            epoch = self.epoch_resid
        else:
            K = self.K
            data = self.shap_values_
            latent_size = self.p
            hidden = self.hidden
            layers = self.layers
            epoch = self.epoch

        # Scale input data
        scaler = StandardScaler()
        input_data = scaler.fit_transform(data)

        if self.seed:
            torch.manual_seed(123)
            torch.cuda.manual_seed_all(123)

        # Normalizing flow definition
        flows = []
        for i in range(K):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, 
                                                                     layers, 
                                                                     hidden, 
                                                                     permute_mask=True,
                                                                     init_identity=True)]
            flows += [nf.flows.ActNorm(latent_size)]

        q0 = nf.distributions.DiagGaussian(latent_size, trainable=False)
        mod = nf.NormalizingFlow(q0, flows)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mod = mod.to(device)

        optimizer = torch.optim.AdamW(mod.parameters(), lr=3e-4)
        x = torch.tensor(input_data, dtype=torch.float32, device=device)

        # Training loop
        for it in tqdm(range(epoch), desc="Normalizing flow"):
            optimizer.zero_grad()
            _, jac = mod.inverse_and_log_det(x)
            loss = mod.forward_kld(x) + lam * torch.mean(jac**2)
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

        # Transform data
        with torch.no_grad():
            z = mod.inverse(x).detach().cpu().numpy()

        # Normality test
        if latent_size == 1:
            pval = stats.shapiro(z.reshape(-1)).pvalue
        else:
            pval = multivariate_normality(z).pval

        return z, pval


    def find_best_lambda(self, alpha=0.05):
        """
        Find largest λ where SHAP values pass normality test.

        - If p=1 (univariate), use Shapiro test.
        - If p>1 (multivariate), use Henze–Zirkler test from pingouin.
        - If data already Gaussian, set λ = 1e10 and apply identity flow.
        """
        print("Finding λ for SHAP values")
        last_pass = None

        # Case: univariate
        if self.p == 1:
            jb_stat, jb_pval = stats.shapiro(self.shap_values_)
            print(f"Shapiro p-value = {jb_pval:.4f}")
            if jb_pval > alpha:
                lam_null = 1e10
                z, pval = self.norm_flow(lam_null, resid=False)
                print(f"Already normal (Shapiro). Using λ={lam_null}, p={pval:.4f}")
                last_pass = (lam_null, pval, z)
                self.shap_values_z, self.pval = last_pass[2], last_pass[1]
                print("Done!")
                return  # stop here

        else:
            # Case: multivariate
            hz_pval = multivariate_normality(self.shap_values_).pval
            print(f"Henze–Zirkler p-value = {hz_pval:.4f}")
            if hz_pval > alpha:
                lam_null = 1e10
                z, pval = self.norm_flow(lam_null, resid=False)
                print(f"Already normal (Henze–Zirkler). Using λ={lam_null}, p={pval:.4f}")
                last_pass = (lam_null, pval, z)
                self.shap_values_z, self.pval = last_pass[2], last_pass[1]
                print("Done!")
                return  # stop here

        # Otherwise, search over lambdas
        for lam in self.lambdas:
            print(f"λ={lam:.4f}")
            z, pval = self.norm_flow(lam, resid=False)
            print(f"p={pval:.4f}")
            print("---")
            if pval < alpha:  # stop at first failure
                break
            else:
                last_pass = (lam, pval, z)

        if last_pass:
            self.shap_values_z, self.pval, self.lam_final = last_pass[2], last_pass[1], last_pass[0]
            print(f"Selected λ={last_pass[0]:.4f}, p={last_pass[1]:.4f}")
        else:
            raise RuntimeError("Increase complexity of normalizing flow.")

        print("Done!")


    def find_best_lambda_resid(self, alpha=0.05):
        """
        Find largest λ where residuals pass normality test.

        - Always use Shapiro test (residuals are univariate).
        - If already Gaussian, set λ = 1e10 and apply identity flow.
        """
        print("Finding λ for residuals")

        last_pass = None

        # Check normality with Shapiro
        jb_stat, jb_pval = stats.shapiro(self.resid.reshape(-1))
        print(f"Shapiro p-value = {jb_pval:.4f}")
        if jb_pval > alpha:
            lam_null = 1e10
            z, pval = self.norm_flow(lam_null, resid=True)
            print(f"Already normal (Shapiro). Using λ={lam_null}, p={pval:.4f}.")
            last_pass = (lam_null, pval, z)
            self.resid_z, self.pval_resid = last_pass[2], last_pass[1]
            print("Done!")
            return  # stop here

        # Otherwise, search over lambdas
        for lam in self.lambdas_resid:
            print(f"λ={lam:.4f}")
            z, pval = self.norm_flow(lam, resid=True)
            print(f"p={pval:.4f}")
            print("---")
            if pval < alpha:  # stop at first failure
                break
            else:
                last_pass = (lam, pval, z)

        if last_pass:
            self.resid_z, self.pval_resid, self.lam_final_resid, = last_pass[2], last_pass[1], last_pass[0]
            print(f"Selected λ={last_pass[0]:.4f}, p={last_pass[1]:.4f}")
        else:
            raise RuntimeError("Increase complexity of normalizing flow.")

        print("Done!")



    def find_threshold(self, alpha=[0.05, 0.01], bracket=(1e-8, 1000)):
        """
        Compute critical thresholds for IOM values based on χ²-product distribution.

        Maps:
        - α-level → IOM critical value

        Parameters
        ----------
        alpha : list
            Significance levels to compute thresholds for.
        bracket : tuple
            Interval for root finding.

        Returns
        -------
        list
            Critical values corresponding to each α.
        """
        def chi_prod(w, m1, m2=1):
            return (w**((m1 + m2) / 4 - 1)) / \
                   (2**((m1 + m2) / 2 - 1) * gamma(m1 / 2) * gamma(m2 / 2)) * \
                   kv((m1 - m2) / 2, np.sqrt(w))

        def survival_prob(i, m1, m2=1):
            val, _ = quad(chi_prod, i, np.inf, args=(m1, m2))
            return val

        self.i_star = []
        for a in alpha:
            func = lambda i: survival_prob(i, self.p, 1) - a
            self.i_star.append(brentq(func, bracket[0], bracket[1]))

        print(f"Threshold at 0.05: {self.i_star[0]:.4f}")
        print(f"Threshold at 0.01: {self.i_star[1]:.4f}")
        return self.i_star


    def IOM(self):
        """
        Compute Influential Outlier Metric.

        Maps:
        - (latent SHAP values, latent residuals) → IOM score

        Returns
        -------
        pd.Series
            Influential outlier scores.
        """
        self.chi_z = np.diag(
            self.shap_values_z @ np.linalg.inv(np.eye(self.shap_values_z.shape[1])) @ self.shap_values_z.T
        )
        self.chi_resid = self.resid_z.reshape(-1)**2
        self.IOM_ = pd.Series(self.chi_z * self.chi_resid, name="IOM")
        return self.IOM_
    
    def summary(self):
        """
        Print and return a summary of fitted parameters and results.

        Returns
        -------
        pd.DataFrame
            Table with model parameters, λ values, p-values, and thresholds.
        """
        data = {
            "Parameter": [
                "Samples (n)", "Features (p)",
                "Flow depth (SHAP)", "Hidden units (SHAP)", "Layers (SHAP)", 
                "Flow depth (Residuals)", "Hidden units (Residuals)", "Layers (Residuals)",
                "Final λ (SHAP)", "Shap p-value",
                "Final λ (Residuals)", "Residual p-value",
                "Threshold (α=0.05)", "Threshold (α=0.01)"
            ],
            "Value": [
                self.n, self.p,
                self.K, self.hidden, self.layers,
                self.K_resid, self.hidden_resid, self.layers_resid,
                getattr(self, "lam_final", None),
                getattr(self, "pval", None),
                getattr(self, "lam_final_resid", None),
                getattr(self, "pval_resid", None),
                self.i_star[0] if hasattr(self, "i_star") else None,
                self.i_star[1] if hasattr(self, "i_star") else None
            ]
        }
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        return df


# Example usage:

if __name__ == "__main__":
    np.random.seed(42)

    # Simulated SHAP values (100 samples, 5 features)
    shap_values = np.random.randn(100, 5)

    # Simulated residuals
    residuals = np.random.randn(100)

    # Initialize object
    iom = InfluentialOutlierMetric(
        shap_values, residuals,
        K=2, hidden=16, layers=2,
        K_resid=2, hidden_resid=8, layers_resid=2,
        epoch=50, epoch_resid=50  # reduce for speed in demo
        )

    # Fit transformations
    iom.find_best_lambda(alpha=0.05)
    iom.find_best_lambda_resid(alpha=0.05)

    # Compute thresholds
    thresholds = iom.find_threshold(alpha=[0.05, 0.01])

    # Compute IOM
    IOM_scores = iom.IOM()
    print(IOM_scores.sort_values().head(10))