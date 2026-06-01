"""SABR (Stochastic Alpha-Beta-Rho) stochastic volatility model baseline."""

import numpy as np
from typing import Tuple, Optional
from scipy.optimize import minimize


class SABRModel:
    """SABR stochastic volatility model for implied volatility surface.
    
    SABR is a stochastic volatility model that captures the dynamics of 
    implied volatility surfaces by modeling:
    - Forward rate with constant elasticity of variance (CEV)
    - Stochastic volatility of the forward
    
    References:
        - Hagan et al. "Managing Smile Risk" (2002)
        - See src/assets/SABR_Model.pdf for detailed documentation
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.7, rho: float = 0.3, nu: float = 0.2):
        """Initialize SABR model parameters.
        
        Args:
            alpha: Initial volatility (α > 0)
            beta: Elasticity of variance (0 < β ≤ 1)
                  β=1: Lognormal, β=0: Normal
            rho: Correlation between forward and volatility (-1 < ρ < 1)
            nu: Volatility of volatility (ν > 0)
        """
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
    
    def implied_vol_hagan(self, F: float, K: float, T: float) -> float:
        """Compute implied volatility using Hagan's approximation.
        
        Hagan et al. (2002) provided a closed-form approximation for SABR
        implied volatility that is accurate for most practical cases.
        
        Args:
            F: Forward rate (current underlying)
            K: Strike price
            T: Time to expiry (in years)
        
        Returns:
            Implied volatility (annualized)
        """
        if abs(F - K) < 1e-10:
            # ATM case
            return self.alpha * (1 + (self.beta ** 2 / 24 * np.log(F / K) ** 2) +
                                (self.rho * self.beta * self.nu / 4 * self.alpha) +
                                ((2 - 3 * self.rho ** 2) / 24 * self.nu ** 2) * T) / F ** (1 - self.beta)
        
        # OTM/ITM case
        z = (self.nu / self.alpha) * F ** (1 - self.beta) * np.log(F / K)
        x_z = np.log((np.sqrt(1 - 2 * self.rho * z + z ** 2) + z - self.rho) / (1 - self.rho))
        
        numerator = self.alpha * (1 + ((1 - self.beta) ** 2 / 24 * (np.log(F / K) ** 2) +
                                       (1 - self.beta) ** 4 / 1920 * (np.log(F / K) ** 4)))
        denominator = F ** (1 - self.beta) * (1 + ((1 - self.beta) ** 2 / 24 * self.alpha ** 2 / F ** (2 - 2 * self.beta)) +
                                             (self.rho * self.beta * self.nu / 4 * self.alpha / F ** (1 - self.beta)) +
                                             ((2 - 3 * self.rho ** 2) / 24 * self.nu ** 2))
        
        if abs(x_z) < 1e-10:
            return numerator / denominator
        
        return numerator * x_z / (denominator * z)
    
    def predict_surface(self, forwards: np.ndarray, strikes: np.ndarray, T: float) -> np.ndarray:
        """Predict IV surface for grid of forwards and strikes.
        
        Args:
            forwards: Array of forward rates
            strikes: Array of strikes
            T: Time to expiry
        
        Returns:
            IV surface of shape (len(forwards), len(strikes))
        """
        surface = np.zeros((len(forwards), len(strikes)))
        for i, F in enumerate(forwards):
            for j, K in enumerate(strikes):
                surface[i, j] = self.implied_vol_hagan(F, K, T)
        return surface
    
    def calibrate(self, forwards: np.ndarray, strikes: np.ndarray, 
                  market_ivs: np.ndarray, T: float) -> dict:
        """Calibrate SABR parameters to market IV data.
        
        Args:
            forwards: Observed forward rates
            strikes: Strike prices
            market_ivs: Observed market implied volatilities
            T: Time to expiry
        
        Returns:
            Dictionary with calibration results
        """
        # Objective function: minimize MSE between model and market IVs
        def loss(params):
            alpha, beta, rho, nu = params
            alpha = max(alpha, 1e-4)
            beta = np.clip(beta, 1e-4, 1.0)
            rho = np.clip(rho, -0.999, 0.999)
            nu = max(nu, 1e-4)
            
            model_ivs = np.zeros_like(market_ivs)
            for i, F in enumerate(forwards):
                for j, K in enumerate(strikes):
                    try:
                        model_ivs[i, j] = self.implied_vol_hagan(F, K, T)
                    except:
                        return 1e10
            
            return np.mean((model_ivs - market_ivs) ** 2)
        
        # Optimization
        result = minimize(
            loss,
            x0=[self.alpha, self.beta, self.rho, self.nu],
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        self.alpha, self.beta, self.rho, self.nu = result.x
        
        return {
            'success': result.success,
            'loss': result.fun,
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'nu': self.nu,
            'iterations': result.nit
        }


if __name__ == "__main__":
    # Example usage
    model = SABRModel(alpha=0.5, beta=0.7, rho=0.3, nu=0.2)
    
    # Sample IV at ATM
    iv_atm = model.implied_vol_hagan(F=100, K=100, T=0.25)
    print(f"SABR IV (ATM): {iv_atm:.4f}")
    
    # Sample IV at OTM
    iv_otm = model.implied_vol_hagan(F=100, K=110, T=0.25)
    print(f"SABR IV (OTM): {iv_otm:.4f}")
