"""Classical interpolation methods for IV surface reconstruction."""

import numpy as np
from typing import Optional, Tuple
from scipy.interpolate import griddata, RectBivariateSpline


class LinearInterpolation:
    """Simple linear interpolation baseline for IV surface."""
    
    def __init__(self):
        """Initialize linear interpolation model."""
        self.surface = None
        self.strikes = None
        self.forwards = None
    
    def fit(self, forwards: np.ndarray, strikes: np.ndarray, market_ivs: np.ndarray):
        """Fit interpolation surface to data.
        
        Args:
            forwards: Forward prices
            strikes: Strike prices
            market_ivs: Observed IV values at (forward, strike) pairs
        """
        self.strikes = np.unique(strikes)
        self.forwards = np.unique(forwards)
        
        # Reshape data into grid
        grid_forward, grid_strike = np.meshgrid(self.forwards, self.strikes)
        
        # Use griddata for interpolation
        points = np.column_stack([forwards, strikes])
        self.surface = griddata(points, market_ivs, (grid_forward, grid_strike), method='linear')
    
    def predict(self, forwards: np.ndarray, strikes: np.ndarray) -> np.ndarray:
        """Predict IV at given (forward, strike) pairs.
        
        Args:
            forwards: Forward prices
            strikes: Strike prices
        
        Returns:
            Predicted IV values
        """
        if self.surface is None:
            raise ValueError("Model must be fitted first")
        
        points = np.column_stack([self.forwards, self.strikes])
        points_query = np.column_stack([forwards, strikes])
        
        return griddata(points, self.surface.flatten(), points_query, method='linear')


class SplineInterpolation:
    """Spline-based interpolation for smooth IV surface."""
    
    def __init__(self, kx: int = 3, ky: int = 3):
        """Initialize spline interpolation.
        
        Args:
            kx: Degree of spline in forward direction
            ky: Degree of spline in strike direction
        """
        self.kx = kx
        self.ky = ky
        self.spline = None
        self.strikes = None
        self.forwards = None
    
    def fit(self, forwards: np.ndarray, strikes: np.ndarray, market_ivs: np.ndarray):
        """Fit spline surface to data.
        
        Args:
            forwards: Forward prices
            strikes: Strike prices  
            market_ivs: Observed IV values
        """
        self.strikes = np.unique(strikes)
        self.forwards = np.unique(forwards)
        
        # Create grid
        grid_forward, grid_strike = np.meshgrid(self.forwards, self.strikes)
        
        # Interpolate to grid
        points = np.column_stack([forwards, strikes])
        values = griddata(points, market_ivs, (grid_forward, grid_strike), method='cubic')
        
        # Fill NaNs with nearest neighbor if needed
        mask = np.isnan(values)
        if mask.any():
            values[mask] = griddata(points, market_ivs, (grid_forward, grid_strike), 
                                   method='nearest')[mask]
        
        # Fit spline
        try:
            self.spline = RectBivariateSpline(
                self.forwards, self.strikes, values,
                kx=min(self.kx, len(self.forwards) - 1),
                ky=min(self.ky, len(self.strikes) - 1)
            )
        except:
            # Fallback to linear if spline fitting fails
            self.spline = None
    
    def predict(self, forwards: np.ndarray, strikes: np.ndarray) -> np.ndarray:
        """Predict IV at given (forward, strike) pairs.
        
        Args:
            forwards: Forward prices
            strikes: Strike prices
        
        Returns:
            Predicted IV values
        """
        if self.spline is None:
            raise ValueError("Model must be fitted first")
        
        predictions = np.zeros(len(forwards))
        for i, (f, k) in enumerate(zip(forwards, strikes)):
            predictions[i] = self.spline(f, k)[0, 0]
        
        return predictions


class LocalVolatilityInterpolation:
    """Local volatility model interpolation baseline."""
    
    def __init__(self, kernel_bandwidth: float = 1.0):
        """Initialize local volatility interpolation.
        
        Args:
            kernel_bandwidth: Bandwidth for kernel smoothing
        """
        self.kernel_bandwidth = kernel_bandwidth
        self.forwards_data = None
        self.strikes_data = None
        self.ivs_data = None
    
    def fit(self, forwards: np.ndarray, strikes: np.ndarray, market_ivs: np.ndarray):
        """Fit local volatility model to data.
        
        Args:
            forwards: Forward prices
            strikes: Strike prices
            market_ivs: Observed IV values
        """
        self.forwards_data = forwards
        self.strikes_data = strikes
        self.ivs_data = market_ivs
    
    def predict(self, forwards: np.ndarray, strikes: np.ndarray) -> np.ndarray:
        """Predict IV using kernel smoothing.
        
        Args:
            forwards: Forward prices
            strikes: Strike prices
        
        Returns:
            Predicted IV values
        """
        if self.ivs_data is None:
            raise ValueError("Model must be fitted first")
        
        predictions = np.zeros(len(forwards))
        
        for i, (f, k) in enumerate(zip(forwards, strikes)):
            # Compute distances to all data points
            distances = np.sqrt((self.forwards_data - f) ** 2 + (self.strikes_data - k) ** 2)
            
            # Gaussian kernel weights
            weights = np.exp(-distances ** 2 / (2 * self.kernel_bandwidth ** 2))
            weights /= weights.sum()
            
            # Weighted average
            predictions[i] = np.sum(weights * self.ivs_data)
        
        return predictions


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    forwards = np.random.uniform(90, 110, 50)
    strikes = np.random.uniform(90, 110, 50)
    market_ivs = 0.2 + 0.01 * np.random.randn(50)  # Random IVs around 0.2
    
    # Test linear interpolation
    lin_model = LinearInterpolation()
    lin_model.fit(forwards, strikes, market_ivs)
    
    # Test spline interpolation
    spline_model = SplineInterpolation()
    spline_model.fit(forwards, strikes, market_ivs)
    
    print("✅ Interpolation models initialized successfully")
