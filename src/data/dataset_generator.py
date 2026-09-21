"""
Synthetic Spatiotemporal Data Cube Generator for Wildfire Forecasting.

Generates realistic 5-day rolling input tensors [T=5, C=7, H=128, W=128]
and ground-truth fire risk maps for 24h, 48h, and 72h horizons based on physical fire dynamics
(wind direction/speed, relative humidity, temperature, fuel load (NDVI), and topography slope).
"""

import os
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class WildfireDataCubeGenerator:
    """
    Simulates physically realistic spatiotemporal data cubes for wildfire risk forecasting.
    Channels (C=7):
      0: active_fire (binary mask)
      1: wind_u (m/s, normalized)
      2: wind_v (m/s, normalized)
      3: humidity (0.0=dry, 1.0=humid)
      4: temperature (0.0=cold, 1.0=hot)
      5: NDVI (0.0=barren, 1.0=dense vegetation fuel)
      6: slope (0.0=flat, 1.0=steep terrain)
    """

    def __init__(self, height: int = 128, width: int = 128, seed: int = 42):
        self.height = height
        self.width = width
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _generate_static_layers(self):
        """Generates static topography (slope) and semi-static vegetation (NDVI)."""
        # Terrain DEM & Slope
        x = np.linspace(-3, 3, self.width)
        y = np.linspace(-3, 3, self.height)
        xx, yy = np.meshgrid(x, y)
        elevation = np.sin(xx) * np.cos(yy) + 0.5 * np.exp(-(xx**2 + yy**2))
        grad_y, grad_x = np.gradient(elevation)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)

        # Vegetation (NDVI) with spatial clustering
        raw_ndvi = gaussian_filter(np.random.randn(self.height, self.width), sigma=12)
        ndvi = (raw_ndvi - raw_ndvi.min()) / (raw_ndvi.max() - raw_ndvi.min() + 1e-8)
        ndvi = 0.2 + 0.7 * ndvi  # Range [0.2, 0.9]

        return slope, ndvi

    def _generate_dynamic_weather(self, num_days: int):
        """Generates time-series weather dynamics (wind_u, wind_v, humidity, temperature)."""
        t_steps = np.arange(num_days)
        # Seasonal/diurnal trend + noise
        base_temp = 0.6 + 0.25 * np.sin(2 * np.pi * t_steps / 30) + 0.05 * np.random.randn(num_days)
        temperature = np.clip(base_temp, 0.1, 0.98)

        # Humidity inversed with temperature
        base_rh = 0.7 - 0.4 * temperature + 0.05 * np.random.randn(num_days)
        humidity = np.clip(base_rh, 0.05, 0.95)

        # Wind vectors (u: east-west, v: north-south)
        wind_u = 0.5 + 0.3 * np.sin(2 * np.pi * t_steps / 15) + 0.1 * np.random.randn(num_days)
        wind_v = 0.3 + 0.3 * np.cos(2 * np.pi * t_steps / 10) + 0.1 * np.random.randn(num_days)

        # Expand spatially with mild spatial turbulence
        weather_cubes = {
            'wind_u': np.zeros((num_days, self.height, self.width)),
            'wind_v': np.zeros((num_days, self.height, self.width)),
            'humidity': np.zeros((num_days, self.height, self.width)),
            'temperature': np.zeros((num_days, self.height, self.width)),
        }

        for t in range(num_days):
            turb_u = gaussian_filter(np.random.randn(self.height, self.width), sigma=8) * 0.1
            turb_v = gaussian_filter(np.random.randn(self.height, self.width), sigma=8) * 0.1
            turb_h = gaussian_filter(np.random.randn(self.height, self.width), sigma=8) * 0.05
            turb_t = gaussian_filter(np.random.randn(self.height, self.width), sigma=8) * 0.05

            weather_cubes['wind_u'][t] = np.clip(wind_u[t] + turb_u, -1.0, 1.0)
            weather_cubes['wind_v'][t] = np.clip(wind_v[t] + turb_v, -1.0, 1.0)
            weather_cubes['humidity'][t] = np.clip(humidity[t] + turb_h, 0.01, 1.0)
            weather_cubes['temperature'][t] = np.clip(temperature[t] + turb_t, 0.01, 1.0)

        return weather_cubes

    def simulate_sequence(self, num_days: int = 40, num_ignition_points: int = 4):
        """
        Simulates full time series sequence of (C=7) data layers and physical fire propagation labels.
        Returns:
            data_cubes: shape [num_days, 7, H, W]
        """
        slope, ndvi = self._generate_static_layers()
        weather = self._generate_dynamic_weather(num_days)

        fire_history = np.zeros((num_days, self.height, self.width), dtype=np.float32)

        # Place initial ignition spots at day 0 and day 10
        for _ in range(num_ignition_points):
            init_day = np.random.randint(0, min(10, num_days))
            cy = np.random.randint(20, self.height - 20)
            cx = np.random.randint(20, self.width - 20)
            fire_history[init_day, cy-2:cy+3, cx-2:cx+3] = 1.0

        # Physical propagation simulation loop over time
        for t in range(1, num_days):
            prev_fire = fire_history[t - 1]
            if prev_fire.max() == 0:
                # Occasional random lightning ignition if no active fires
                cy = np.random.randint(20, self.height - 20)
                cx = np.random.randint(20, self.width - 20)
                fire_history[t, cy-1:cy+2, cx-1:cx+2] = 1.0
                continue

            wu = weather['wind_u'][t]
            wv = weather['wind_v'][t]
            hum = weather['humidity'][t]
            temp = weather['temperature'][t]

            # Environmental risk factor calculation
            # Risk increases with high temp, high ndvi (fuel), high slope, low humidity
            env_factor = (temp * 0.3 + ndvi * 0.3 + slope * 0.2 + (1.0 - hum) * 0.2)

            # Advection/spread direction by shifting existing fire by wind vector
            shift_y = int(np.round(wv.mean() * 2))
            shift_x = int(np.round(wu.mean() * 2))

            shifted_fire = np.roll(prev_fire, shift_y, axis=0)
            shifted_fire = np.roll(shifted_fire, shift_x, axis=1)

            # Spatial diffusion + spread
            spread_prob = gaussian_filter(prev_fire + 0.7 * shifted_fire, sigma=1.5) * env_factor
            new_fire = np.where(spread_prob > 0.18, 1.0, 0.0)

            # Combine with previous fire (burnt area stays burnt or active)
            fire_history[t] = np.clip(prev_fire * 0.6 + new_fire * 1.0, 0.0, 1.0)

        # Construct full 7-channel array per day
        # Channels: [fire, wind_u, wind_v, humidity, temperature, NDVI, slope]
        full_series = np.zeros((num_days, 7, self.height, self.width), dtype=np.float32)
        for t in range(num_days):
            full_series[t, 0] = fire_history[t]
            full_series[t, 1] = weather['wind_u'][t]
            full_series[t, 2] = weather['wind_v'][t]
            full_series[t, 3] = weather['humidity'][t]
            full_series[t, 4] = weather['temperature'][t]
            full_series[t, 5] = ndvi
            full_series[t, 6] = slope

        return full_series

    def create_dataset_samples(self, sequence_length: int = 5, num_days: int = 40):
        """
        Extracts sliding window samples.
        Input X: [T=5, C=7, H=128, W=128]
        Target Y: [3, H=128, W=128] representing fire risk at +24h (t+1), +48h (t+2), +72h (t+3)
        """
        full_series = self.simulate_sequence(num_days=num_days)

        X_samples = []
        Y_samples = []

        # Sliding window
        # For sequence length 5, target needs 3 steps ahead -> max index = num_days - 3
        for t in range(sequence_length, num_days - 3):
            x_window = full_series[t - sequence_length:t]  # [5, 7, H, W]
            y_horizon = np.stack([
                full_series[t, 0],       # +24h target (fire mask)
                full_series[t + 1, 0],   # +48h target
                full_series[t + 2, 0],   # +72h target
            ], axis=0)  # [3, H, W]

            X_samples.append(x_window)
            Y_samples.append(y_horizon)

        X_tensor = torch.tensor(np.array(X_samples), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y_samples), dtype=torch.float32)

        return X_tensor, Y_tensor


if __name__ == "__main__":
    generator = WildfireDataCubeGenerator()
    X, Y = generator.create_dataset_samples(num_days=30)
    print(f"Generated X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Positive fire pixels in Y (+24h): {(Y[:, 0] > 0.5).sum().item()} / {Y[:, 0].numel()}")
