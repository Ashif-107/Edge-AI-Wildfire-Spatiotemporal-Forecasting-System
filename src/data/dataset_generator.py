"""
Synthetic Spatiotemporal Data Cube Generator for Wildfire Forecasting.

Generates realistic 5-day rolling input tensors [T=5, C=7, H=128, W=128]
and ground-truth fire risk maps for 24h, 48h, and 72h horizons.

Key physical properties:
 - Fire ignites in small localized patches (2-4 points)
 - Fire spreads with wind direction, slope gradient, low humidity, and high NDVI
 - Burnt areas cool down over time (extinction model) — prevents all-fire saturation
 - Multiple short independent fire episodes keep the dataset challenging and sparse
"""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


class WildfireDataCubeGenerator:
    """
    Simulates physically realistic spatiotemporal data cubes for wildfire risk forecasting.

    Channels (C=7):
      0: active_fire   — binary mask (1.0 = active fire front, 0.0 = safe)
      1: wind_u        — ERA5-Land East-West wind velocity component (m/s, normalized)
      2: wind_v        — ERA5-Land North-South wind velocity component (m/s, normalized)
      3: humidity      — ERA5-Land relative humidity (0.0 = severe drought, 1.0 = humid)
      4: temperature   — ERA5-Land surface temperature (0.0 = cold, 1.0 = hot)
      5: NDVI          — MODIS MOD13A2 vegetation fuel density (0.0 = barren, 1.0 = dense)
      6: slope         — USGS SRTM terrain slope gradient (0.0 = flat, 1.0 = steep)
    """

    def __init__(self, height: int = 128, width: int = 128, seed: int = 42):
        self.height = height
        self.width  = width
        self.seed   = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _generate_static_layers(self):
        """Generates static topography (slope) and semi-static vegetation (NDVI)."""
        x = np.linspace(-3, 3, self.width)
        y = np.linspace(-3, 3, self.height)
        xx, yy = np.meshgrid(x, y)
        elevation = np.sin(xx) * np.cos(yy) + 0.5 * np.exp(-(xx**2 + yy**2))
        grad_y, grad_x = np.gradient(elevation)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        slope = (slope - slope.min()) / (slope.max() - slope.min() + 1e-8)

        # NDVI: spatially clustered vegetation with random seed variation
        raw_ndvi = gaussian_filter(np.random.randn(self.height, self.width), sigma=14)
        ndvi = (raw_ndvi - raw_ndvi.min()) / (raw_ndvi.max() - raw_ndvi.min() + 1e-8)
        ndvi = 0.2 + 0.65 * ndvi  # Keep in range [0.2, 0.85]

        return slope.astype(np.float32), ndvi.astype(np.float32)

    def _generate_dynamic_weather(self, num_days: int):
        """Generates spatially-varying time-series weather with realistic diurnal + synoptic patterns."""
        t_steps = np.arange(num_days)
        base_temp = 0.55 + 0.20 * np.sin(2 * np.pi * t_steps / 20) + 0.08 * np.random.randn(num_days)
        temperature = np.clip(base_temp, 0.1, 0.92)

        base_rh = 0.75 - 0.45 * temperature + 0.06 * np.random.randn(num_days)
        humidity = np.clip(base_rh, 0.05, 0.92)

        wind_u = 0.4 + 0.25 * np.sin(2 * np.pi * t_steps / 12) + 0.12 * np.random.randn(num_days)
        wind_v = 0.25 + 0.20 * np.cos(2 * np.pi * t_steps / 9) + 0.10 * np.random.randn(num_days)

        cubes = {k: np.zeros((num_days, self.height, self.width), dtype=np.float32)
                 for k in ('wind_u', 'wind_v', 'humidity', 'temperature')}

        for t in range(num_days):
            for key, base, sigma, clip_range in [
                ('wind_u',      wind_u[t],      7, (-1.0, 1.0)),
                ('wind_v',      wind_v[t],      7, (-1.0, 1.0)),
                ('humidity',    humidity[t],    9, (0.02, 0.98)),
                ('temperature', temperature[t], 9, (0.02, 0.98)),
            ]:
                turb = gaussian_filter(np.random.randn(self.height, self.width), sigma=sigma) * 0.08
                cubes[key][t] = np.clip(base + turb, *clip_range)

        return cubes

    def simulate_sequence(self, num_days: int = 35):
        """
        Simulates a fire season time series with:
          - Multiple short independent fire episodes (each episode resets after extinguishing)
          - Extinction: active fire front cools off after ~5 days without spreading
          - This keeps fire pixel ratio LOW and spatially localized (realistic class imbalance)

        Returns:
            full_series: shape [num_days, 7, H, W]
        """
        slope, ndvi = self._generate_static_layers()
        weather = self._generate_dynamic_weather(num_days)

        fire_history   = np.zeros((num_days, self.height, self.width), dtype=np.float32)
        burnt_map      = np.zeros((self.height, self.width), dtype=np.float32)  # Burnt = no fuel
        active_fire    = np.zeros((self.height, self.width), dtype=np.float32)
        days_since_spread = 0  # Track stagnation for extinction

        # Random ignition schedule: 3–5 independent episodes spread across the season
        num_episodes = np.random.randint(3, 6)
        episode_days = sorted(np.random.choice(np.arange(2, num_days - 8), num_episodes, replace=False))

        for t in range(num_days):
            # Fresh ignition event
            if t in episode_days:
                cy = np.random.randint(20, self.height - 20)
                cx = np.random.randint(20, self.width  - 20)
                patch_r = np.random.randint(4, 9)  # Larger initial ignition
                active_fire[cy-patch_r:cy+patch_r, cx-patch_r:cx+patch_r] = 1.0
                burnt_map[cy-patch_r:cy+patch_r, cx-patch_r:cx+patch_r] = 0.0
                days_since_spread = 0

            # Physics-based spread step
            wu  = weather['wind_u'][t]
            wv  = weather['wind_v'][t]
            hum = weather['humidity'][t]
            tmp = weather['temperature'][t]

            # Available fuel: NDVI minus already-burnt areas
            fuel = np.clip(ndvi - burnt_map, 0.0, 1.0)

            # Environmental risk:  high temp, high fuel, steep slope, dry conditions -> high risk
            env_factor = (tmp * 0.30 + fuel * 0.35 + slope * 0.15 + (1.0 - hum) * 0.20) * 1.4

            # Directional advection (fire front carried by wind)
            shift_y = int(np.round(np.clip(wv.mean() * 1.5, -3, 3)))
            shift_x = int(np.round(np.clip(wu.mean() * 1.5, -3, 3)))
            shifted = np.roll(np.roll(active_fire, shift_y, axis=0), shift_x, axis=1)

            # Diffusion probability
            spread_prob = gaussian_filter(active_fire * 0.55 + shifted * 0.45, sigma=1.2) * env_factor

            # New fire ignites where probability exceeds threshold
            new_fire = (spread_prob > 0.14).astype(np.float32)

            # Extinction: active pixels that haven't spread in >6 steps cool down
            if active_fire.max() > 0:
                prev_area = (active_fire > 0.5).sum()
                curr_area = (new_fire > 0.5).sum()
                if curr_area <= prev_area * 1.02:
                    days_since_spread += 1
                else:
                    days_since_spread = 0

            # After 8 days of stagnation -> extinguish fire (burnt, no longer active)
            if days_since_spread >= 8:
                burnt_map = np.clip(burnt_map + active_fire * 0.85, 0.0, 1.0)
                active_fire = np.zeros_like(active_fire)
                days_since_spread = 0

            # Update active fire: new spread minus already-burnt areas
            active_fire = np.clip(new_fire * (1.0 - burnt_map), 0.0, 1.0)

            # Record current active fire front as the fire channel at time t
            fire_history[t] = active_fire.copy()

        # Assemble 7-channel time series
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

    def create_dataset_samples(self, sequence_length: int = 5, num_days: int = 35):
        """
        Extracts sliding-window samples from multiple independent fire season simulations.
        Combines multiple simulated episodes (different seeds) for dataset diversity.

        Input X:  [T=5, C=7, H=128, W=128]
        Target Y: [3, H=128, W=128]  — fire risk at +24h, +48h, +72h
        """
        X_all, Y_all = [], []

        # Run multiple independent simulations with different seeds for diversity
        seeds = [self.seed, self.seed + 7, self.seed + 13, self.seed + 31]
        days_per_sim = max(30, num_days // len(seeds))

        for sim_seed in seeds:
            np.random.seed(sim_seed)
            full_series = self.simulate_sequence(num_days=days_per_sim)

            for t in range(sequence_length, days_per_sim - 3):
                x_win = full_series[t - sequence_length:t]          # [5, 7, H, W]
                y_hor = np.stack([
                    (full_series[t,     0] > 0.5).astype(np.float32),  # Binary +24h
                    (full_series[t + 1, 0] > 0.5).astype(np.float32),  # Binary +48h
                    (full_series[t + 2, 0] > 0.5).astype(np.float32),  # Binary +72h
                ], axis=0)

                # Skip trivially empty frames (no fire in target or input)
                if (full_series[t - sequence_length:t, 0] > 0.5).sum() == 0:
                    continue

                X_all.append(x_win)
                Y_all.append(y_hor)

        # Reset generator seed
        np.random.seed(self.seed)

        X_tensor = torch.tensor(np.array(X_all), dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y_all), dtype=torch.float32)

        return X_tensor, Y_tensor


if __name__ == "__main__":
    gen = WildfireDataCubeGenerator(seed=42)
    X, Y = gen.create_dataset_samples(sequence_length=5, num_days=35)
    fire_ratio = (Y > 0.5).float().mean().item()
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Fire pixel ratio: {fire_ratio*100:.2f}%  (target: < 15%)")
    print(f"Min fire pixels in any sample: {(Y.reshape(len(Y),-1) > 0.5).sum(dim=1).min().item()}")
    print(f"Max fire pixels in any sample: {(Y.reshape(len(Y),-1) > 0.5).sum(dim=1).max().item()}")
