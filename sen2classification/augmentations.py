import numpy as np


def augment_boa_and_time(
    boa,
    time,
    doy_encoding,
    mean,
    stddev,
    p_random_noise=0.5,
    p_constant_offset=0.8,
    p_time_jitter=0.5,
    p_time_dependent_noise=0.8,
    p_blackout=0.02,
    p_gamma=0.8,
    p_cloud_simulation=0.02,
    p_cloud_shadow=0.02,
    p_observation_dropout=0.2,
    noise_scale=0.02,
    offset_scale=0.04,
    time_jitter_max=4,
    blackout_percentage=0.02,
    dropout_percentage=0.05,
    time_noise_strength=0.02,
    gamma_offset=0.002,
    rng=None
):
    """
    Augment BOA and time data with various transformations.
    
    Args:
        boa: Array of unnormalized BOA values with shape (max_seq_len, channels)
        time: Array of time values with shape (max_seq_len)
        doy_encoding: Whether time values are day-of-year encoded
        mean: Channel-wise mean values used for normalization.
        stddev: Channel-wise standard deviation values used for normalization.
        p_random_noise: Probability of applying random noise (default: 0.5)
        p_constant_offset: Probability of applying constant band-wise offset (default: 0.8)
        p_time_jitter: Probability of applying time jitter (default: 0.5)
        p_time_dependent_noise: Probability of applying time-dependent noise (default: 0.8)
        p_blackout: Probability of applying random blackouts (default: 0.02)
        p_gamma: Probability of applying gamma correction (default: 0.8)
        p_cloud_simulation: Probability of applying cloud simulation (default: 0.02)
        p_cloud_shadow: Probability of simulating cloud shadow (default: 0.02)
        p_observation_dropout: Probability of applying observation dropout (default: 0.2)
        noise_scale: Scale of random noise (default: 0.02)
        offset_scale: Scale of band-wise offsets (default: 0.04)
        time_jitter_max: Maximum time jitter in days (default: 4)
        blackout_percentage: Percentage of time steps to black out (default: 0.02)
        dropout_percentage: Percentage of observations to drop (default: 0.05)
        time_noise_strength: Strength factor for time-dependent noise (default: 0.02)
        gamma_offset: Maximum deviation from neutral gamma (1.0). Gamma will be in range [1-offset, 1+offset] (default: 0.002)
        rng: Random number generator. If None, a new one will be created.
    
    Augmentation Order:
        Before normalization:
        - Random blackouts
        - Cloud simulation
        - Cloud shadow
        - Gamma correction
        
        After normalization:
        - Random noise
        - Constant band-wise offsets
        - Time jitter
        - Time-dependent noise
        - Observation dropout
    
    Returns:
        Tuple of (augmented_boa, augmented_time)
    """
    # Use provided RNG or create a new one
    if rng is None:
        rng = np.random.default_rng()
    
    # Make copies to avoid modifying the originals
    boa_aug = boa.copy()
    time_aug = time.copy()
    seq_len = len(time)
    new_seq_len = seq_len

    # ~~ Augmentations that work better on unnormalized data ~~ #

    # Apply random blackouts
    if rng.random() < p_blackout:
        # Calculate number of time steps to black out
        num_blackout = max(1, int(seq_len * blackout_percentage))
        # Randomly select time steps to black out
        blackout_indices = rng.choice(seq_len, num_blackout, replace=False)
        boa_aug[blackout_indices] = -9999
    
    # Apply cloud simulation
    if rng.random() < p_cloud_simulation:
        # Simulate thin clouds by increasing brightness and reducing contrast in visible bands
        # Typically affects visible bands more than NIR/SWIR
        cloud_indices = rng.choice(seq_len, max(1, int(seq_len * 0.02)), replace=False)
        
        # Visible bands (blue, green, red) are generally more affected
        visible_bands = slice(3)  # Assuming bands 0, 1, 2 are blue, green, red
        nir_swir_bands = slice(3,10)  # Remaining bands
        
        # Add positive offset to simulate increased brightness from clouds
        # Use appropriate values for unnormalized reflectance
        cloud_brightness = rng.uniform(200, 500)
        boa_aug[cloud_indices, visible_bands] += cloud_brightness
        
        # Add smaller offset to NIR/SWIR bands
        cloud_brightness_nir = cloud_brightness * 0.3
        boa_aug[cloud_indices, nir_swir_bands] += cloud_brightness_nir
        
        # Reduce contrast by scaling values toward the mean
        contrast_factor = rng.uniform(0.6, 0.9)
        mean_values = np.mean(boa_aug[cloud_indices], axis=0, keepdims=True)
        boa_aug[cloud_indices, :] = contrast_factor * (boa_aug[cloud_indices] - mean_values) + mean_values

    if rng.random() < p_cloud_shadow:
        cloud_indices = rng.choice(seq_len, max(1, int(seq_len * 0.02)), replace=False)
        boa_aug[cloud_indices] *= 0.4

    # Apply random gamma correction
    if rng.random() < p_gamma:
        # Apply gamma correction only to positive values
        # Use gamma_offset to control the range: 1.0 ± gamma_offset
        gamma = rng.uniform(1.0 - gamma_offset, 1.0 + gamma_offset)
        pos_mask = boa_aug > 0
        boa_aug[pos_mask] = boa_aug[pos_mask] ** gamma

    # Normalize
    inv_stddev = 1 / (stddev + 1e-7)
    boa_aug = (boa_aug - mean) * inv_stddev

    # ~~ Augmentations that work better on normalized data ~~ #

    # Apply random noise
    if rng.random() < p_random_noise:
        noise = rng.normal(0, noise_scale, boa_aug.shape)
        boa_aug += noise
    
    # Apply random constant offsets (band-specific)
    if rng.random() < p_constant_offset:
        offsets = rng.uniform(-offset_scale, offset_scale, (1, boa_aug.shape[1]))
        boa_aug += offsets
    
    # Apply time jitter
    if rng.random() < p_time_jitter:
        jitter = rng.integers(-time_jitter_max, time_jitter_max + 1, seq_len)
        time_aug += jitter
        if doy_encoding:
            time_aug = np.clip(time_aug, 0, 366)
        else:
            time_aug = np.clip(time_aug, 0, 10*366)
        sort_indices = np.argsort(time_aug)
        time_aug = time_aug[sort_indices]
        boa_aug[:seq_len, :] = boa_aug[sort_indices, :]
    
    # Apply time-dependent noise (noise that varies with time)
    if rng.random() < p_time_dependent_noise and doy_encoding:
        # times are encoded as doy so between 0 and 366
        # Use actual time values (normalized to [0, 1]) for generating noise
        # First get the time values for the sequence
        actual_times = time_aug.copy()
        
        # Normalize times to [0, 2π] for sinusoidal functions
        t_normalized = 2 * np.pi * actual_times / 366
        
        # Create different frequency components using actual time values
        f0 = np.sin(0.5 * t_normalized)
        f1 = np.sin(t_normalized)
        f2 = np.sin(2 * t_normalized)
        f3 = np.sin(4 * t_normalized)
        
        # Combine with random weights scaled by time_noise_strength
        weights0 = rng.uniform(-time_noise_strength, time_noise_strength, 1)
        weights1 = rng.uniform(-time_noise_strength, time_noise_strength, 1)
        weights2 = rng.uniform(-time_noise_strength, time_noise_strength, 1)
        weights3 = rng.uniform(-time_noise_strength, time_noise_strength, 1)

        time_noise = f0 * weights0 + f1 * weights1 + f2 * weights2 + f3 * weights3
        boa_aug += time_noise[:, None]

    # Apply observation dropout (completely remove observations)
    if rng.random() < p_observation_dropout:
        if seq_len > 12:
            # Calculate number of observations to drop
            num_dropout = max(1, int(seq_len * dropout_percentage))
            # Randomly select observations to drop
            dropout_indices = rng.choice(seq_len, num_dropout, replace=False)

            # Move all data after dropout positions forward
            for idx in sorted(dropout_indices):
                if idx < seq_len - 1:
                    # Shift all subsequent observations one position forward
                    boa_aug[idx:seq_len-1] = boa_aug[idx+1:seq_len]
                    time_aug[idx:seq_len-1] = time_aug[idx+1:seq_len]

                # Set the last positions to zeros or default values
                boa_aug[seq_len-1] = 0
                time_aug[seq_len-1] = 0

            new_seq_len = seq_len - num_dropout

    return boa_aug[:new_seq_len], time_aug[:new_seq_len]
