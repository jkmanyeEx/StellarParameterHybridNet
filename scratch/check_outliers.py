import numpy as np
import os

BASE_DIR = "/Users/devmeko/Documents/KSA/3rdSem/GenAstro/TermProject/TermProject"
MASTAR_FLUX = os.path.join(BASE_DIR, "data", "processed", "X_flux_telluric.npy")

if os.path.exists(MASTAR_FLUX):
    flux = np.load(MASTAR_FLUX)
    print(f"Total spectra: {flux.shape[0]}")
    
    # Check max and min per spectrum
    maxs = np.max(flux, axis=1)
    mins = np.min(flux, axis=1)
    
    outliers_max = np.sum(maxs > 10.0)
    outliers_min = np.sum(mins < -2.0)
    outliers_nan = np.sum(np.isnan(flux))
    outliers_inf = np.sum(np.isinf(flux))
    
    print(f"Spectra with max > 10.0: {outliers_max}")
    print(f"Spectra with min < -2.0: {outliers_min}")
    print(f"NaNs: {outliers_nan}")
    print(f"Infs: {outliers_inf}")
    
    # Let's see some example outlier values
    bad_idx = np.where((maxs > 10.0) | (mins < -2.0))[0]
    if len(bad_idx) > 0:
        print(f"First 5 bad indices: {bad_idx[:5]}")
        print(f"Max values of first 5 bad: {maxs[bad_idx[:5]]}")
        print(f"Min values of first 5 bad: {mins[bad_idx[:5]]}")
else:
    print("File not found")
