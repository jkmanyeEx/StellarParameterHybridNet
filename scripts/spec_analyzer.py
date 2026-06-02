import os
from astropy.io import fits

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_dir = os.path.join(BASE_DIR, "data", "validation_dataset")
path = os.path.join(dataset_dir, "spec-3615-55179-0010.fits")

if not os.path.exists(path):
    # Try finding any fits file
    cands = [f for f in os.listdir(dataset_dir) if f.endswith(".fits")] if os.path.isdir(dataset_dir) else []
    if cands:
        path = os.path.join(dataset_dir, cands[0])

if os.path.exists(path):
    print(f"Analyzing: {path}")
    with fits.open(path) as hdul:
        hdul.info()
        print("\n--- HDU 0 Header keys ---")
        for k, v in hdul[0].header.items():
            if any(x in k.upper() for x in ['RA', 'DEC', 'PLUG', 'WAVE', 'FLUX']):
                print(f"  {k}: {v}")
        for i, hdu in enumerate(hdul):
            if hdu.data is not None:
                print(f"\n--- HDU {i}: {type(hdu).__name__} ---")
                if hasattr(hdu, 'columns'):
                    print("  Columns:", hdu.columns.names)
                else:
                    print("  Shape:", hdu.data.shape)
else:
    print(f"No spec FITS file found for analysis in {dataset_dir}")
