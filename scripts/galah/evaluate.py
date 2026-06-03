import os
import sys

# Add project root to python path for modular import resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.validation.galah.error_calculator import run_real_bulk_evaluation

if __name__ == "__main__":
    run_real_bulk_evaluation()
