import os
import sys

# Add project root to python path for modular import resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.validation.mastar.error_calculation_mastar import run_mastar_bulk_evaluation

if __name__ == "__main__":
    run_mastar_bulk_evaluation()
