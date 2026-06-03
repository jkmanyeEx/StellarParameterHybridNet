import os
import sys

# Add project root to python path for modular import resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.validation.galah.xai_analyzer import run_xai_line_profile_analysis

if __name__ == "__main__":
    run_xai_line_profile_analysis()
