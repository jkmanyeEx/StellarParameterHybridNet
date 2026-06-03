import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.apogee.engine import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train APOGEE StellarParameterHybridNet"
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from the best saved checkpoint'
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        metavar='N',
        help=(
            'Use only the first N valid stars for training + val + test. '
            'Useful for quick data-scale comparisons '
            '(e.g. --limit 15000 vs --limit 50000). '
            'Default: use all available stars.'
        )
    )
    args = parser.parse_args()
    main(resume=args.resume, max_stars=args.limit)
