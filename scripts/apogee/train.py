import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.training.apogee.engine import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from best saved checkpoint')
    args = parser.parse_args()
    main(resume=args.resume)
