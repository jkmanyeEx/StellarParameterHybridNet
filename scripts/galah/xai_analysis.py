import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.validation.galah.xai_analyzer import run_xai_line_profile_analysis


def _resolve_weights(weights_arg, survey_dir):
    if weights_arg:
        if not os.path.exists(weights_arg):
            raise FileNotFoundError(f"Specified weights file not found: {weights_arg}")
        return weights_arg
    latest = os.path.join(survey_dir, "stellar_hybrid_model.pth")
    if os.path.exists(latest):
        return latest
    candidates = sorted(
        [os.path.join(survey_dir, f) for f in os.listdir(survey_dir)
         if f.startswith("stellar_hybrid_model_n") and f.endswith(".pth")],
        key=os.path.getmtime,
        reverse=True,
    )
    if candidates:
        print(f"  [Weights] stellar_hybrid_model.pth not found — "
              f"using most recent: {os.path.basename(candidates[0])}")
        return candidates[0]
    raise FileNotFoundError(
        f"No model weights found in: {survey_dir}\n"
        "Execute the GALAH training pipeline first."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XAI analysis for GALAH StellarParameterHybridNet"
    )
    parser.add_argument('--weights', type=str, default=None, metavar='PATH',
                        help='Path to a specific .pth checkpoint.')
    parser.add_argument('--samples', type=int, default=100, metavar='N',
                        help='Number of spectra to analyse (default: 100)')
    parser.add_argument('--ig_steps', type=int, default=50, metavar='N',
                        help='Integrated Gradients interpolation steps (default: 50)')
    args = parser.parse_args()

    base_dir     = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    survey_dir   = os.path.join(base_dir, "weights", "galah")
    weights_path = _resolve_weights(args.weights, survey_dir)

    run_xai_line_profile_analysis(
        num_samples=args.samples,
        ig_steps=args.ig_steps,
        weights_path=weights_path,
    )
