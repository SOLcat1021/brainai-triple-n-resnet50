"""Run the existing res4 SAE pilot with a 4096-latent Top-K dictionary."""
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import pilot_res4_sae_axis_stability as pilot

# Feasibility configuration: retain the same ~6.25% active fraction as the
# previous Top-K=8/128 run, while using a substantially wider dictionary.
pilot.HIDDEN = 4096
pilot.TOPK_VALUES = (256,)
pilot.N_REPS = 3
pilot.EPOCHS = 100
pilot.SEED = 20260905

if __name__ == "__main__":
    pilot.main()
