"""Frozen configuration for the interpretable neural-dynamics framework."""
from pathlib import Path

PROJECT = Path(r"D:\Coding\BrainAI\ResNet50_Final_AllUnits_2026-08-24")
RESULTS = PROJECT / "results"
TCAV = PROJECT / "tcav_broden500"
CIRCLE_RESULTS = RESULTS / "circle5000_axis500_2026-08-29"
MAPPING_RESULTS = RESULTS / "six_area_top5_circle_axis_G_2026-08-29"
FACE_CONTROL_RESULTS = MAPPING_RESULTS

N_CONCEPT_AXES = 500
N_CIRCLE_CENTERS = 25 * 25
N_CIRCLE_RADII = 8
N_CIRCLE_TEMPLATES = N_CIRCLE_CENTERS * N_CIRCLE_RADII
N_IMAGES = 1000
AREAS = ("V1", "V2", "V4", "MF", "MO", "CLC")
GTD_DIRECTIONS = ("G", "T", "D")

# Unit selection is response-blind to concepts: top five by frozen proxy OOF r.
UNIT_SELECTION_COLUMN = "repeat_mean_oof_r"
SPATIAL_RULE = "minimum containing candidate circle; nearest center on radius tie"

