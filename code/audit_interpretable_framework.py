"""Build a compact reproducibility manifest for the interpretable framework."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from interpretable_framework_config import (
    AREAS, CIRCLE_RESULTS, FACE_CONTROL_RESULTS, GTD_DIRECTIONS, MAPPING_RESULTS,
    N_CIRCLE_TEMPLATES, N_CONCEPT_AXES, N_IMAGES, PROJECT, RESULTS, SPATIAL_RULE,
    TCAV, UNIT_SELECTION_COLUMN,
)


def artifact(path, required=True):
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "required": required,
    }


def main():
    artifacts = [
        artifact(TCAV / "broden500_native_channel_cav_bank.npz"),
        artifact(TCAV / "broden500_cav_quality_by_layer.csv"),
        artifact(CIRCLE_RESULTS / "circle_manifest_5000.csv"),
        artifact(CIRCLE_RESULTS / "axis_manifest_500.csv"),
        artifact(CIRCLE_RESULTS / "axis500_circle5000_scores_f16.dat"),
        artifact(RESULTS / "all_unit_best_window_results.csv"),
        artifact(RESULTS / "all_fold_spatial_and_depth_parameters.csv"),
        artifact(MAPPING_RESULTS / "top_axis_per_unit.csv"),
        artifact(FACE_CONTROL_RESULTS / "mf_face_part_axis_pairwise_correlations.csv", False),
    ]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all(x["exists"] for x in artifacts if x["required"]) else "incomplete",
        "frozen_design": {
            "neural_targets": list(GTD_DIRECTIONS),
            "concept_axes": N_CONCEPT_AXES,
            "candidate_circles": N_CIRCLE_TEMPLATES,
            "images": N_IMAGES,
            "areas": list(AREAS),
            "unit_selection": UNIT_SELECTION_COLUMN,
            "spatial_mapping": SPATIAL_RULE,
            "concept_selection_is_response_blind": True,
        },
        "stages": {
            "1_temporal": "G/T/D are computed from response curves; nuisance baseline and template terms are excluded from weights.",
            "2_concepts": "Broden TCAV axes are trained and selected by independent held-out CAV quality.",
            "3_spatial": "Each unit representative-window fwRF is mapped to a frozen candidate circle.",
            "4_mapping": "Circle-weighted concept scores are correlated with observed G/T/D.",
            "5_controls": "Face-part axis collinearity is audited before interpreting local preferences.",
        },
        "artifacts": artifacts,
    }
    out = RESULTS / "interpretable_framework_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

