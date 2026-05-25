# Seg3d VLM export is a second step after trace (keep trace script small).
# Override paths in your environment or pass a custom job JSON.

ANY_ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: trace-example export-vlm-example

trace-example:
	cd $(ANY_ROOT) && python scripts/trace_instance_to_gaussians.py --trace_config configs/trace/alocasia.json

export-vlm-example:
	cd $(ANY_ROOT) && python scripts/export_seg3d_vlm_views.py --job configs/vlm_export/example.export.json
