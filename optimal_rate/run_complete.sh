#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
JULIA_BIN="${JULIA_BIN:-julia}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-16}"
export CECR_BLAS_THREADS="${CECR_BLAS_THREADS:-4}"
export CECR_SECTOR_PARALLEL="${CECR_SECTOR_PARALLEL:-4}"
export OPENBLAS_NUM_THREADS="$CECR_BLAS_THREADS"
mkdir -p optimal_rate/{meshes,results_complete,logs}
cases=("$@")
if [ "${#cases[@]}" -eq 0 ]; then cases=(2d_one_centre 2d_two_centre 3d_h2plus); fi
for case_name in "${cases[@]}"; do
    case "$case_name" in
        2d_one_centre|2d_two_centre)
            for h in ${CECR_H2D_LEVELS:-0.8 0.4 0.2 0.1}; do
                mesh_dir="optimal_rate/meshes/${case_name}_h${h}"
                python3 optimal_rate/gen_mesh.py --case "$case_name" --h "$h" --out "$mesh_dir" \
                    > "optimal_rate/logs/mesh_${case_name}_h${h}.log"
                "$JULIA_BIN" --project=. optimal_rate/complete_case.jl "$case_name" "$mesh_dir" \
                    "optimal_rate/results_complete/${case_name}_h${h}.json" \
                    > "optimal_rate/logs/complete_${case_name}_h${h}.log" 2>&1
                tail -n 2 "optimal_rate/logs/complete_${case_name}_h${h}.log"
            done
            ;;
        3d_h2plus)
            for h in ${CECR_H3D_LEVELS:-1.0 0.6666666666666666 0.5}; do
                mesh_dir="optimal_rate/meshes/3d_h2plus_cap_h${h}"
                python3 optimal_rate/gen_mesh_3d.py --h "$h" --global-cap 1 --out "$mesh_dir" \
                    > "optimal_rate/logs/mesh_3d_h2plus_cap_h${h}.log"
                "$JULIA_BIN" --project=. optimal_rate/complete_case_3d.jl "$mesh_dir" \
                    "optimal_rate/results_complete/3d_h2plus_cap_h${h}.json" \
                    > "optimal_rate/logs/complete_3d_h2plus_cap_h${h}.log" 2>&1
                tail -n 2 "optimal_rate/logs/complete_3d_h2plus_cap_h${h}.log"
            done
            ;;
        *) printf 'Unknown case: %s\n' "$case_name" >&2; exit 2 ;;
    esac
done
if [ "${CECR_RUN_TESTS:-1}" = 1 ]; then
    "$JULIA_BIN" --project=. optimal_rate/test_stable.jl > optimal_rate/logs/validation.log 2>&1
    "$JULIA_BIN" --project=. optimal_rate/test_complete_3d.jl > optimal_rate/logs/independent_3d_validation.log 2>&1
    if [ -d optimal_rate/meshes/3d_h2plus_cap_h1.0 ]; then
        python3 optimal_rate/test_mesh_3d.py --prefix 3d_h2plus_cap_h \
            --levels 1.0 0.6666666666666666 0.5 \
            --output optimal_rate/meshes/mesh_validation_3d_cap.json \
            > optimal_rate/logs/test_mesh_3d_cap.log
    fi
fi
python3 optimal_rate/geometric_certificate.py
python3 optimal_rate/paper_assets_tools/make_assets.py
