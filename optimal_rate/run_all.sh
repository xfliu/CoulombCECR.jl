#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
JULIA_BIN="${JULIA_BIN:-julia}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS=1
mkdir -p optimal_rate/{meshes,results,logs}
cases=("$@")
if [ "${#cases[@]}" -eq 0 ]; then cases=(2d_one_centre 2d_two_centre); fi
for case_name in "${cases[@]}"; do
    for h in ${H_LEVELS:-0.8 0.4 0.2 0.1}; do
        mesh_dir="optimal_rate/meshes/${case_name}_h${h}"
        python3 optimal_rate/gen_mesh.py --case "$case_name" --h "$h" --out "$mesh_dir" \
            > "optimal_rate/logs/mesh_${case_name}_h${h}.log"
        "$JULIA_BIN" --project=. optimal_rate/compute.jl "$case_name" "$mesh_dir" \
            "optimal_rate/results/${case_name}_h${h}.json" \
            > "optimal_rate/logs/${case_name}_h${h}.log" 2>&1
        tail -n 2 "optimal_rate/logs/${case_name}_h${h}.log"
    done
done
python3 optimal_rate/analyze.py
python3 optimal_rate/geometric_certificate.py
