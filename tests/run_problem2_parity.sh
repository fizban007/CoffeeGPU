#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <cpu_build_dir> <gpu_build_dir> [abs_tol] [rel_tol]" >&2
  exit 2
fi

cpu_build_dir=$(cd "$1" && pwd)
gpu_build_dir=$(cd "$2" && pwd)
abs_tol=${3:-}
rel_tol=${4:-}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
compare_bin="${cpu_build_dir}/problem2_parity_compare"
cpu_exec="${cpu_build_dir}/coffee-alfven3d"
gpu_exec="${gpu_build_dir}/coffee-alfven3d"

for path in "${compare_bin}" "${cpu_exec}" "${gpu_exec}"; do
  if [[ ! -x "${path}" ]]; then
    echo "Required executable not found: ${path}" >&2
    exit 1
  fi
done

tmp_root=$(mktemp -d "${TMPDIR:-/tmp}/problem2-parity.XXXXXX")
trap 'rm -rf "${tmp_root}"' EXIT

cpu_run_dir="${tmp_root}/cpu"
gpu_run_dir="${tmp_root}/gpu"
mkdir -p "${cpu_run_dir}" "${gpu_run_dir}"

cp "${repo_root}/tests/problem2_parity_config.toml" \
   "${cpu_run_dir}/config.toml"
cp "${repo_root}/tests/problem2_parity_config.toml" \
   "${gpu_run_dir}/config.toml"

(
  cd "${cpu_run_dir}"
  "${cpu_exec}"
)

(
  cd "${gpu_run_dir}"
  "${gpu_exec}"
)

compare_args=("${cpu_run_dir}/Data" "${gpu_run_dir}/Data")
if [[ -n "${abs_tol}" ]]; then
  compare_args+=("${abs_tol}")
fi
if [[ -n "${rel_tol}" ]]; then
  compare_args+=("${rel_tol}")
fi

"${compare_bin}" "${compare_args[@]}"
