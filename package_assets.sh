#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

archive="${1:-aero-dml-assets.zip}"

if [[ -e "$archive" ]]; then
    echo "Error: $archive already exists. Remove it first." >&2
    exit 1
fi

mapfile -t asset_paths < <(grep -vE '^[[:space:]]*(#|$)' asset_paths.txt)

for path in "${asset_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Error: asset path does not exist: $path" >&2
        exit 1
    fi
done

zip -r "$archive" "${asset_paths[@]}"

echo "Created $archive"
