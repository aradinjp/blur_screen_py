#!/bin/bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "Error: run this script in a native ARM64 macOS shell." >&2
    exit 1
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
download_models=false
if [[ "${1:-}" == "--download-models" && "$#" == 1 ]]; then
    download_models=true
elif [[ "$#" != 0 ]]; then
    echo "Usage: ./setup_macos.sh [--download-models]" >&2
    exit 1
fi

python_bin="${PYTHON_BIN:-python3.12}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python 3.12 is required. Install it with: brew install python@3.12" >&2
    exit 1
fi
"$python_bin" -c 'import platform, sys; assert sys.version_info[:2] == (3, 12) and platform.machine() == "arm64", "Use native ARM64 Python 3.12, without Rosetta"'
for executable in ffmpeg ffprobe; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        echo "FFmpeg is required. Install it with: brew install ffmpeg" >&2
        exit 1
    fi
done
encoder_list="$(ffmpeg -hide_banner -encoders 2>/dev/null)"
[[ "$encoder_list" == *libx264* ]] || {
    echo "FFmpeg must include libx264 support." >&2
    exit 1
}

environment_dir="$project_dir/.venv-macos"
"$python_bin" -m venv "$environment_dir"
"$environment_dir/bin/python" -c 'import platform, sys; assert sys.version_info[:2] == (3, 12) and platform.machine() == "arm64", "Existing environment is incompatible"'
"$environment_dir/bin/python" -m pip install --only-binary=:all: -r "$project_dir/requirements-macos.txt"
"$environment_dir/bin/python" -m pip check
diagnostic_args=(--hardware-info --inference-test --device auto)
if "$download_models"; then
    diagnostic_args+=(--download-models)
fi
"$environment_dir/bin/python" "$project_dir/anonymize.py" "${diagnostic_args[@]}"
echo "Installation and OCR initialization test completed."
printf 'Activate with: source "%s/bin/activate"\n' "$environment_dir"
echo "Run: python anonymize.py --device auto --input video.mp4 --output video_anonymized.mp4 --config redaction_data.json"
