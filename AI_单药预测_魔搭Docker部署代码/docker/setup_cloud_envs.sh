#!/usr/bin/env bash
set -u
cd /app
mkdir -p /app/.venvs /app/runtime_reports
for target in logs pka logpapp; do
 case "$target" in logs|pka|logpapp) py=3.11.11;; esac
 echo "Installing original environment for $target"
 if conda create -y -p "/app/.venvs/$target" "python=$py" pip -c conda-forge --override-channels >>/app/runtime_reports/build_${target}.log 2>&1 && \
    "/app/.venvs/$target/bin/python" -m pip install --no-cache-dir -r "/app/envs/requirements_${target}_exact.txt" >>/app/runtime_reports/build_${target}.log 2>&1; then
    echo "$target installed; run-time smoke test will verify real inference"
 else
    echo "WARNING: $target install failed; model will stay unavailable. See build_${target}.log"
 fi
done
# Density was trained with GPU CUDA 12.8 torch; CPU deployment intentionally uses
# a different PyTorch wheel only when explicitly allowed. A real smoke test is mandatory.
if [ "${INSTALL_DENSITY_CPU:-0}" = "1" ]; then
 echo 'Installing density CPU adaptation (non-original torch CUDA build)'
 if conda create -y -p /app/.venvs/density "python=3.12.13" pip -c conda-forge --override-channels >>/app/runtime_reports/build_density.log 2>&1 && \
    /app/.venvs/density/bin/python -m pip install --no-cache-dir -r /app/envs/requirements_density_exact.txt >>/app/runtime_reports/build_density.log 2>&1 && \
    /app/.venvs/density/bin/python -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu >>/app/runtime_reports/build_density.log 2>&1; then
   echo 'Density CPU setup complete (requires explicit verifier opt-in and inference check)'
 else
   echo 'WARNING: density CPU setup failed'
 fi
fi
# Legacy logP Python3.5 + TensorFlow 1.14 is deliberately not replaced.
# IDR is not installed: trained .joblib is not present in the supplied assets.
if [ "${INSTALL_DENSITY_CPU:-0}" = "1" ]; then export DENSITY_CPU_ADAPTATION=1; fi
python /app/scripts/match_and_run_v06.py --verify-only || true
exit 0
