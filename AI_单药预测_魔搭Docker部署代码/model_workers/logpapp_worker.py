"""Trusted, PRETRAINED LogPapp -> Papp worker; no retraining or fallback.

Run this worker in the original pinned RDKit 2022.09.5 environment:
  echo '{"smiles":"CCO"}' | /path/to/logpapp_python model_workers/logpapp_worker.py

stdin: {"smiles": "..."}; stdout: one JSON record. If RDKit descriptors
or any feature contract changed, the original predictor raises an error.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "model_assets" / "logpapp_deploy"
MODEL = ASSET / "model" / "logPapp_xgboost_deployment_all_data.joblib"
sys.path.insert(0, str(ASSET / "src"))


def predict_request(request):
    if not isinstance(request, dict):
        raise ValueError("输入必须为 JSON 对象")
    smiles = request.get("smiles")
    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 500:
        raise ValueError("请输入不超过 500 字符的非空 SMILES")
    if not MODEL.is_file():
        raise FileNotFoundError("LogPapp 部署权重不存在")

    # Strictly call the author's reconstructed inference implementation.
    # Do not bypass feature registry or RDKit-version checks to get a number.
    from logpapp_repro.author_cli import predict
    result = predict(smiles, MODEL, rdkit_policy="target")
    log_papp = float(result["predicted_logPapp"])
    papp = float(result["predicted_Papp_cm_s"])
    if not (math.isfinite(log_papp) and math.isfinite(papp) and papp > 0):
        raise ValueError("预测值不是合法的正 Papp")
    if not math.isclose(math.log10(papp), log_papp, rel_tol=0, abs_tol=1e-10):
        raise ValueError("LogPapp 与 Papp 单位转换不一致")
    if result.get("artifact_role") != "deployment_model_refit_all_data_after_evaluation":
        raise ValueError("意外的模型产物类型；请重新审核权重")
    return {
        "value": papp,
        "unit": "cm/s",
        "logPapp_log10_cm_s": log_papp,
        "canonical_smiles": result["canonical_smiles"],
        "model_version": "trained-LogPapp-XGBoost-fullfit-sha256-" + hashlib.sha256(MODEL.read_bytes()).hexdigest()[:12],
        "model_source": "user logpapp_DEPLOY.tar.gz; model/logPapp_xgboost_deployment_all_data.joblib",
        "scientific_status": "RECONSTRUCTION_NOT_EXACT_PAPER_MODEL",
        "artifact_role": result["artifact_role"],
        "training_scope": "all_available_data_after_separate_evaluation; do_not_reuse_independent_test_metrics_as_this_artifact_score",
        "runtime_version_warnings": result.get("runtime_version_warnings", {}),
    }


def main():
    request = json.load(sys.stdin)
    print(json.dumps(predict_request(request), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"LogPapp worker error: {exc}", file=sys.stderr)
        raise SystemExit(1)
