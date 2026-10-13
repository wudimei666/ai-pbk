"""Read JSON from stdin; predict using the user's trained LogS LightGBM artifact."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / 'model_assets' / 'logs_deploy'
sys.path.insert(0, str(ASSET))


def main():
    import joblib
    from predict_logs import predict_one
    req = json.load(sys.stdin)
    path = ASSET / 'model' / 'logS_lightgbm.joblib'
    bundle = joblib.load(path)  # Only deploy an artifact from a trusted researcher.
    pred = predict_one(bundle, req['smiles'])
    if pred.get('status') != 'ok':
        raise ValueError('LogS featurization rejected SMILES: ' + str(pred['status']))
    value = pred['predicted_logS']
    print(json.dumps({'value': value, 'unit':'log10 mol/L',
        'model_version':'trained-LogS-LightGBM-sha256-' + hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        'model_source':'user LogS deploy archive; model/logS_lightgbm.joblib'}, allow_nan=False))


if __name__ == '__main__':
    try:main()
    except Exception as exc:
        print(f'LogS worker error: {exc}', file=sys.stderr)
        sys.exit(1)
