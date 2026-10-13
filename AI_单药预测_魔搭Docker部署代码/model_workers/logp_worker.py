"""Original legacy DeepChem 2.2 / TensorFlow 1.x logP inference.
Run using your *separate* Python 3.5 environment with compatible DeepChem.
"""
import hashlib
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT/'model_assets'/'logp'
sys.path.insert(0,str(ASSET.parent))

def main():
    # Add no fallbacks: never call RDKit Crippen.MolLogP in place of the AI.
    from logp.ai_models.logp.predictor import predict_logp
    req=json.load(sys.stdin)
    val=predict_logp(req['smiles'])
    model_path=ASSET/'ai_models'/'logp'/'model'/'model-12000.data-00000-of-00001'
    print(json.dumps({'value':val,'unit':'dimensionless',
       'model_version':'DeepChem-legacy-model-12000-sha256-'+hashlib.sha256(model_path.read_bytes()).hexdigest()[:12],
       'model_source':'user logp(1).zip; TensorGraph checkpoint model-12000'},allow_nan=False))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('LogP worker error: {}'.format(exc),file=sys.stderr);sys.exit(1)
