"""Acid/base pKa in independently locked Python 3.11 environment.

This intentionally preserves the original model's environment/feature-contract
checks and refuses to silently replace the RDKit descriptor set.
"""
import hashlib
import json
import sys
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ASSET=ROOT/'model_assets'/'pka_deploy_rev5'
sys.path.insert(0,str(ASSET))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['acidic','basic'],required=True)
    args=p.parse_args()
    from src.core import load_model,predict_smiles
    req=json.load(sys.stdin)
    path=ASSET/'models'/f'{args.kind}_pka_deployment_all_data.joblib'
    obj=load_model(path,args.kind)
    values,failures=predict_smiles(obj,[req['smiles']])
    print(json.dumps({'value':float(values[0]),'unit':'dimensionless',
        'model_version':f'trained-pKa-rev5-{args.kind}-sha256-'+hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        'model_source':f'user pka_DEPLOY_rev5.tar.gz; {args.kind}_pka_deployment_all_data.joblib',
        'descriptor_failures':failures},allow_nan=False))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'pKa worker error: {exc}',file=sys.stderr);sys.exit(1)
