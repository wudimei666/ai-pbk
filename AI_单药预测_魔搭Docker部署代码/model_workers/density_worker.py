"""User-trained crystal-density MPNN, loaded from final artifact without retraining."""
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT/'model_assets'/'density_mpnn_modelscope'
sys.path.insert(0,str(ASSET))

def main():
    import torch
    from src.inference import load_model
    from predict_density import predict_one
    req=json.load(sys.stdin)
    path=ASSET/'models'/'density_mpnn_final.pt'
    device=torch.device('cpu')
    model,config=load_model(path,device=device) # Only trusted torch checkpoints.
    pred=predict_one(model,req['smiles'],device)
    if pred.get('status')!='ok':raise ValueError(str(pred))
    print(json.dumps({'value':pred['predicted_density_g_cm3'],'unit':'g/cm3',
        'model_version':'trained-density-MPNN-sha256-'+hashlib.sha256(path.read_bytes()).hexdigest()[:12],
        'model_source':'user density multipart archive; models/density_mpnn_final.pt',
        'training_configuration':config},allow_nan=False))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'Density worker error: {exc}',file=sys.stderr);sys.exit(1)
