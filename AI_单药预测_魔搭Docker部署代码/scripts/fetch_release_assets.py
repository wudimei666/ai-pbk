#!/usr/bin/env python3
"""Fetch real researcher-trained deploy artifacts from existing GitHub Release v1.0.

No alternate models. Downloads are verified with GitHub's published sha256 digests
when available. Extract only expected paths, protecting against tar traversal.
"""
import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'model_assets'
DOWNLOAD = ROOT / '_downloads'
REPO = 'wudimei666/ai-pbk'
TAG = 'v1.0'
API = 'https://api.github.com/repos/{}/releases/tags/{}'.format(REPO,TAG)


def request(url):
    return urllib.request.Request(url,headers={'User-Agent':'AI-PBPK-research-model-deployer','Accept':'application/octet-stream'})


def release_manifest():
    with urllib.request.urlopen(request(API),timeout=60) as stream:
        obj=json.load(stream)
    return {item['name']:item for item in obj['assets']}


def download_asset(name,manifest,source_dir=None):
    item=manifest.get(name)
    if item is None:
        raise ValueError('GitHub release v1.0 is missing asset: '+name)
    DOWNLOAD.mkdir(exist_ok=True)
    target=DOWNLOAD / name
    if source_dir is not None:
        src=Path(source_dir)/name
        if not src.is_file():
            raise FileNotFoundError(str(src))
        shutil.copyfile(src,target)
    elif not target.is_file():
        with urllib.request.urlopen(request(item['browser_download_url']),timeout=180) as reader, target.open('wb') as writer:
            shutil.copyfileobj(reader,writer,1024*1024)
    if target.stat().st_size!=item['size']:
        target.unlink(missing_ok=True)
        raise ValueError('Size mismatch downloading '+name)
    digest=item.get('digest') or ''
    if digest.startswith('sha256:'):
        actual=hashlib.sha256(target.read_bytes()).hexdigest()
        if actual.lower()!=digest.removeprefix('sha256:').lower():
            target.unlink(missing_ok=True)
            raise ValueError('SHA256 mismatch downloading '+name)
    return target


def safe_extract_tar(tarpath, topdir):
    with tarfile.open(tarpath,'r:gz') as tar:
        for member in tar:
            n=Path(member.name)
            if not member.isfile(): continue
            if n.is_absolute() or '..' in n.parts or n.parts[0]!=topdir: continue
            target=ASSETS/n
            target.parent.mkdir(parents=True,exist_ok=True)
            with tar.extractfile(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src,dst,512*1024)


def safe_extract_zip(zippath, topdir):
    with zipfile.ZipFile(zippath) as z:
        for info in z.infolist():
            n=Path(info.filename)
            if info.is_dir() or n.is_absolute() or '..' in n.parts or n.parts[0]!=topdir: continue
            # No training event logs needed for inference.
            if n.name.startswith('events.out.'):continue
            target=ASSETS/n
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as src,target.open('wb') as dst:shutil.copyfileobj(src,dst,512*1024)


def density_parts(manifest,source_dir):
    # Density archive is 4 zip files containing sequential fragments of one .tar.gz.
    # Stream to disk and extract only minimal inference files, not original training datasets.
    with tempfile.NamedTemporaryFile(suffix='.tar.gz',delete=False) as tmp:
        archive=Path(tmp.name)
        for idx in range(1,5):
            p=download_asset('density_part{}.zip'.format(idx),manifest,source_dir)
            with zipfile.ZipFile(p) as z:
                names=z.namelist()
                if len(names)!=1 or not names[0].endswith('.part{:02d}'.format(idx-1)):
                    raise ValueError('Unexpected density part format: '+p.name)
                with z.open(names[0]) as stream:shutil.copyfileobj(stream,tmp,1024*1024)
    try:
        prefixes=('density_mpnn_modelscope/src/','density_mpnn_modelscope/models/',
                  'density_mpnn_modelscope/reports/')
        exact={'density_mpnn_modelscope/predict_density.py','density_mpnn_modelscope/requirements.lock.txt'}
        with tarfile.open(archive,'r:gz') as tar:
            found=set()
            for member in tar:
                if not member.isfile():continue
                name=member.name.removeprefix('./')
                if (name not in exact and not name.startswith(prefixes)) or '__pycache__' in name or name.endswith('.pyc'):continue
                n=Path(name)
                if n.is_absolute() or '..' in n.parts:continue
                dest=ASSETS/n;dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as input_stream,dest.open('wb') as output_stream:
                    shutil.copyfileobj(input_stream,output_stream,1024*1024)
                found.add(name)
            if 'density_mpnn_modelscope/models/density_mpnn_final.pt' not in found:
                raise ValueError('Density archive missing final .pt checkpoint')
    finally:archive.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--core',action='store_true',help='Fetch LogS, pKa and LogPapp')
    parser.add_argument('--density',action='store_true',help='Fetch 4 large density archives')
    parser.add_argument('--logp',action='store_true',help='Fetch legacy LogP checkpoint (not environment)')
    parser.add_argument('--local-assets',help='Offline testing only: source directory with Release asset filenames')
    args=parser.parse_args()
    manifest=release_manifest() if not args.local_assets else {p.name:{'size':p.stat().st_size,'browser_download_url':''} for p in Path(args.local_assets).iterdir() if p.is_file()}
    ASSETS.mkdir(exist_ok=True)
    if args.core:
        for filename,folder in [('LogS.gz','logs_deploy'),('pka_DEPLOY_rev5.tar.gz','pka_deploy_rev5'),('logpapp_DEPLOY.tar.gz','logpapp_deploy')]:
            asset=download_asset(filename,manifest,args.local_assets)
            safe_extract_tar(asset,folder)
            print('Extracted',folder,flush=True)
    if args.logp:
        asset=download_asset('logp.zip',manifest,args.local_assets)
        safe_extract_zip(asset,'logp')
    if args.density:density_parts(manifest,args.local_assets)
    print('Real checkpoint files available:',sorted(x.name for x in ASSETS.iterdir()))

if __name__=='__main__':main()
