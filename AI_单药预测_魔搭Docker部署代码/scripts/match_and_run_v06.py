#!/usr/bin/env python3
"""Verify five user-supplied model runtime locks and actually execute SMILES inference.

No version substitution, checkpoint retraining, or fabricated model outputs.
The UI launcher enables ONLY verified workers; use --verify-only to audit.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runtime_reports' / 'environment_match_v06.json'

EXPECTED = {
    'logs': {'python':'3.11.11','rdkit':'2026.03.6','lightgbm':'4.7.0','numpy':'2.4.6','pandas':'3.0.6','scikit-learn':'1.9.1','joblib':'1.6.0','hyperopt':'0.3.0'},
    'pka': {'python':'3.11.11','rdkit':'2025.09.6','lightgbm':'4.6.0','numpy':'2.1.3','pandas':'2.2.3','scikit-learn':'1.6.1','scipy':'1.15.3','joblib':'1.4.2','optuna':'4.8.0'},
    'logpapp': {'python':'3.11.11','rdkit':'2022.09.5','xgboost':'3.1.3','numpy':'1.26.4','pandas':'2.2.3','scikit-learn':'1.8.0','joblib':'1.4.2','hyperopt':'0.2.7'},
    'density': {'python':'3.12.13','rdkit':'2026.03.5','torch':'2.10.0+cu128','numpy':'2.4.3','pandas':'2.3.3','scikit-learn':'1.9.0','pyarrow':'25.0.0'},
    'logp': {'python':'3.5.6','rdkit':'2018.03.4','tensorflow':'1.14.0','numpy':'1.15.2','pandas':'0.23.4','scikit-learn':'0.20.0','scipy':'1.1.0','matplotlib':'3.0.0','xgboost':'0.80'},
}

ENV_KEYS = {'logs':'LOGS_PYTHON','pka':'PKA_PYTHON','logpapp':'LOGPAPP_PYTHON','density':'DENSITY_PYTHON','logp':'LOGP_PYTHON'}
WORKERS = {
    'logs': [('logs','model_workers/logs_worker.py',[])],
    'pka': [('acidic','model_workers/pka_worker.py',['--kind','acidic']),('basic','model_workers/pka_worker.py',['--kind','basic'])],
    'logpapp': [('logpapp','model_workers/logpapp_worker.py',[])],
    'density': [('density','model_workers/density_worker.py',[])],
    'logp': [('logp','model_workers/logp_worker.py',[])],
}

# Keep the child program parseable by the old Python 3.5 used for LogP.
RUNTIME_CODE = '''
from __future__ import print_function
import json,sys
result={'python':'.'.join(map(str,sys.version_info[:3])), 'executable':sys.executable,'versions':{},'errors':{}}
try:
 from rdkit import rdBase
 result['versions']['rdkit']=rdBase.rdkitVersion
except Exception as ex:result['errors']['rdkit']=repr(ex)
try:
 if sys.version_info >= (3,8):
  from importlib.metadata import version as lookup
 else:
  import pkg_resources
  def lookup(name):return pkg_resources.get_distribution(name).version
 for name in __MODULES__:
  try:result['versions'][name]=lookup(name)
  except Exception as ex:result['errors'][name]=repr(ex)
except Exception as ex:result['errors']['metadata']=repr(ex)
print(json.dumps(result))
'''

def normalize(s):
    """Only ignore zero-padding in numeric components; preserve build qualifiers."""
    parts=str(s).split('.')
    return '.'.join(str(int(x)) if x.isdigit() else x for x in parts)


def find_interpreter(kind, explicit=None):
    raw = explicit or os.environ.get(ENV_KEYS[kind], '').strip()
    if raw:
        return Path(raw).expanduser().resolve()
    root=ROOT/'.venvs'/kind
    choices=(root/'bin/python',root/'python.exe',root/'Scripts/python.exe')
    return next((p for p in choices if p.is_file()),None)


def run_capture(command, *, timeout=120, input_text=None):
    try:
        return subprocess.run(command, cwd=str(ROOT), text=True, input=input_text,
                 capture_output=True, timeout=timeout, env={**os.environ,'PYTHONIOENCODING':'utf-8'})
    except (OSError,subprocess.TimeoutExpired) as ex:
        return {'error':str(ex)}


def check(kind, executable, smoke_enabled=True):
    # Deliberate opt-in ONLY: deployment CPU PyTorch is not identical to
    # research training CUDA PyTorch. Real checkpoint smoke-test is mandatory.
    expected=dict(EXPECTED[kind])
    cpu_adaptation=(kind=='density' and os.environ.get('DENSITY_CPU_ADAPTATION')=='1')
    if cpu_adaptation:
        expected['torch']='2.10.0+cpu'
    if executable is None or not executable.is_file():
        return {'ok':False, 'status':'missing_interpreter', 'message':'尚未建立原训练版本环境或解释器路径错误'}
    modules=[x for x in expected if x not in ('python','rdkit')]
    probe=run_capture([str(executable),'-c',RUNTIME_CODE.replace('__MODULES__',repr(modules))],timeout=60)
    if isinstance(probe,dict):
        return {'ok':False,'status':'runtime_failed','executable':str(executable),'details':probe}
    if probe.returncode:
        return {'ok':False,'status':'runtime_failed','executable':str(executable),'error':probe.stderr[-2000:]}
    try:
        fingerprint=json.loads(probe.stdout)
    except ValueError:
        return {'ok':False,'status':'runtime_bad_json','executable':str(executable),'error':probe.stdout[-1000:]}
    installed={'python':fingerprint.get('python'),**fingerprint.get('versions',{})}
    mismatch={name:{'required':ver,'detected':installed.get(name,'missing')}
              for name,ver in expected.items() if normalize(installed.get(name,'missing'))!=normalize(ver)}
    if fingerprint.get('errors'):
        mismatch['metadata_or_import_errors']=fingerprint['errors']
    info={'ok':False,'executable':str(executable),'installed_versions':installed,
          'status':'version_mismatch' if mismatch else 'version_matched_unverified','mismatches':mismatch}
    if cpu_adaptation:
        info['deployment_adaptation']='CPU torch 2.10.0+cpu replaces research CUDA 12.8 wheel; real SMILES smoke-test still required'
    if mismatch or not smoke_enabled:
        return info
    cases=[]
    for label, path, extra in WORKERS[kind]:
        action=run_capture([str(executable),str(ROOT/path),*extra],input_text=json.dumps({'smiles':'CCO'}),timeout=200)
        if isinstance(action,dict):
            cases.append({'label':label,'ok':False,'error':action['error']});continue
        if action.returncode:
            cases.append({'label':label,'ok':False,'error':(action.stderr or action.stdout)[-1600:]});continue
        try:
            response=json.loads(action.stdout.strip())
            v=response['value']
            if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v):
                raise ValueError('输出不是有限实数')
            if not response.get('model_source') or not response.get('model_version'):
                raise ValueError('缺少模型权重溯源信息')
            expected_unit='log10 mol/L' if kind=='logs' else 'g/cm3' if kind=='density' else 'cm/s' if kind=='logpapp' else 'dimensionless'
            if response.get('unit')!=expected_unit:
                raise ValueError('预测单位不匹配')
            if kind in ('density','logpapp') and v<=0:
                raise ValueError('结果必须为正数')
            if kind=='logpapp' and not math.isclose(math.log10(v),response['logPapp_log10_cm_s'],abs_tol=1e-10,rel_tol=0):
                raise ValueError('LogPapp/Papp 转换不符')
            cases.append({'label':label,'ok':True,'value':v,'unit':response['unit'],
                          'model_version':response['model_version'],'model_source':response['model_source']})
        except (ValueError,KeyError,TypeError) as ex:
            cases.append({'label':label,'ok':False,'error':'无效 worker JSON: '+str(ex)})
    info['smiles_smoke_tests']=cases
    info['ok']=bool(cases) and all(x['ok'] for x in cases)
    info['status']='verified_real_inference' if info['ok'] else 'worker_failed'
    return info


def main(argv=None):
    parser=argparse.ArgumentParser(description='Verify original user-supplied model environments without substitution')
    for kind in EXPECTED:
        parser.add_argument('--'+kind+'-python',dest=kind+'_python')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--no-smoke',action='store_true',help='Debug only: version checks cannot certify inference')
    parser.add_argument('--require-all',action='store_true',help='Treat missing/unverified models as nonzero exit')
    args=parser.parse_args(argv)
    checks={}
    for kind in EXPECTED:
        path=find_interpreter(kind,getattr(args,kind+'_python'))
        checks[kind]=check(kind,path,smoke_enabled=not args.no_smoke)
        print('[{}] {}'.format(kind,checks[kind]['status']),flush=True)
        if checks[kind].get('mismatches'):
            print('  不匹配:',json.dumps(checks[kind]['mismatches'],ensure_ascii=False),flush=True)
        for item in checks[kind].get('smiles_smoke_tests',[]):
            print('  {}: {}'.format(item['label'],'通过' if item['ok'] else item.get('error','失败')),flush=True)
    verified={k:v['executable'] for k,v in checks.items() if v['ok']}
    summary={'release':'v0.6','checked_at_utc':datetime.now(timezone.utc).isoformat(),
             'test_smiles':'CCO','verified_interpreters':verified,'checks':checks,
             'caution':'代码自检和单分子推理都不等于犬 PK 外部验证'}
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('检测报告:',REPORT,flush=True)
    if args.verify_only:
        return 2 if args.require_all and len(verified)<len(EXPECTED) else 0
    from src.auto_register import register_workers
    # Do not autowire the launching interpreter for incompatible checkpoints.
    os.environ['AI_V06_VERIFIED_ONLY']='1'
    register_workers(ROOT,verified_interpreters=verified)
    print('网页将仅自动接入已完成真实推理的模型:',','.join(verified) or '无',flush=True)
    try:
        return subprocess.call([sys.executable,'-m','streamlit','run',str(ROOT/'app.py')],cwd=str(ROOT))
    except OSError as ex:
        print('Streamlit 启动失败: '+str(ex),file=sys.stderr)
        return 3

if __name__=='__main__':
    sys.exit(main())
