"""Local, single-SMILES UI for user's own trained models (no batch/PBPK/training)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess

from scripts.match_and_run_v06 import EXPECTED, WORKERS, check, find_interpreter

ROOT = Path(__file__).resolve().parent
PROPERTIES = [
    ("LogS", "logs", "logs", "log10(mol/L)"),
    ("LogP", "logp", "logp", "无量纲"),
    ("酸性 pKa", "pka", "acidic", "无量纲"),
    ("碱性 pKa", "pka", "basic", "无量纲"),
    ("Density", "density", "density", "g/cm³"),
    ("Papp", "logpapp", "logpapp", "cm/s"),
    ("IDR", "idr", "idr", "μg/(min·cm²)"),
]


def _blank_rows():
    return [[name, "—", unit, "等待预测"] for name, _, _, unit in PROPERTIES]


@lru_cache(maxsize=1)
def verify_models():
    """Strict source-environment verification, plus actual CCO inference smoke test.

    This prevents accidentally using a different RDKit descriptor set or training artifact.
    Environment installation or file changes require restarting the application.
    """
    checks = {}
    for kind in EXPECTED:
        exe = find_interpreter(kind)
        checks[kind] = check(kind, exe, smoke_enabled=True)
    return checks


def _run_one(kind, label, smiles, executable, timeout=240):
    matches = [x for x in WORKERS[kind] if x[0] == label]
    if not matches:
        raise ValueError('缺少对应预测入口')
    _, script, args = matches[0]
    proc = subprocess.run(
        [str(executable), str(ROOT/script), *args],
        input=json.dumps({'smiles': smiles}, ensure_ascii=False),
        text=True, capture_output=True, cwd=str(ROOT), timeout=timeout,
        env={**os.environ,'PYTHONIOENCODING':'utf-8'}, shell=False,
    )
    if proc.returncode:
        raise RuntimeError('真实模型推理失败；请检查原环境、SMILES 和模型文件')
    try:
        data = json.loads(proc.stdout.strip())
    except json.JSONDecodeError as ex:
        raise RuntimeError('模型返回格式错误') from ex
    if not isinstance(data, dict) or not data.get('model_version') or not data.get('model_source'):
        raise ValueError('模型未返回可核查的权重来源和版本')
    expected = 'log10 mol/L' if kind == 'logs' else 'g/cm3' if kind == 'density' else 'cm/s' if kind == 'logpapp' else 'dimensionless'
    if data.get('unit') != expected:
        raise ValueError('模型输出单位不匹配')
    value = data.get('value')
    if isinstance(value,bool) or value is None:
        raise ValueError('模型没有返回有效数值')
    value=float(value)
    if not math.isfinite(value) or (kind in ('density','logpapp') and value <= 0):
        raise ValueError('预测数值不合理')
    if kind == 'logpapp':
        lpapp = float(data['logPapp_log10_cm_s'])
        if not math.isfinite(lpapp) or not math.isclose(math.log10(value),lpapp,abs_tol=1e-10,rel_tol=0):
            raise ValueError('Papp/LogPapp 换算验证失败')
    return {'value':value,'metadata':data}


def _render_value(kind, value):
    if kind == 'logpapp':
        return f'{value:.6e}'
    return f'{value:.5f}'


def predict_single(smiles, progress=None):
    """Single molecule only. No inferences using placeholders or alternative models."""
    if not isinstance(smiles, str) or not smiles.strip():
        return _blank_rows(), '请输入一个药物的 SMILES。', '未开始运行。'
    smiles = smiles.strip()
    if len(smiles)>500 or '\n' in smiles or '\r' in smiles:
        return _blank_rows(), '请只输入一个 SMILES（不超过 500 个字符），不要输入多行。', '未开始运行。'

    if progress is not None:
        progress(0.1, desc='检查各模型的原始环境和真实权重…')
    checks=verify_models()
    results={}
    warnings=[]
    available_kinds={k:v['executable'] for k,v in checks.items() if v.get('ok')}
    tasks=[]
    for name,kind,label,unit in PROPERTIES:
        if kind != 'idr' and kind in available_kinds:
            tasks.append((kind,label,available_kinds[kind]))
    if progress is not None:
        progress(0.35, desc='使用已核验环境执行 SMILES 预测…')
    with ThreadPoolExecutor(max_workers=min(4,len(tasks)) or 1) as pool:
        futures={pool.submit(_run_one,kind,label,smiles,exe):(kind,label) for kind,label,exe in tasks}
        for future in as_completed(futures):
            kind,label=futures[future]
            try:
                results[(kind,label)]=future.result()
            except Exception as ex:
                results[(kind,label)]=ex

    successful=0
    rows=[]
    provenance=[]
    for name,kind,label,unit in PROPERTIES:
        if kind == 'idr':
            rows.append([name,'—',unit,'未接入：缺少 IDR 模型'])
            continue
        if kind not in available_kinds:
            status=checks[kind].get('status','未检测')
            mapping={
                'missing_interpreter':'未安装原环境',
                'version_mismatch':'环境版本不匹配',
                'runtime_failed':'环境启动失败',
                'worker_failed':'原模型推理检测未通过',
            }
            rows.append([name,'—',unit,mapping.get(status,'未通过原环境核验')])
            continue
        result=results.get((kind,label))
        if not result or isinstance(result,Exception):
            rows.append([name,'—',unit,'实际预测失败'])
            continue
        successful+=1
        rows.append([name,_render_value(kind,result['value']),unit,'预测成功'])
        data=result['metadata']
        provenance.append(f"- {name}：`{data['model_version']}`")

    if progress is not None:
        progress(1.0, desc='完成')
    total=len(PROPERTIES)
    status=f'**本次预测：{successful}/{total} 项成功**。未连接的模型不会以人工数值或其他模型补齐。'
    if successful and any(k=='logs' for k,l in results if isinstance(results[(k,l)],dict)):
        warnings.append('LogS 的原训练数据尚未证明统一为 pH 7.0；此处仅供药物性质研究，不应直接称为 pH 7 溶解度。')
    if os.environ.get('DENSITY_CPU_ADAPTATION')=='1' and isinstance(results.get(('density','density')),dict):
        warnings.append('Density 使用经实际推理验证的 CPU PyTorch 移植环境，而非原训练 CUDA 12.8 版本。')
    if ('logpapp','logpapp') in results and isinstance(results[('logpapp','logpapp')],dict):
        warnings.append('Papp 使用你上传的 XGBoost 重建模型，并非原论文作者完全相同的模型。')
    details='**本地科研使用说明**\n\n' + ('\n\n'.join(warnings)+'\n\n' if warnings else '')
    details+='**本次成功模型的权重版本**\n\n'+ ('\n'.join(provenance) if provenance else '当前没有通过原环境核验并完成实际推理的模型。')
    details+='\n\n没有运行批量任务、没有重训模型，也没有调用任何在线预测服务。'
    return rows, status, details


def build_demo():
    import gradio as gr
    with gr.Blocks(title='我的 AI 药物性质预测平台') as demo:
        gr.Markdown('## AI 药物性质预测平台\n输入 **一个 SMILES**，点击预测，即可使用你自己训练好的模型得到结果。')
        smiles=gr.Textbox(label='药物 SMILES',placeholder='例如：CCO',lines=2,max_lines=2)
        button=gr.Button('开始预测',variant='primary')
        status=gr.Markdown('准备就绪。首次预测会先检查原始模型环境，因此可能稍慢。')
        table=gr.Dataframe(
            value=_blank_rows(),
            headers=['药物性质','预测结果','单位','模型状态'],
            datatype=['str','str','str','str'], interactive=False,
            label='预测结果',wrap=True,
        )
        with gr.Accordion('模型说明与溯源（需要时展开）',open=False):
            details=gr.Markdown('模型未开始检测。')
        gr.Markdown('**说明：** 仅单药预测；IDR 模型尚未接入。不同 AI 模型原训练环境不一致，平台会严格检测，未验证成功的模型显示状态，不生成数值。')
        button.click(predict_single,inputs=[smiles],outputs=[table,status,details],show_progress='full')
        smiles.submit(predict_single,inputs=[smiles],outputs=[table,status,details],show_progress='full')
    return demo


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--check',action='store_true',help='核查模型环境，退出，不打开界面')
    args=parser.parse_args()
    if args.check:
        for kind,result in verify_models().items():
            print(f'{kind}: {result.get("status","unknown")}')
    else:
        # Docker / ModelScope must bind to all interfaces; localhost still works for local runs.
        build_demo().launch(server_name=os.environ.get('GRADIO_SERVER_NAME','0.0.0.0'),server_port=args.port,share=False,inbrowser=False,show_error=False)
