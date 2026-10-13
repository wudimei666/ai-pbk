"""Safely call researcher-provided ML / PBPK executables.

No untrained fallback is permitted: missing workers are a hard error.
Worker IO is one JSON object on stdin and one JSON object on stdout.
"""
from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from typing import Any, Dict

AI_KEYS = (
    "logS_pH7_log10_mol_L", "logP", "acid_pKa", "base_pKa",
    "Papp_cm_s", "density_g_cm3", "IDR_ug_min_cm2",
)

class WorkerNotConfigured(RuntimeError):
    pass

class WorkerError(RuntimeError):
    pass


def split_command(command: str, *, windows: bool | None = None) -> list[str]:
    """Correctly preserve backslashes in quoted Windows executable paths."""
    if windows is None:
        windows = os.name == "nt"
    tokens = shlex.split(command, posix=not windows)
    if windows:
        tokens = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t for t in tokens]
    return tokens


def worker_ready(setting: str) -> bool:
    return bool(os.environ.get(setting, "").strip())


def invoke_worker(setting: str, payload: Dict[str, Any], timeout_s: int = 180) -> Dict[str, Any]:
    command = os.environ.get(setting, "").strip()
    if not command:
        raise WorkerNotConfigured(f"{setting} 尚未配置：请先接入已经训练并验证的真实模型。")
    argv = split_command(command)
    if not argv:
        raise WorkerNotConfigured(f"{setting} 为空")
    try:
        result = subprocess.run(
            argv,
            input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise WorkerError(f"模型调用超时（>{timeout_s} 秒）") from exc
    except OSError as exc:
        raise WorkerError("无法启动工作程序；请检查所在服务器的环境配置") from exc
    if result.returncode:
        # Avoid displaying raw stderr on a publicly accessible scientific web app.
        raise WorkerError(f"工作程序执行失败（退出码 {result.returncode}）；请由管理员检查服务日志")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise WorkerError("工作程序未返回有效 JSON") from exc
    if not isinstance(data, dict):
        raise WorkerError("工作程序输出必须是 JSON 对象")
    return data


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise WorkerError(f"缺失或不合法的预测项：{name}")
    try:
        num = float(value)
    except (ValueError, TypeError) as exc:
        raise WorkerError(f"预测项 {name} 不是数字") from exc
    if not math.isfinite(num):
        raise WorkerError(f"预测项 {name} 是非有限数")
    return num


def validate_ai(response: Dict[str, Any]) -> Dict[str, Any]:
    values = response.get("properties")
    if not isinstance(values, dict):
        raise WorkerError("AI 返回值缺少 properties 对象")
    missing = [k for k in AI_KEYS if k not in values]
    if missing:
        raise WorkerError("AI 缺失属性：" + ", ".join(missing))
    clean = {key: _finite(values[key], key) for key in AI_KEYS}
    if clean["Papp_cm_s"] <= 0 or clean["density_g_cm3"] <= 0 or clean["IDR_ug_min_cm2"] < 0:
        raise WorkerError("Papp、density 必须为正；IDR 不能为负")
    version = response.get("model_version")
    source = response.get("model_source")
    if not isinstance(version, str) or not version.strip() or not isinstance(source, str) or not source.strip():
        raise WorkerError("AI 预测必须提供真实 model_version 和 model_source 进行溯源")
    return {
        "properties": clean,
        "model_version": version.strip(),
        "model_source": source.strip(),
        "notes": str(response.get("notes", "")),
        "per_property_models": response.get("per_property_models", {}),
    }


def validate_pbpk(response: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(response.get("time_h"), list) or not isinstance(response.get("plasma_mg_L"), list):
        raise WorkerError("PBPK 必须返回 time_h 和 plasma_mg_L 两个数组")
    t = response["time_h"]
    c = response["plasma_mg_L"]
    if not (2 <= len(t) == len(c) <= 10000):
        raise WorkerError("时间和浓度数组长度不一致，或超出 2-10000 条范围")
    ts = [_finite(x, "time_h") for x in t]
    cs = [_finite(x, "plasma_mg_L") for x in c]
    if ts[0] < 0 or any(b <= a for a, b in zip(ts, ts[1:])):
        raise WorkerError("时间轴必须从非负数开始且严格递增")
    if any(x < -1e-9 for x in cs):
        raise WorkerError("浓度存在负数，需检查 PBPK 数值计算")
    summary = response.get("summary", {})
    if not isinstance(summary, dict):
        raise WorkerError("summary 必须是对象")
    # Never guess method-specific NCA statistics from potentially sparse curves.
    return {
        "time_h": ts,
        "plasma_mg_L": [max(0.0, v) for v in cs],
        "summary": summary,
        "tissues": response.get("tissues", {}),
        "engine_version": str(response.get("engine_version", "未提供")),
        "mass_balance_relative_error": response.get("mass_balance_relative_error"),
    }


def predict_properties(smiles: str) -> Dict[str, Any]:
    return validate_ai(invoke_worker("AI_PREDICT_COMMAND", {"smiles": smiles}, 240))


def simulate_beagle(payload: Dict[str, Any]) -> Dict[str, Any]:
    return validate_pbpk(invoke_worker("DOG_PBPK_COMMAND", payload, 360))
