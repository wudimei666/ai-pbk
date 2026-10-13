# 原始环境清单的出处（从用户上传档案逐项提取）

| 模型 | 来源文件及内部路径 | Python | 关键运行时 | 安装列表 |
|---|---|---|---|---|
| LogS | `网页直接预测 LogS 用.gz` → `logs_deploy/environment/core_versions.txt` + `requirements_exact.txt`; `环境.gz` → `logs_env/pyvenv.cfg` | 3.11.11 | RDKit 2026.03.6, LightGBM 4.7.0, NumPy 2.4.6, sklearn 1.9.1 | `requirements_logs_exact.txt`（原样） |
| pKa | `pka_DEPLOY_rev5.tar(2).gz` → `pka_deploy_rev5/config/environment.json` 和 `requirements.lock.txt`; `pka_ENV.tar(1).gz` → `pka_env/pyvenv.cfg` | 3.11.11 | RDKit 2025.09.6, LightGBM 4.6.0, NumPy 2.1.3, sklearn 1.6.1 | `requirements_pka_exact.txt` |
| LogPapp | `logpapp_DEPLOY.tar.gz` → `logpapp_deploy/runtime/runtime_lock.txt`（另有 `requirements_dsw.txt` 为非精确列表） | 3.11.11 | RDKit 2022.09.5, XGBoost 3.1.3, NumPy 1.26.4, sklearn 1.8.0 | `requirements_logpapp_exact.txt`（按锁版本整理） |
| Density | `density_part1..4.zip` 组合存档 → `density_mpnn_modelscope/runtime_lock.txt`, `requirements.lock.txt`, `reports/environment.json` | 3.12.13 | RDKit 2026.03.5, NumPy 2.4.3, Torch 2.10.0+cu128 (CUDA 12.8) | `requirements_density_exact.txt`（原样）和官方 torch CUDA 索引 |
| LogP | `logp(1).zip` 代码权重；先前用户成功运行的 Linux 原环境记录 | 3.5.6 | RDKit 2018.03.4, TF 1.14, DeepChem Git tag 2.2.0 | `logp_legacy_reference.txt` 仅参考；原包无完整 Conda 构建锁 |

注：不能用一个统一 Python 环境安装上述不同版本；不能通过更换 RDKit、降级依赖、重训模型或改动特征顺序来规避运行失败。

LogPapp `FULL` 自带 `RECONSTRUCTION_NOT_EXACT_PAPER_MODEL` 声明。LogS 的参考 pH 未得到确认，仍不能自动认定为原论文 pH 7 溶解度。

安装脚本执行实际 pip/conda 下载时**需要目标服务器能连接 Conda 软件源、PyPI 和 PyTorch CUDA 索引**。脚本在依赖不可用时停止并报告失败，不会声称已安装。
