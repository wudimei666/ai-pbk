# 原始 LogP Linux 3.5.6 环境（不自动重建）

旧版 LogP 环境来源是用户此前已验证可运行的服务器环境记录，并非本批压缩包的完整可复现 Conda 锁文件：
Python 3.5.6, RDKit 2018.03.4, TensorFlow 1.14.0, DeepChem Git tag 2.2.0,
NumPy 1.15.2, pandas 0.23.4, scikit-learn 0.20.0, SciPy 1.1.0,
matplotlib 3.0.0, xgboost 0.80。

该软件栈已停止维护，旧版 wheel/Conda 包在 2026 年不一定能再获得；原包没有提供
所有渠道及构建号，因此**不能声称仅凭本清单就能无损、自动重建**。推荐保留原可运行 Linux 环境，
把环境路径导出：`export LOGP_PYTHON=/path/to/old/env/bin/python`。

注意：v0.5 的 `model_workers/logp_worker.py` 错误处理包含 Python 3.6+ f-string，v0.6 改为 Python 3.5 能解析的 `.format()`。
