# 自有 AI 药物性质单药预测网站（ModelScope Docker 预部署工程）

**状态：部署工程已生成，未成功在魔搭云端构建或公开上线。** 不会假装所有模型已经预测成功。

本工程只负责 **一个 SMILES → 7 项药物性质**，不做犬 PBPK、不批量预测、不重新训练。

## 仓库已有的资料

GitHub [wudimei666/ai-pbk v1.0 Release](https://github.com/wudimei666/ai-pbk/releases/tag/v1.0) 内已有 LogS / pKa / LogPapp / Density 分包 / LogP 真实模型。
本部署版使用 `scripts/fetch_release_assets.py` 获取其中**原训练权重**，并以 SHA256/文件大小核对。**不需要用户重复上传这些模型到 Code**。

## 部署路线（选魔搭 Docker 创空间）

1. 把 **本工程内的文件和文件夹**上传到 GitHub `ai-pbk` 仓库的 Code/main 根目录（不要把 ZIP 本身当成代码上传）。
2. 在 ModelScope 创建 `Docker` 类型创空间，服务端口设为 **7860**；将网页代码同步到该创空间，或采用平台支持的文件夹上传。
3. 第一次构建会联网获取 GitHub Release 附件、Conda 包和 PyPI 包；需要相应网络访问和磁盘空间。
4. 建立独立环境：logs / pka / logpapp 各自 Python 3.11.11。它们有独立 RDKit 等版本，不会混装。
5. 运行后程序会做 Python/包版本核验和真实 `CCO` 单分子推理。成功才返回预测值；失败则显示“未安装原环境”等状态。
6. 如果有足够下载带宽和存储空间，可构建时追加 `--build-arg INSTALL_DENSITY_CPU=1`，自动下载四个 Density 大分包并尝试 CPU PyTorch 移植。移植模式不会冒充原 CUDA 训练环境，必须通过真实 `CCO` 推理测试。**默认不开启 Density，以避免初次构建占用过多云资源。**

### 当前重要限制

- **Density** 原环境为 PyTorch `2.10.0+cu128`，不适合通用 CPU；可选的 `INSTALL_DENSITY_CPU=1` 会显式标注 CPU 移植，并校验原权重真实推理。默认不下载 4 个大分包以避免资源浪费。
- **LogP** 使用 Python 3.5.6 / TensorFlow 1.14 / DeepChem 2.2 的老环境，当前不包含可重建的 2026 年完整锁文件；不会偷偷换成 RDKit LogP。
- **IDR** 现有上传包里尚未发现可直接调用的最终训练 `.joblib` 权重。不会填假数据。
- **LogS** 训练数据未统一证实为 pH7 条件，请勿写成 LogS@pH7。
- **Papp** 是已训练的文献方法重建模型，不是原作者完全相同的权重。
- `Dockerfile` 和下载器尚未在 ModelScope 构建器中成功部署：Docker 构建可能因为旧 RDKit、Conda 包版本或云端资源不足而失败，必须查看构建日志。

### 本地 Docker 预检查

```bash
docker build -t single-smiles-ai .
docker run --rm -p 7860:7860 single-smiles-ai
```

网页地址 `http://127.0.0.1:7860`。云端则由魔搭提供访问网址。

## 不上传什么

不要在公开仓库上传私人 Token、原始研究数据或不打算开放许可的其他模型。GitHub Release 已是公开下载入口，任何人都能取得其中的权重；请自行确认知识产权及数据授权。
