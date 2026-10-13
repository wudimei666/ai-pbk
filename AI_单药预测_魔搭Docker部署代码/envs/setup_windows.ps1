param([ValidateSet('logs','pka','logpapp','density','all')][string[]]$Models = @('all'))
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw '未找到 conda，请先在 Windows 安装 Miniforge/Miniconda。' }
if ($Models -contains 'all') { $Models = @('logs','pka','logpapp','density') }
foreach ($m in $Models) {
  $pyver = if ($m -eq 'density') { '3.12.13' } else { '3.11.11' }
  $prefix = Join-Path $project ".venvs\$m"
  $python = Join-Path $prefix 'python.exe'
  if (-not (Test-Path $python)) {
    & conda create --yes --prefix $prefix "python=$pyver" pip
    if ($LASTEXITCODE -ne 0) { throw "创建 $m 失败" }
  }
  $actual = (& $python -c 'import sys;print(".".join(map(str,sys.version_info[:3])))').Trim()
  if ($LASTEXITCODE -ne 0 -or $actual -ne $pyver) { throw "Python 版本不匹配: $m ($actual != $pyver)" }
  $lock = Join-Path $project "envs\requirements_${m}_exact.txt"
  & $python -m pip install --disable-pip-version-check -r $lock
  if ($LASTEXITCODE -ne 0) { throw "依赖安装失败: $m" }
  if ($m -eq 'density') {
    & $python -m pip install --disable-pip-version-check --index-url 'https://download.pytorch.org/whl/cu128' 'torch==2.10.0+cu128'
    if ($LASTEXITCODE -ne 0) { throw 'Density 原 CUDA PyTorch 不兼容当前 Windows，请转 Linux NVIDIA 服务器，不自动降级。' }
  }
  Write-Host "Installed $m: $python"
}
Write-Host '执行: python scripts\match_and_run_v06.py --verify-only'
Write-Host '旧版 LogP 采用原先验证的 Python 3.5.6 Linux 环境，不在此强制重装。'
