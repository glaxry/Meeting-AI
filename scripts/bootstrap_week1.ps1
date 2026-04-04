$ErrorActionPreference = "Stop"

$envName = "meeting-ai-w1"
$envRoot = Join-Path $env:USERPROFILE ".conda\envs\$envName"
$pythonExe = Join-Path $envRoot "python.exe"

conda create -y -n $envName python=3.10 pip
& $pythonExe -m pip install -r requirements.txt
& $pythonExe -m pip install -e .

Write-Host "Week 1 environment is ready: $envName"
