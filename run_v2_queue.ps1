# Protocol v2 training queue. Resume-safe: finished runs are skipped
# by fl_v2.py (done.json), interrupted ones continue from state.pt.
# Seed-major order, so every model has seed 42 before any has seed 43.
#   powershell -File run_v2_queue.ps1 -Dataset nfton
param([string]$Dataset = "nfton",
      [string[]]$Models = @("BNN-MATCHED","MLP-TWIN","MLP","MLP-INT8",
                            "BNN-INT8IO-MATCHED","CNN","LSTM",
                            "BNN-FULL-MATCHED"),
      [int[]]$Seeds = @(42, 43, 44),
      [string]$Extra = "",
      [int]$MaxRetries = 3)
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force -Path "logs\v2" | Out-Null
$q = "logs\v2\queue_$Dataset.log"
"=== queue start $(Get-Date) models=$($Models -join ',') seeds=$($Seeds -join ',') extra='$Extra' ===" | Out-File -Append $q
foreach ($s in $Seeds) {
  foreach ($m in $Models) {
    $tag = "$($m)_s$($s)$($Extra -replace '[^A-Za-z0-9.]','')"
    $ok = $false; $try = 0
    while (-not $ok -and $try -lt $MaxRetries) {
      $try++
      "=== $tag start (attempt $try) $(Get-Date) ===" | Out-File -Append $q
      $args_ = @("code\fl_v2.py","--dataset",$Dataset,"--model",$m,"--seed",$s,"--resume")
      if ($Extra) { $args_ += $Extra.Split(" ") }
      python -W ignore @args_ *>> "logs\v2\$($Dataset)_$tag.log"
      if ($LASTEXITCODE -eq 0) { $ok = $true; "=== $tag done $(Get-Date) ===" | Out-File -Append $q }
      else { "=== $tag FAILED $LASTEXITCODE $(Get-Date) ===" | Out-File -Append $q; Start-Sleep 20 }
    }
  }
}
python -W ignore code\eval_v2.py --dataset $Dataset *>> "logs\v2\eval_$Dataset.log"
"=== queue done $(Get-Date) ===" | Out-File -Append $q
