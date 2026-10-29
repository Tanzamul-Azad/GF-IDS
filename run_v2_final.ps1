# Remaining protocol-v2 work on CICIoT2023 (resume-safe, same rules as run_v2_queue.ps1).
#   1. CNN, LSTM, BNN-INT8IO, BNN-Full x 3 seeds (replaces the single-seed v1 last-round numbers).
#   2. Dirichlet alpha 0.5 / 0.1 for BNN, twin and INT8 x 3 seeds.
#   powershell -NoProfile -ExecutionPolicy Bypass -File run_v2_final.ps1
Set-Location $PSScriptRoot
$q = ".\run_v2_queue.ps1"
& $q -Dataset cic -Models @("BNN-INT8IO-MATCHED","BNN-FULL-MATCHED","CNN","LSTM")
& $q -Dataset cic -Models @("BNN-MATCHED","MLP-TWIN","MLP-INT8") -Extra "--partition dirichlet --alpha 0.5"
& $q -Dataset cic -Models @("BNN-MATCHED","MLP-TWIN","MLP-INT8") -Extra "--partition dirichlet --alpha 0.1"
"=== final done $(Get-Date) ===" | Out-File -Append "logs\v2\final.log"
