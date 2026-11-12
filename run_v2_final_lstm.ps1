# Waits for run_v2_final.ps1 to finish, then reruns CIC LSTM for all three seeds.
# LSTM seed 42 failed with CUDA OOM in validation (chunk was 16384, now 4096).
# Finished runs are skipped, interrupted ones resume.
Set-Location $PSScriptRoot
while (-not (Select-String -Path "logs\v2\final.log" -Pattern "final done" -Quiet -ErrorAction SilentlyContinue)) {
  Start-Sleep 120
}
& .\run_v2_queue.ps1 -Dataset cic -Models @("LSTM")
"=== final_lstm done $(Get-Date) ===" | Out-File -Append "logs\v2\final_lstm.log"
