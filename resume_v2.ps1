# Resume all protocol-v2 work after any stop (crash, restart, power cut).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\resume_v2.ps1
#
# Safe to run at any time:
#   * finished runs are skipped (done.json);
#   * an interrupted run continues from its last completed round, with
#     the same RNG state, so the result equals an uninterrupted run;
#   * a damaged checkpoint falls back to the previous round's copy;
#   * if the queue is already running, this script does nothing.
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path "logs\v2" | Out-Null
if (Select-String -Path "logs\v2\followup.log" -Pattern "followup done" -Quiet -ErrorAction SilentlyContinue) {
  Write-Host "All v2 runs have finished. Nothing to resume."
  "=== resume_v2 $(Get-Date): all work already finished ===" | Out-File -Append "logs\v2\resume.log"
  exit 0
}
$busy = Get-CimInstance Win32_Process | Where-Object {
  $_.ProcessId -ne $PID -and $_.CommandLine -and (
    $_.CommandLine -like "*fl_v2.py*" -or
    $_.CommandLine -like "*run_v2_queue.ps1*" -or
    $_.CommandLine -like "*run_v2_followup.ps1*" -or
    ($_.CommandLine -like "*resume_v2.ps1*" -and $_.Name -eq "powershell.exe"))
}
if ($busy) {
  Write-Host "v2 training is already running (PID $($busy.ProcessId -join ', ')). Nothing to do."
  exit 0
}
"=== resume_v2 start $(Get-Date) ===" | Out-File -Append "logs\v2\resume.log"
& .\run_v2_queue.ps1 -Dataset nfton
& .\run_v2_followup.ps1
"=== resume_v2 finished $(Get-Date) ===" | Out-File -Append "logs\v2\resume.log"
