# Second queue, runs beside run_v2_final.ps1 on the same GPU.
# Takes the last stage (Dirichlet alpha 0.1) so the two queues do not meet
# until the first one finishes everything before it. Resume-safe.
Set-Location $PSScriptRoot
& .\run_v2_queue.ps1 -Dataset cic -Models @("BNN-MATCHED","MLP-TWIN","MLP-INT8") -Extra "--partition dirichlet --alpha 0.1"
"=== final_b done $(Get-Date) ===" | Out-File -Append "logs\v2\final_b.log"
