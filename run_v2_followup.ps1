# Ordered v2 work after the NF headline queue finishes.
#   1. CIC core: BNN-MATCHED, MLP-TWIN, MLP-INT8, MLP x 3 seeds. These
#      decide whether the NF result (twin beats BNN on macro F1) holds.
#   2. NF follow-ups: latent_broadcast control, sqrt class weighting,
#      Dirichlet alpha 0.5 / 0.1 for the three key models.
#   (CIC secondary models use v1 last-round checkpoints instead.)
param([string]$WaitFor = "queue_nfton.log")
Set-Location $PSScriptRoot
while (-not (Select-String -Path "logs\v2\$WaitFor" -Pattern "queue done" -Quiet -ErrorAction SilentlyContinue)) {
  Start-Sleep 60
}
$q = ".\run_v2_queue.ps1"
& $q -Dataset cic -Models @("BNN-MATCHED","MLP-TWIN","MLP-INT8","MLP")
& $q -Dataset nfton -Models @("BNN-MATCHED") -Extra "--aggregator latent_broadcast"
& $q -Dataset nfton -Models @("BNN-MATCHED","MLP-TWIN","MLP-INT8") -Extra "--class-weight sqrt"
& $q -Dataset nfton -Models @("BNN-MATCHED","MLP-TWIN","MLP-INT8") -Extra "--partition dirichlet --alpha 0.5"
& $q -Dataset nfton -Models @("BNN-MATCHED","MLP-TWIN","MLP-INT8") -Extra "--partition dirichlet --alpha 0.1"
# CNN, LSTM, BNN-INT8IO and BNN-Full on CIC are not rerun: their v1
# last-round checkpoints (no selection) are scored on the v2 holdout by
# docs/revision_v2/v1_final_on_holdout.py and reported as single-seed.
"=== followup done $(Get-Date) ===" | Out-File -Append "logs\v2\followup.log"
