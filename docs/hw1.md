# HW1 — DVC dataset versioning

## Dataset versions

The dataset has two DVC-tracked versions, committed separately using the same
pointer path, `data/raw/dataset.csv.dvc`:

- **v1** (`d1ee14e`): 6399 rows and 21 columns. MD5
  `d391e79ae4fd5441337128a4a0b1de2e`; size 889803 bytes.
- **v2** (`27ba209`): 7043 rows and the same 21 columns. MD5
  `0f9de68e012bd3aed5fa7cdc9fc421af`; size 977501 bytes.

Version 2 adds 644 customer records. All v1 customer IDs are present in v2,
and their rows are unchanged; no columns were added or modified. This second
batch demonstrates a meaningful dataset update while preserving the original
data.

## Checking out versions on Windows

`md5sum` is not included with standard Windows PowerShell. Use
`Get-FileHash -Algorithm MD5` instead. The pointer is nested in `data/raw/`, so
pass its path explicitly to `dvc checkout`. The commands below were run from
the repository root in PowerShell. `HEAD~1` is the v1 commit and `HEAD` is the
v2 commit.

```powershell
PS> (Get-FileHash data\raw\dataset.csv -Algorithm MD5).Hash
0F9DE68E012BD3AED5FA7CDC9FC421AF

PS> git checkout HEAD~1 -- data\raw\dataset.csv.dvc
PS> uv run dvc checkout data\raw\dataset.csv.dvc --summary
1 file modified
PS> (Get-FileHash data\raw\dataset.csv -Algorithm MD5).Hash
D391E79AE4FD5441337128A4A0B1DE2E

PS> git checkout HEAD -- data\raw\dataset.csv.dvc
PS> uv run dvc checkout data\raw\dataset.csv.dvc --summary
1 file modified
PS> (Get-FileHash data\raw\dataset.csv -Algorithm MD5).Hash
0F9DE68E012BD3AED5FA7CDC9FC421AF
```

Both checksums match the MD5 values recorded in the pointer at the corresponding
commit. `md5sum` is a Unix command; `Get-FileHash` is its PowerShell equivalent.

## Repro skipping 
After running `uv run dvc repro` the console looked like this:

```powershell

PS E:\Sulis cuccok\MLOps\Házi\mlops-project> uv run dvc repro
Stage 'prepare' didn't change, skipping                                                                                                                                                                                                                                    
Stage 'train' didn't change, skipping                                                                                                                                                                                                                                      
Stage 'evaluate' didn't change, skipping                                                                                                                                                                                                                                   
Data and pipelines are up to date.
```

## Prepare stage skipping
After changing some params, the stage "prepare" was skipped.
```powershell
PS E:\Sulis cuccok\MLOps\Házi\mlops-project> uv run dvc repro
Stage 'prepare' didn't change, skipping                                                                                                                                                                                                                                    
Running stage 'train':                                                                                                                                                                                                                                                     
🏃 View run random-forest-training at: http://127.0.0.1:5500/#/experiments/1/runs/14be682bdffc443d8e9472a7632623a0
🧪 View experiment at: http://127.0.0.1:5500/#/experiments/1
Trained random_forest model; MLflow run ID: 14be682bdffc443d8e9472a7632623a0
Updating lock file 'dvc.lock'                                                                                                                                                                                                                                              

Running stage 'evaluate':                                                                                                                                                                                                                                                  
> uv run python -m mlops_project.evaluate
🏃 View run random-forest-training at: http://127.0.0.1:5500/#/experiments/1/runs/14be682bdffc443d8e9472a7632623a0
🧪 View experiment at: http://127.0.0.1:5500/#/experiments/1
Evaluation metrics logged to MLflow run 14be682bdffc443d8e9472a7632623a0: {'accuracy': 0.8013, 'precision': 0.7908, 'recall': 0.8013, 'f1': 0.7926}
Updating lock file 'dvc.lock'                                                                                                                                                                                                                                              

To track the changes with git, run:

        git add dvc.lock

To enable auto staging, run:

        dvc config core.autostage true
Use `dvc push` to send your updates to remote storage.
```

## R4 — Verify data after cloning

In a fresh clone named `hw1-check`, I pulled the DVC-tracked files and
verified the raw dataset checksum in PowerShell. The checksum matches the v2
MD5 in `data/raw/dataset.csv.dvc`.

```powershell
PS E:\Sulis cuccok\MLOps\Házi check\hw1-check> Get-FileHash data/raw/dataset.csv -Algorithm MD5

Algorithm       Hash                                                                   Path
---------       ----                                                                   ----
MD5             0F9DE68E012BD3AED5FA7CDC9FC421AF                                       E:\Sulis cuccok\MLOps\Házi check\hw1-check\data...
```