[CmdletBinding()]
param(
    [ValidateSet("Check", "Preprocess", "Train", "Test", "OdChord", "All")]
    [string]$Stage = "Check",
    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",
    [switch]$OverwritePreprocess
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
Set-Location -LiteralPath $RepoRoot

$RawPath = "C:\硕士\数据集\Rome\taxi_february.txt"
$RomeRoot = "outputs\road_full\rome"
$InstructionRoot = Join-Path $RomeRoot "instruction_data"
$RealRoot = Join-Path $RomeRoot "real"
$TrainMeta = Join-Path $InstructionRoot "rome_instruction_meta.train.jsonl"
$ValMeta = Join-Path $InstructionRoot "rome_instruction_meta.val.jsonl"
$TestMeta = Join-Path $InstructionRoot "rome_instruction_meta.test.jsonl"
$TrainReal = Join-Path $RealRoot "rome_train_real_trajectories.npz"
$ValReal = Join-Path $RealRoot "rome_val_real_trajectories.npz"
$TestReal = Join-Path $RealRoot "rome_test_real_trajectories.npz"
$RunRoot = "outputs\paper_full\rome_table1"
$TrainOutput = Join-Path $RunRoot "selector"
$TestOutput = Join-Path $RunRoot "eval_test"
$CacheRoot = Join-Path $RunRoot "cache"

function Invoke-Python {
    param([string[]]$Arguments)
    & python -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

function Assert-Path {
    param([string]$Path, [string]$Label)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }
}

function Show-Inputs {
    Write-Host "Rome raw data:       $RawPath"
    Write-Host "Processed data root: $RomeRoot"
    Write-Host "Final metrics:        $(Join-Path $TestOutput 'test_full\table1_metrics.json')"
    Write-Host "GPU protocol:         candidate_topm=16, memory=50000, prior=10000, seed=7"
    Assert-Path $RawPath "Rome raw dataset"
    try {
        & nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    } catch {
        Write-Warning "nvidia-smi is unavailable; use -Device cpu only for debugging, not the matched final run."
    }
}

function Invoke-Preprocess {
    Assert-Path $RawPath "Rome raw dataset"
    if ((Test-Path -LiteralPath $RomeRoot) -and -not $OverwritePreprocess) {
        throw "$RomeRoot already exists. Verify it or rerun with -OverwritePreprocess."
    }
    $Arguments = @(
        "-m", "langdirecttraj.data.preprocess_public_taxi_raw",
        "--dataset", "rome",
        "--raw-path", $RawPath,
        "--output-dir", "outputs\road_full",
        "--max-samples", "0",
        "--sample-strategy", "sequential",
        "--split-ratios", "0.8,0.1,0.1",
        "--split-seed", "20260617",
        "--target-len", "200",
        "--min-traj-len", "20",
        "--min-distance-km", "0.5",
        "--max-duration-min", "180",
        "--max-speed-kmh", "140",
        "--max-gap-seconds", "600",
        "--storage-format", "compact"
    )
    if ($OverwritePreprocess) {
        $Arguments += "--overwrite"
    }
    Invoke-Python $Arguments
}

function Get-CommonSelectorArguments {
    return @(
        "--memory-real-npz", $TrainReal,
        "--train-query-real-npz", $TrainReal,
        "--val-real-npz", $ValReal,
        "--test-real-npz", $TestReal,
        "--train-input", $TrainMeta,
        "--val-input", $ValMeta,
        "--test-input", $TestMeta,
        "--val-counterfactual-input", $ValMeta,
        "--test-counterfactual-input", $TestMeta,
        "--memory-samples", "50000",
        "--train-query-samples", "0",
        "--prior-samples", "10000",
        "--candidate-topm", "16",
        "--feature-mode", "od_len_angle",
        "--retrieval-method", "kdtree",
        "--target-length-scale", "1.15",
        "--target-length-knn", "16",
        "--target-length-weight", "10.0",
        "--grid-num", "32",
        "--od-grid", "6",
        "--length-bins", "32",
        "--min-group-count", "3",
        "--global-mix", "0.1",
        "--cell-weight", "0.25",
        "--transition-weight", "1.0",
        "--length-ratio-weight", "2.0",
        "--prototype-weight", "0.5",
        "--mode-index-weight", "0.005",
        "--prior-score-feature",
        "--table1-cost-feature",
        "--disable-semantic-costs",
        "--selection-policy", "weighted_sum",
        "--selection-logit-weight", "1.0",
        "--semantic-weight", "0.0",
        "--batch-refine-cycles", "3",
        "--batch-density-refine-penalty", "0.1",
        "--batch-length-refine-penalty", "4.0",
        "--batch-refine-topk", "32",
        "--batch-density-hot-cells", "64",
        "--batch-refine-length-target", "train",
        "--batch-refine-score-source", "selection",
        "--hidden-dim", "192",
        "--batch-size", "256",
        "--seed", "7",
        "--device", $Device,
        "--skip-table3",
        "--skip-table4"
    )
}

function Assert-ProcessedData {
    Assert-Path $TrainMeta "Rome train metadata"
    Assert-Path $ValMeta "Rome validation metadata"
    Assert-Path $TestMeta "Rome test metadata"
    Assert-Path $TrainReal "Rome train trajectories"
    Assert-Path $ValReal "Rome validation trajectories"
    Assert-Path $TestReal "Rome test trajectories"
}

function Invoke-Train {
    Assert-ProcessedData
    New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
    $Arguments = @("scripts\train_semantic_route_memory_reranker.py")
    $Arguments += Get-CommonSelectorArguments
    $Arguments += @(
        "--output-dir", $TrainOutput,
        "--eval-split", "val",
        "--val-output-name", "val_10000",
        "--eval-samples", "10000",
        "--teacher-ce-weight", "1.0",
        "--teacher-soft-weight", "1.0",
        "--teacher-score-weight", "0.2",
        "--teacher-temperature", "0.35",
        "--teacher-score-mode", "route_first_soft",
        "--teacher-density-weight", "1.0",
        "--teacher-density-marginal-weight", "1.0",
        "--teacher-transition-weight", "2.0",
        "--teacher-length-weight", "2.5",
        "--teacher-raw-length-weight", "0.0",
        "--teacher-length-marginal-weight", "0.0",
        "--teacher-use-real-train-length",
        "--teacher-semantic-cost-weight", "0.0",
        "--student-density-cost-weight", "0.5",
        "--student-transition-cost-weight", "0.8",
        "--student-length-cost-weight", "4.0",
        "--student-raw-length-cost-weight", "1.5",
        "--stream-train-features",
        "--train-route-memory-candidate-cache", (Join-Path $CacheRoot "train_top16_candidates.npz"),
        "--train-feature-cache", (Join-Path $CacheRoot "train_top16_features"),
        "--train-feature-chunk-size", "1024",
        "--candidate-cache-chunk-size", "2048",
        "--epochs", "80",
        "--lr", "0.001",
        "--weight-decay", "0.0001"
    )
    Invoke-Python $Arguments
}

function Invoke-Test {
    Assert-ProcessedData
    $Checkpoint = Join-Path $TrainOutput "model.pt"
    Assert-Path $Checkpoint "Rome selector checkpoint"
    New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
    $Arguments = @("scripts\train_semantic_route_memory_reranker.py")
    $Arguments += Get-CommonSelectorArguments
    $Arguments += @(
        "--eval-only",
        "--checkpoint", $Checkpoint,
        "--output-dir", $TestOutput,
        "--eval-split", "test",
        "--test-output-name", "test_full",
        "--eval-samples", "0",
        "--route-memory-candidate-cache", (Join-Path $CacheRoot "test_full_top16_candidates.npz"),
        "--candidate-cache-chunk-size", "2048"
    )
    Invoke-Python $Arguments
}

function Invoke-OdChordBaseline {
    Assert-ProcessedData
    $Output = Join-Path $RunRoot "od_chord"
    New-Item -ItemType Directory -Force -Path $Output | Out-Null
    Invoke-Python @(
        "-m", "langdirecttraj.data.export_baseline_dataset",
        "--dataset", "rome",
        "--input", $TestMeta,
        "--output-dir", $Output,
        "--max-samples", "0",
        "--target-len", "200",
        "--grid-size", "64",
        "--continuous-mode", "references",
        "--no-compress"
    )
    Invoke-Python @(
        "-m", "langdirecttraj.eval.distribution_metrics",
        "--real", $TestReal,
        "--generated", (Join-Path $Output "rome\rome_continuous_eval.npz"),
        "--real-key", "real_trajectories",
        "--generated-key", "references",
        "--grid-num", "32",
        "--length-bins", "32",
        "--output", (Join-Path $Output "table1_metrics.json")
    )
}

Show-Inputs
switch ($Stage) {
    "Check" { }
    "Preprocess" { Invoke-Preprocess }
    "Train" { Invoke-Train }
    "Test" { Invoke-Test }
    "OdChord" { Invoke-OdChordBaseline }
    "All" {
        Invoke-Preprocess
        Invoke-Train
        Invoke-Test
        Invoke-OdChordBaseline
    }
}
