param(
    [string]$Config = "",
    [int]$TrainingSeed = 0,
    [string]$CandidateResult = "",
    [string]$ResumeRun = "",
    [string]$PreparedRunDir = "",
    [string]$NewRunDir = "",
    [string]$LaunchReceipt = "",
    [string]$RuntimePython = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($PSScriptRoot, ".."))
$python = if ($RuntimePython) {
    [System.IO.Path]::GetFullPath($RuntimePython)
} else {
    [System.IO.Path]::Combine($projectRoot, ".venv-directml", "Scripts", "python.exe")
}
if (-not [System.IO.File]::Exists($python)) {
    throw "DirectML Python is missing: $python"
}

function Get-LiveStage1Worker {
    param([string]$RunDirectory)
    $candidatePids = @()
    $statusPath = [System.IO.Path]::Combine($RunDirectory, "status.json")
    if ([System.IO.File]::Exists($statusPath)) {
        try {
            $statusRecord = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            if ($statusRecord.pid) {
                $candidatePids += [int]$statusRecord.pid
            }
        } catch {
            # A concurrently replaced heartbeat may be retried through pid.json.
        }
    }
    $recordPath = [System.IO.Path]::Combine($RunDirectory, "pid.json")
    if ([System.IO.File]::Exists($recordPath)) {
        try {
            $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            if ($record.worker_pid) {
                $candidatePids += [int]$record.worker_pid
            }
        } catch {
            # A malformed record cannot establish a live worker.
        }
    }
    foreach ($candidatePid in ($candidatePids | Select-Object -Unique)) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $candidatePid" -ErrorAction SilentlyContinue
        if (
            $process -and
            $process.CommandLine -and
            $process.CommandLine.IndexOf("stage1_worker.py", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $process.CommandLine.IndexOf($RunDirectory, [StringComparison]::OrdinalIgnoreCase) -ge 0
        ) {
            return $process
        }
    }
    return $null
}

function Get-PendingStage1Launcher {
    param([string]$RunDirectory)
    $recordPath = [System.IO.Path]::Combine($RunDirectory, "pid.json")
    if (-not [System.IO.File]::Exists($recordPath)) {
        return $null
    }
    try {
        $record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
        $launcherPid = if ($record.launcher_pid) { [int]$record.launcher_pid } else { [int]$record.pid }
        $startedAt = [DateTime]::Parse([string]$record.started_at).ToUniversalTime()
        if (([DateTime]::UtcNow - $startedAt).TotalSeconds -gt 60) {
            return $null
        }
        $process = Get-CimInstance Win32_Process -Filter "ProcessId = $launcherPid" -ErrorAction SilentlyContinue
        if (
            $process -and
            $process.CommandLine -and
            $process.CommandLine.IndexOf("stage1_worker.py", [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $process.CommandLine.IndexOf($RunDirectory, [StringComparison]::OrdinalIgnoreCase) -ge 0
        ) {
            return $process
        }
    } catch {
        return $null
    }
    return $null
}

if ($ResumeRun -and $PreparedRunDir) {
    throw "ResumeRun and PreparedRunDir are mutually exclusive."
}

$preparedCampaignRun = $false
if ($ResumeRun) {
    $runDir = [System.IO.Path]::GetFullPath($ResumeRun)
    $snapshot = [System.IO.Path]::Combine($runDir, "snapshot")
    if (-not [System.IO.Directory]::Exists($snapshot)) {
        throw "Snapshot is missing: $snapshot"
    }
    $priorPidPath = [System.IO.Path]::Combine($runDir, "pid.json")
    if (-not [System.IO.File]::Exists($priorPidPath)) {
        throw "Resume requires the original pid.json config record: $priorPidPath"
    }
    $priorRecord = Get-Content -LiteralPath $priorPidPath -Raw | ConvertFrom-Json
    $configPath = [string]$priorRecord.config
    if ($priorRecord.training_seed) {
        $TrainingSeed = [int]$priorRecord.training_seed
    }
    if ($priorRecord.candidate_result) {
        $CandidateResult = [string]$priorRecord.candidate_result
    }
    $preparedCampaignRun = [bool]$priorRecord.prepared_campaign_run
    if (-not [System.IO.File]::Exists($configPath)) {
        throw "Original snapshot config is missing: $configPath"
    }
} elseif ($PreparedRunDir) {
    $preparedCampaignRun = $true
    if (-not $Config) {
        throw "Prepared campaign launch requires a snapshot-relative Config."
    }
    if ([System.IO.Path]::IsPathRooted($Config)) {
        throw "Prepared campaign Config must be snapshot-relative."
    }
    $runDir = [System.IO.Path]::GetFullPath($PreparedRunDir)
    if (-not [System.IO.Directory]::Exists($runDir)) {
        throw "Prepared campaign run directory is missing: $runDir"
    }
    $allowedEntries = @("snapshot", "campaign-receipt.json")
    $unexpectedEntries = @(
        Get-ChildItem -LiteralPath $runDir -Force |
            Where-Object { $_.Name -notin $allowedEntries }
    )
    if ($unexpectedEntries.Count -ne 0) {
        throw "Prepared campaign run contains unexpected launch evidence."
    }
    $snapshot = [System.IO.Path]::Combine($runDir, "snapshot")
    if (-not [System.IO.Directory]::Exists($snapshot)) {
        throw "Prepared campaign snapshot is missing: $snapshot"
    }
    $configPath = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine(
            $snapshot,
            $Config.Replace(
                "/",
                [System.IO.Path]::DirectorySeparatorChar
            )
        )
    )
    $snapshotPrefix = $snapshot.TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar
    ) + [System.IO.Path]::DirectorySeparatorChar
    if (-not $configPath.StartsWith(
        $snapshotPrefix,
        [StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Prepared campaign Config escapes the canonical snapshot."
    }
    if (-not [System.IO.File]::Exists($configPath)) {
        throw "Prepared campaign config is missing: $configPath"
    }
    if (-not $CandidateResult) {
        $sourceConfigRecord = Get-Content -LiteralPath $configPath -Raw |
            ConvertFrom-Json
        if ($sourceConfigRecord.candidate_prerequisite_result_path) {
            $declaredCandidate = [string]$sourceConfigRecord.candidate_prerequisite_result_path
            $CandidateResult = [System.IO.Path]::GetFullPath(
                [System.IO.Path]::Combine($projectRoot, $declaredCandidate)
            )
        }
    }
} else {
    if (-not $Config) {
        throw "A revised Stage 1 config must be selected explicitly."
    }
    $sourceConfigPath = if ([System.IO.Path]::IsPathRooted($Config)) {
        [System.IO.Path]::GetFullPath($Config)
    } else {
        [System.IO.Path]::GetFullPath(
            [System.IO.Path]::Combine($projectRoot, $Config)
        )
    }
    if (-not [System.IO.File]::Exists($sourceConfigPath)) {
        throw "Selected config is missing: $sourceConfigPath"
    }
    if (-not $CandidateResult) {
        $sourceConfigRecord = Get-Content -LiteralPath $sourceConfigPath -Raw |
            ConvertFrom-Json
        if ($sourceConfigRecord.candidate_prerequisite_result_path) {
            $declaredCandidate = [string]$sourceConfigRecord.candidate_prerequisite_result_path
            $CandidateResult = if ([System.IO.Path]::IsPathRooted($declaredCandidate)) {
                [System.IO.Path]::GetFullPath($declaredCandidate)
            } else {
                [System.IO.Path]::GetFullPath(
                    [System.IO.Path]::Combine($projectRoot, $declaredCandidate)
                )
            }
        }
    }
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $runDir = if ($NewRunDir) {
        [System.IO.Path]::GetFullPath($NewRunDir)
    } else {
        [System.IO.Path]::Combine($projectRoot, "runs", "stage1-$stamp")
    }
    if ([System.IO.Directory]::Exists($runDir)) {
        throw "New run directory already exists: $runDir"
    }
    & $python ([System.IO.Path]::Combine($projectRoot, "scripts", "create_stage1_snapshot.py")) `
        --project-root $projectRoot --run-dir $runDir
    if ($LASTEXITCODE -ne 0) {
        throw "Snapshot creation failed with exit code $LASTEXITCODE"
    }
    $snapshot = [System.IO.Path]::Combine($runDir, "snapshot")
    $configPath = [System.IO.Path]::Combine($snapshot, $Config.Replace("/", [System.IO.Path]::DirectorySeparatorChar))
}

$pidPath = [System.IO.Path]::Combine($runDir, "pid.json")
$launchLockPath = [System.IO.Path]::Combine($runDir, ".launch.lock")
try {
    $launchLock = [System.IO.File]::Open(
        $launchLockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
} catch {
    throw "Another launch operation already holds the per-run launch lock: $launchLockPath"
}
try {
$liveWorker = Get-LiveStage1Worker -RunDirectory $runDir
if ($liveWorker) {
    throw "Run already has a live worker: PID $($liveWorker.ProcessId)"
}
$pendingLauncher = Get-PendingStage1Launcher -RunDirectory $runDir
if ($pendingLauncher) {
    throw "Run startup is still pending through launcher PID $($pendingLauncher.ProcessId)"
}

$worker = [System.IO.Path]::Combine($snapshot, "scripts", "stage1_worker.py")
$stdoutPath = [System.IO.Path]::Combine($runDir, "stdout.log")
$stderrPath = [System.IO.Path]::Combine($runDir, "stderr.log")
$launchId = [Guid]::NewGuid().ToString("N")
$supportsWorkerRegistration = [System.IO.File]::ReadAllText($worker).Contains('parser.add_argument("--launch-id")')
$arguments = @($worker, "--run-dir", $runDir, "--config", $configPath)
if ($supportsWorkerRegistration) {
    $arguments += @("--launch-id", $launchId)
}
if ($TrainingSeed -ne 0) {
    $arguments += @("--training-seed", [string]$TrainingSeed)
}
if ($CandidateResult) {
    $candidateResultPath = [System.IO.Path]::GetFullPath($CandidateResult)
    if (-not [System.IO.File]::Exists($candidateResultPath)) {
        throw "Candidate result is missing: $candidateResultPath"
    }
    $arguments += @("--candidate-result", $candidateResultPath)
}
if ($ResumeRun) {
    $arguments += "--resume"
}
$quotedArguments = $arguments | ForEach-Object { '"' + $_.Replace('"', '\"') + '"' }
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = [System.IO.Path]::Combine($snapshot, "src")
try {
    $process = Start-Process -FilePath $python `
        -ArgumentList $quotedArguments `
        -WorkingDirectory $snapshot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
} finally {
    $env:PYTHONPATH = $oldPythonPath
}

$pidRecord = @{
    pid = $process.Id
    launcher_pid = $process.Id
    worker_pid = $null
    pid_role = "launcher_redirector"
    launch_id = $launchId
    record_state = if ($supportsWorkerRegistration) { "launching" } else { "launcher_only_legacy_snapshot" }
    worker_registration_supported = $supportsWorkerRegistration
    started_at = [DateTime]::UtcNow.ToString("o")
    run_dir = $runDir
    snapshot = $snapshot
    config = $configPath
    training_seed = if ($TrainingSeed -ne 0) { $TrainingSeed } else { $null }
    candidate_result = if ($CandidateResult) { $candidateResultPath } else { $null }
    resume = [bool]$ResumeRun
    prepared_campaign_run = $preparedCampaignRun
    runtime_python = $python
} | ConvertTo-Json
$pidTemporary = $pidPath + "." + $PID + ".tmp"
[System.IO.File]::WriteAllText($pidTemporary, $pidRecord)
if ([System.IO.File]::Exists($pidPath)) {
    $pidBackup = $pidPath + "." + $PID + ".backup"
    [System.IO.File]::Replace($pidTemporary, $pidPath, $pidBackup)
    [System.IO.File]::Delete($pidBackup)
} else {
    [System.IO.File]::Move($pidTemporary, $pidPath)
}

$launchResult = @{
    launcher_pid = $process.Id
    worker_pid = $null
    launch_id = $launchId
    worker_registration_supported = $supportsWorkerRegistration
    run_dir = $runDir
    snapshot = $snapshot
    config = $configPath
    training_seed = if ($TrainingSeed -ne 0) { $TrainingSeed } else { $null }
    candidate_result = if ($CandidateResult) { $candidateResultPath } else { $null }
    resume = [bool]$ResumeRun
    prepared_campaign_run = $preparedCampaignRun
    runtime_python = $python
}
if ($LaunchReceipt) {
    $receiptPath = [System.IO.Path]::GetFullPath($LaunchReceipt)
    $receiptDirectory = [System.IO.Path]::GetDirectoryName($receiptPath)
    [System.IO.Directory]::CreateDirectory($receiptDirectory) | Out-Null
    $receiptTemporary = $receiptPath + "." + $PID + ".tmp"
    [System.IO.File]::WriteAllText(
        $receiptTemporary,
        ($launchResult | ConvertTo-Json)
    )
    if ([System.IO.File]::Exists($receiptPath)) {
        [System.IO.File]::Replace(
            $receiptTemporary,
            $receiptPath,
            $null
        )
    } else {
        [System.IO.File]::Move($receiptTemporary, $receiptPath)
    }
}
$launchResult | ConvertTo-Json
} finally {
    $launchLock.Dispose()
}
