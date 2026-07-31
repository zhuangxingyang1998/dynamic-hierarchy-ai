param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,
    [Parameter(Mandatory = $true)]
    [ValidateSet("Status", "Pause", "Resume", "Stop")]
    [string]$Action
)

$ErrorActionPreference = "Stop"
$run = [System.IO.Path]::GetFullPath($RunDir)
$control = [System.IO.Path]::Combine($run, "control")
[System.IO.Directory]::CreateDirectory($control) | Out-Null

if ($Action -eq "Status") {
    Get-Content -LiteralPath ([System.IO.Path]::Combine($run, "status.json")) -Raw
    exit 0
}

$pause = [System.IO.Path]::Combine($control, "PAUSE")
$resume = [System.IO.Path]::Combine($control, "RESUME")
$stop = [System.IO.Path]::Combine($control, "STOP")
if ($Action -eq "Pause") {
    [System.IO.File]::WriteAllText($pause, [DateTime]::UtcNow.ToString("o"))
} elseif ($Action -eq "Resume") {
    [System.IO.File]::Delete($pause)
    [System.IO.File]::Delete($stop)
    [System.IO.File]::WriteAllText($resume, [DateTime]::UtcNow.ToString("o"))
} elseif ($Action -eq "Stop") {
    [System.IO.File]::WriteAllText($stop, [DateTime]::UtcNow.ToString("o"))
}
@{ run_dir = $run; action = $Action; written_at = [DateTime]::UtcNow.ToString("o") } | ConvertTo-Json
