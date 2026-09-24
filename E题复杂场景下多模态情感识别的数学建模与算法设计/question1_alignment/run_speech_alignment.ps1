param(
    [string]$DataRoot = "",
    [string]$OutputDir = "",
    [string]$AvPython = "",
    [string]$BertPython = "D:\Anaconda3\envs\pytorch\python.exe",
    [string]$TransformersPath = ""
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
if (-not $DataRoot) { $DataRoot = Join-Path $ProjectDir "E题数据\E题数据" }
if (-not $OutputDir) { $OutputDir = Join-Path $ScriptDir "output_speech_aware" }
if (-not $AvPython) { $AvPython = Join-Path $env:TEMP "codex_q1_alignment_env\Scripts\python.exe" }
if (-not $TransformersPath) { $TransformersPath = Join-Path $env:TEMP "codex_q1_pytorch_packages" }
$SourceWork = Join-Path $ScriptDir "work"
$WorkDir = Join-Path $ScriptDir "work_speech_aware"
if (-not (Test-Path -LiteralPath $AvPython)) { throw "Audio/video Python not found: $AvPython" }
if (-not (Test-Path -LiteralPath $BertPython)) { throw "Text Python not found: $BertPython" }
if (-not (Test-Path -LiteralPath (Join-Path $SourceWork "base_features.pkl"))) {
    & $AvPython (Join-Path $ScriptDir "extract_av_features.py") --data-root $DataRoot --work-dir $SourceWork
    if ($LASTEXITCODE -ne 0) { throw "Base features failed" }
}
& $AvPython (Join-Path $ScriptDir "prepare_speech_alignment.py") --data-root $DataRoot --source-work-dir $SourceWork --work-dir $WorkDir
if ($LASTEXITCODE -ne 0) { throw "Acoustic activity preparation failed" }
$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $TransformersPath
try {
    & $BertPython (Join-Path $ScriptDir "force_align_transcripts.py") --data-root $DataRoot --work-dir $WorkDir
    if ($LASTEXITCODE -ne 0) { throw "CTC forced alignment failed" }
    & $BertPython (Join-Path $ScriptDir "extract_text_features.py") --work-dir $WorkDir --device auto
    if ($LASTEXITCODE -ne 0) { throw "Text feature extraction failed" }
}
finally { $env:PYTHONPATH = $PreviousPythonPath }
& $AvPython (Join-Path $ScriptDir "assemble_outputs.py") --work-dir $WorkDir --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Output assembly failed" }
& $AvPython (Join-Path $ScriptDir "export_typical_frames.py") --data-root $DataRoot --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Typical frames failed" }
& $AvPython (Join-Path $ScriptDir "verify_outputs.py") --output-dir $OutputDir
if ($LASTEXITCODE -ne 0) { throw "Output verification failed" }
Write-Output "Speech-aware alignment completed: $OutputDir"
