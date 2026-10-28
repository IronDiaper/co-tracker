Set-Location -LiteralPath $PSScriptRoot
$sw = [Diagnostics.Stopwatch]::StartNew()

$Image = "cotracker3"
$Demo = ""
$Rebuild = $false
$Gpu = $false
$MakeRef = $false
$Extra = @()

function Show-Usage {
    Write-Host @'
Usage:
  .\run.ps1                      build the image if needed and run all tests
  .\run.ps1 --demo video.mp4     demo on your video -> results\demo\<name>\
  .\run.ps1 --rebuild            rebuild the image first
  .\run.ps1 --gpu                GPU image, run with --gpus all
  .\run.ps1 --make-reference     recreate reference\
  .\run.ps1 -k shift             other arguments go to pytest (or run_demo.py)
'@
}

$i = 0
while ($i -lt $args.Count) {
    $a = [string]$args[$i]
    if ($a -match '^--?demo$') {
        if ($i + 1 -ge $args.Count) { Write-Host "error: --demo needs a path to a video"; exit 2 }
        $Demo = [string]$args[$i + 1]; $i += 2
    }
    elseif ($a -match '^--?rebuild$') { $Rebuild = $true; $i++ }
    elseif ($a -match '^--?gpu$') { $Gpu = $true; $i++ }
    elseif ($a -match '^--?make-?reference$') { $MakeRef = $true; $i++ }
    elseif ($a -match '^--?(h|help)$') { Show-Usage; exit 0 }
    elseif ($a -eq '--') {
        if ($i + 1 -lt $args.Count) { $Extra = @($args[($i + 1)..($args.Count - 1)]) }
        break
    }
    else {
        $Extra = @($args[$i..($args.Count - 1)])
        break
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "error: docker not found, install Docker Desktop" -ForegroundColor Red; exit 1
}
$null = docker info 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "error: Docker daemon is not running (start Docker Desktop)" -ForegroundColor Red; exit 1
}

$Tag = "${Image}:test"
$BuildArgs = @()
$GpuArgs = @()
if ($Gpu) {
    $Tag = "${Image}-gpu:test"
    $BuildArgs = @("--build-arg", "REQUIREMENTS=requirements-gpu.txt")
    $GpuArgs = @("--gpus", "all", "-e", "COTRACKER_DEVICE=cuda")
}

Write-Host "== CoTracker3 in Docker =="
$null = docker image inspect $Tag 2>&1
if ($Rebuild -or $LASTEXITCODE -ne 0) {
    Write-Host "[1/3] Building image $Tag (first build takes 3-5 min, needs internet)"
    & docker build --platform linux/amd64 @BuildArgs --target test -t $Tag .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "error: docker build failed, see TESTING.md -> troubleshooting" -ForegroundColor Red; exit 1
    }
}
else {
    Write-Host "[1/3] Image $Tag found (use --rebuild to rebuild)"
}

New-Item -ItemType Directory -Force -Path results | Out-Null
$Results = (Resolve-Path -LiteralPath results).Path
$RunArgs = @("run", "--rm", "--network", "none", "--platform", "linux/amd64") + $GpuArgs +
           @("-v", "${Results}:/app/results")

if ($Demo) {
    if (-not (Test-Path -LiteralPath $Demo -PathType Leaf)) { Write-Host "error: video not found: $Demo"; exit 2 }
    $video = Get-Item -LiteralPath $Demo
    Write-Host "[2/3] Running demo on $Demo (offline)"
    & docker @RunArgs -v "$($video.DirectoryName):/input:ro" $Tag `
        python scripts/run_demo.py --video "/input/$($video.Name)" @Extra
    $rc = $LASTEXITCODE
    $Out = Join-Path "results\demo" $video.BaseName
    $DoneMsg = "DEMO DONE!"
}
elseif ($MakeRef) {
    Write-Host "[2/3] Creating reference\ (offline)"
    New-Item -ItemType Directory -Force -Path reference | Out-Null
    $Ref = (Resolve-Path -LiteralPath reference).Path
    & docker @RunArgs -v "${Ref}:/app/reference" $Tag python scripts/make_reference.py
    $rc = $LASTEXITCODE
    $Out = "reference"
    $DoneMsg = "REFERENCE DONE!"
}
else {
    Write-Host "[2/3] Running tests (offline, --network none)"
    $Ref = (Resolve-Path -LiteralPath reference).Path
    & docker @RunArgs -v "${Ref}:/app/reference:ro" $Tag python -m pytest @Extra
    $rc = $LASTEXITCODE
    $Out = "results\tests"
    $DoneMsg = "TESTS DONE!"
}

Write-Host "[3/3] Summary"
if (Test-Path -LiteralPath $Out) {
    $base = (Resolve-Path -LiteralPath $Out).Path
    Write-Host "Files in .\${Out}:"
    Get-ChildItem -LiteralPath $base -Recurse -File | Sort-Object FullName |
        ForEach-Object { "  " + $_.FullName.Substring($base.Length + 1) }
}
Write-Host ("Elapsed: {0:N0}s" -f $sw.Elapsed.TotalSeconds)
if ($rc -eq 0) {
    Write-Host $DoneMsg -ForegroundColor Green
}
else {
    Write-Host "FAILED (exit code $rc), see the output above and TESTING.md -> troubleshooting" -ForegroundColor Red
}
exit $rc
