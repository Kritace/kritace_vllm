# Render + quant-check the full inference flow diagram (headless Chrome, Windows side).
#
# Why this exists: the SVG must be exported by a real browser (mermaid + KaTeX layout).
# On Windows chrome.exe is a GUI-subsystem program, so stdout cannot be captured --
# the page POSTs its artifacts back to a local HTTP server instead.
#
# NOTE: this file must stay ASCII-only. Windows PowerShell reads .ps1 using the ANSI
# code page (GBK here); UTF-8 Chinese comments without BOM swallow the following
# newline or quote and produce bogus "missing terminator" errors.
#
# Usage:
#   pwsh -File run_render.ps1
#   pwsh -File run_render.ps1 -Python 'C:\path\to\python.exe'

param(
  [string]$Diagrams = $PSScriptRoot,
  [string]$Python   = 'D:\dsh-home\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe',
  [int]   $Port     = 8791,
  [string]$Only     = '',
  [switch]$KeepTemp
)

$ErrorActionPreference = 'Stop'

# The page writes UTF-8 JSON. Windows PowerShell 5.1's Get-Content defaults to the
# ANSI code page (GBK here) and mangles it, so ConvertFrom-Json fails on the Chinese
# node labels. Read the bytes and decode as UTF-8 explicitly.
function Read-Json([string]$Path) {
  $text = [System.Text.Encoding]::UTF8.GetString([System.IO.File]::ReadAllBytes($Path))
  return ($text | ConvertFrom-Json)
}

$chrome = @(
  'C:\Program Files\Google\Chrome\Application\chrome.exe',
  'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $chrome) { throw 'chrome.exe not found' }
if (-not (Test-Path $Python)) { throw "python not found: $Python" }

$status  = Join-Path $Diagrams '_status.json'
$metrics = Join-Path $Diagrams '_metrics.json'
$rendered = Join-Path $Diagrams '_rendered.svg'

# ---- 1. stage runtime files -------------------------------------------------
Copy-Item (Join-Path $Diagrams 'render_page.html') (Join-Path $Diagrams '_render.html') -Force
Copy-Item (Join-Path $Diagrams 'zoom_page.html')   (Join-Path $Diagrams '_zoom.html')   -Force
if (-not (Test-Path (Join-Path $Diagrams '_mermaid.min.js'))) {
  throw '_mermaid.min.js missing (download mermaid@11 dist bundle into the diagrams dir)'
}

# ---- 2. serve the diagrams dir ---------------------------------------------
Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# 每张图：图源 / 产物前缀（前缀 _ 表示沿用旧命名 _metrics.json 等）
$jobs = @(
  @{ Src = 'full_flow.mmd';    Out = '_' },
  @{ Src = 'notes.mmd';        Out = 'notes_' },
  @{ Src = 'cg_flow.mmd';      Out = 'cg_' },
  @{ Src = 'compile_flow.mmd'; Out = 'cc_' },
  @{ Src = 'boundary_flow.mmd'; Out = 'bd_' },
  @{ Src = 'merged_flow.mmd';   Out = 'mg_' }
)
$allFail = @()

$srv = Start-Process -FilePath $Python `
  -ArgumentList (Join-Path $Diagrams 'mm_render_server.py'), $Diagrams, $Port `
  -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $Diagrams '_server.log') `
  -RedirectStandardError  (Join-Path $Diagrams '_server.err.log')
Start-Sleep -Seconds 2

try {
  foreach ($j in $jobs) {
    if ($Only -and $j.Src -ne $Only) { continue }
    $pfx = $j.Out
    $st  = Join-Path $Diagrams ($pfx + 'status.json')
    $mt  = Join-Path $Diagrams ($pfx + 'metrics.json')
    Remove-Item $st, $mt -ErrorAction SilentlyContinue

    # ---- 3. headless render -------------------------------------------------
    # Do NOT pass --virtual-time-budget here: virtual time starves mermaid's async
    # rendering (mermaid.render never returns and the status file is never written).
    $ud = Join-Path $env:TEMP ('ff-chrome-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
    Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 800
    Start-Process -FilePath $chrome -ArgumentList @(
      '--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
      "--user-data-dir=$ud", '--window-size=1600,1200',
      "http://127.0.0.1:$Port/_render.html?src=$($j.Src)&out=$pfx"
    ) -WindowStyle Hidden

    $deadline = (Get-Date).AddSeconds(240)
    while ((Get-Date) -lt $deadline -and -not (Test-Path $st)) { Start-Sleep -Seconds 3 }
    if (-not (Test-Path $st)) { throw "timeout waiting for $($pfx)status.json ($($j.Src))" }

    $s = Read-Json $st
    if (-not $s.ok) { throw "render failed ($($j.Src)): $($s.error)" }

    # ---- 4. report + gate ---------------------------------------------------
    $m = Read-Json $mt
    ''
    "================ render metrics: $($j.Src) ================"
    "canvas          : $($m.viewBox[2]) x $($m.viewBox[3])"
    "nodes/clusters  : $($m.nodes) / $($m.clusters)"
    "edges           : $($m.edgePaths)"
    "katex formulas  : $($m.katexNodes)"
    "fonts loaded    : $($m.fontsLoaded)/$($m.fontsTotal)"
    "crossings       : $($m.crossings)"
    "clipped         : $($m.clipped)"
    "shapeClips      : $($m.shapeClips)"
    "nodeOverlapArea : $($m.nodeOverlapArea)"
    "clusterOverlap  : $($m.clusterOverlapArea)"
    "longestEdge     : $($m.longestEdge)"
    "minMargin L/R   : $($m.minMarginL) / $($m.minMarginR)"
    '================================================'
    ''

    if ($m.clipped -ne 0)            { $allFail += "$($j.Src): clipped=$($m.clipped)" }
    if ($m.shapeClips -ne 0)         { $allFail += "$($j.Src): shapeClips=$($m.shapeClips)" }
    if ($m.nodeOverlapArea -ne 0)    { $allFail += "$($j.Src): nodeOverlapArea=$($m.nodeOverlapArea)" }
    if ($m.clusterOverlapArea -ne 0) { $allFail += "$($j.Src): clusterOverlapArea=$($m.clusterOverlapArea)" }
    if ($m.crossings -ne 0) {
      Write-Host "NOTE: $($j.Src) crossings=$($m.crossings) (non-blocking; inspect metrics file)"
    }
  }

  if ($allFail.Count) { throw ('GATE FAILED: ' + ($allFail -join '; ')) }

  # ---- 5. assemble the offline viewers -------------------------------------
  $bArgs = @((Join-Path $Diagrams 'build_flow_viewer.py'))
  if ($Only) { $bArgs += @('--only', $Only) }
  & $Python @bArgs
  if ($LASTEXITCODE -ne 0) { throw "build_flow_viewer.py failed ($LASTEXITCODE)" }

  'DONE'
}
finally {
  if ($srv -and -not $srv.HasExited) { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue }
  Get-Process chrome -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
  if (-not $KeepTemp) {
    Remove-Item (Join-Path $Diagrams '_render.html') -ErrorAction SilentlyContinue
    Remove-Item (Join-Path $Diagrams '_zoom.html')   -ErrorAction SilentlyContinue
  }
}
