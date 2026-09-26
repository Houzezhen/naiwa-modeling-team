param([switch]$ExportSvg, [switch]$ModelOnly)

$ErrorActionPreference = 'Stop'
$figureDir = Join-Path $PSScriptRoot 'figures'
$figures = @(
    @('fig_three_question_roadmap.pptx', 'fig_pipeline'),
    @('fig_problem1_analysis.pptx', 'fig_problem1_analysis'),
    @('fig_problem2_analysis.pptx', 'fig_problem2_analysis'),
    @('fig_problem3_analysis.pptx', 'fig_problem3_analysis')
)
if ($ModelOnly) { $figures = @() }

$powerPoint = $null
$presentation = $null

function Set-LatinFont($shapes) {
    foreach ($shape in $shapes) {
        if ($shape.Type -eq 6) {
            Set-LatinFont $shape.GroupItems
        } elseif ($shape.HasTextFrame -ne 0 -and $shape.TextFrame.HasText -ne 0) {
            $shape.TextFrame.TextRange.Font.Name = 'Times New Roman'
            $shape.TextFrame.TextRange.Font.NameFarEast = 'Microsoft YaHei'
        }
    }
}

try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    foreach ($entry in $figures) {
        $pptxPath = Join-Path $figureDir $entry[0]
        if (-not (Test-Path -LiteralPath $pptxPath)) {
            throw "Editable source is missing: $pptxPath"
        }
        $pdfPath = Join-Path $figureDir ($entry[1] + '.pdf')
        $pngPath = Join-Path $figureDir ($entry[1] + '.png')
        $presentation = $powerPoint.Presentations.Open($pptxPath, $false, $true, $false)
        $slide = $presentation.Slides.Item(1)
        Set-LatinFont $slide.Shapes
        $exportHeight = if ($entry[1] -eq 'fig_pipeline') { 1968 } else { 605 }
        $slide.Export($pngPath, 'PNG', 2400, $exportHeight)
        $presentation.SaveAs($pdfPath, 32)
        if ($ExportSvg -and (Get-Command pdftocairo -ErrorAction SilentlyContinue)) {
            $svgPath = Join-Path $figureDir ($entry[1] + '.svg')
            & pdftocairo -svg $pdfPath $svgPath
            if ($LASTEXITCODE -ne 0) { throw "SVG export failed: $pdfPath" }
        }
        Write-Output $pdfPath
        $presentation.Close()
        $presentation = $null
    }

    $architecturePptx = Join-Path $figureDir 'fig_model_architecture.pptx'
    if (-not (Test-Path -LiteralPath $architecturePptx)) {
        throw "Editable model architecture is missing: $architecturePptx"
    }
    $presentation = $powerPoint.Presentations.Open($architecturePptx, $false, $true, $false)
    $architecture = $presentation.Slides.Item(1)
    Set-LatinFont $architecture.Shapes
    $architecture.Export((Join-Path $figureDir 'fig_model_architecture.png'), 'PNG', 2400, 956)
    $presentation.SaveAs((Join-Path $figureDir 'fig_model_architecture.pdf'), 32)
    $presentation.Close()
    $presentation = $null

    # Statistical plots keep the PDFs produced directly by Matplotlib. Reimporting
    # their SVGs into PowerPoint can drop raster layers and change Latin fonts.
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $powerPoint) { $powerPoint.Quit() }
}
