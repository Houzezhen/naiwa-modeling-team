$ErrorActionPreference = 'Stop'
$figureDir = Join-Path $PSScriptRoot 'figures'
$figures = @(
    @('fig_three_question_roadmap.pptx', 'fig_pipeline'),
    @('fig_problem1_analysis.pptx', 'fig_problem1_analysis'),
    @('fig_problem2_analysis.pptx', 'fig_problem2_analysis'),
    @('fig_problem3_analysis.pptx', 'fig_problem3_analysis')
)

$powerPoint = $null
$presentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    foreach ($entry in $figures) {
        $pptxPath = Join-Path $figureDir $entry[0]
        if (-not (Test-Path -LiteralPath $pptxPath)) {
            throw "Editable source is missing: $pptxPath"
        }
        $pdfPath = Join-Path $figureDir ($entry[1] + '.pdf')
        $pngPath = Join-Path $figureDir ($entry[1] + '.png')
        $svgPath = Join-Path $figureDir ($entry[1] + '.svg')
        $presentation = $powerPoint.Presentations.Open($pptxPath, $false, $true, $false)
        $slide = $presentation.Slides.Item(1)
        $exportHeight = if ($entry[1] -eq 'fig_pipeline') { 1968 } else { 605 }
        $slide.Export($pngPath, 'PNG', 2400, $exportHeight)
        $presentation.SaveAs($pdfPath, 32)
        if (Get-Command pdftocairo -ErrorAction SilentlyContinue) {
            & pdftocairo -svg $pdfPath $svgPath
            if ($LASTEXITCODE -ne 0) { throw "SVG export failed: $pdfPath" }
        }
        Write-Output $pdfPath
        $presentation.Close()
        $presentation = $null
    }
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $powerPoint) { $powerPoint.Quit() }
}
