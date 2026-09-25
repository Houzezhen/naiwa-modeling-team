param([switch]$ExportSvg)

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

    $architectureSvg = Join-Path $figureDir 'fig_model_architecture_ppt.svg'
    $presentation = $powerPoint.Presentations.Add($false)
    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540
    $architecture = $presentation.Slides.Add(1, 12)
    $null = $architecture.Shapes.AddPicture($architectureSvg, $false, $true, 0, 0, 960, 540)
    $architecture.Export((Join-Path $figureDir 'fig_model_architecture.png'), 'PNG', 2400, 1350)
    $presentation.SaveAs((Join-Path $figureDir 'fig_model_architecture.pdf'), 32)
    $presentation.Close()
    $presentation = $null

    foreach ($stem in @('fig_validation_regression', 'fig_temporal_attention', 'fig_ablation_waterfall', 'fig_explanation_card')) {
        $svgPath = Join-Path $figureDir ($stem + '_times.svg')
        [xml]$svg = Get-Content -LiteralPath $svgPath -Raw -Encoding UTF8
        $slideWidth = [double]($svg.svg.width -replace 'pt$', '')
        $slideHeight = [double]($svg.svg.height -replace 'pt$', '')
        $presentation = $powerPoint.Presentations.Add($false)
        $presentation.PageSetup.SlideWidth = $slideWidth
        $presentation.PageSetup.SlideHeight = $slideHeight
        $slide = $presentation.Slides.Add(1, 12)
        $null = $slide.Shapes.AddPicture($svgPath, $false, $true, 0, 0, $slideWidth, $slideHeight)
        $slide.Export((Join-Path $figureDir ($stem + '.png')), 'PNG', 2400, [int](2400 * $slideHeight / $slideWidth))
        $presentation.SaveAs((Join-Path $figureDir ($stem + '.pdf')), 32)
        $presentation.Close()
        $presentation = $null
    }
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $powerPoint) { $powerPoint.Quit() }
}
