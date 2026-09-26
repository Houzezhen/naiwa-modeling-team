param([switch]$Export)

$ErrorActionPreference = 'Stop'
$figureDir = Join-Path $PSScriptRoot 'figures'
$pptxPath = Join-Path $figureDir 'fig_model_architecture.pptx'
$pngPath = Join-Path $figureDir 'fig_model_architecture.png'
$pdfPath = Join-Path $figureDir 'fig_model_architecture.pdf'

function Rgb([int]$r, [int]$g, [int]$b) {
    return $r + ($g -shl 8) + ($b -shl 16)
}

function Set-TextStyle($shape, [string]$text, [double]$size, [int]$color, [bool]$bold = $false, [int]$align = 2) {
    $shape.TextFrame.TextRange.Text = $text
    $shape.TextFrame.TextRange.Font.Name = 'Times New Roman'
    $shape.TextFrame.TextRange.Font.NameFarEast = 'Microsoft YaHei'
    $shape.TextFrame.TextRange.Font.Size = $size
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    $shape.TextFrame.TextRange.Font.Bold = if ($bold) { -1 } else { 0 }
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    $shape.TextFrame.VerticalAnchor = 3
    $shape.TextFrame.MarginLeft = 5
    $shape.TextFrame.MarginRight = 5
    $shape.TextFrame.MarginTop = 2
    $shape.TextFrame.MarginBottom = 2
}

function Add-Text($slide, [double]$left, [double]$top, [double]$width, [double]$height, [string]$text, [double]$size, [int]$color, [bool]$bold = $false, [int]$align = 2) {
    $shape = $slide.Shapes.AddTextbox(1, $left, $top, $width, $height)
    Set-TextStyle $shape $text $size $color $bold $align
    return $shape
}

function Add-Box($slide, [double]$left, [double]$top, [double]$width, [double]$height, [int]$fill, [int]$line, [double]$weight = 1.5, [double]$radius = 0.14) {
    $shape = $slide.Shapes.AddShape(5, $left, $top, $width, $height)
    $shape.Fill.ForeColor.RGB = $fill
    $shape.Line.ForeColor.RGB = $line
    $shape.Line.Weight = $weight
    return $shape
}

function Add-Arrow($slide, [double]$x1, [double]$y1, [double]$x2, [double]$y2, [int]$color, [double]$weight = 1.7) {
    $line = $slide.Shapes.AddConnector(1, $x1, $y1, $x2, $y2)
    $line.Line.ForeColor.RGB = $color
    $line.Line.Weight = $weight
    $line.Line.EndArrowheadStyle = 3
    return $line
}

function Add-Stage($slide, [double]$left, [double]$top, [string]$title, [int]$accent) {
    $marker = $slide.Shapes.AddShape(1, $left, ($top + 6), 5, 22)
    $marker.Fill.ForeColor.RGB = $accent
    $marker.Line.Visible = 0
    Add-Text $slide ($left + 12) $top 127 34 $title 12 (Rgb 45 55 72) $false 1 | Out-Null
}

function Add-InputCard($slide, [double]$top, [string]$title, [string]$dim, [string]$mark, [int]$accent, [int]$tint) {
    $card = Add-Box $slide 34 $top 148 82 (Rgb 255 255 255) $accent 2.0
    $bar = $slide.Shapes.AddShape(1, 34, $top, 10, 82)
    $bar.Fill.ForeColor.RGB = $accent
    $bar.Line.Visible = 0
    $badge = $slide.Shapes.AddShape(9, 56, ($top + 18), 43, 43)
    $badge.Fill.ForeColor.RGB = $tint
    $badge.Line.ForeColor.RGB = $accent
    $badge.Line.Weight = 1.2
    Set-TextStyle $badge $mark 17 $accent $true 2
    Add-Text $slide 108 ($top + 14) 63 25 $title 14 (Rgb 42 53 71) $true 2 | Out-Null
    Add-Text $slide 108 ($top + 43) 63 20 $dim 10 (Rgb 100 116 139) $false 2 | Out-Null
}

function Add-EncoderCard($slide, [double]$top, [int]$accent, [int]$tint) {
    $card = Add-Box $slide 226 $top 207 82 (Rgb 255 255 255) $accent 1.7
    $stripe = $slide.Shapes.AddShape(1, 226, $top, 207, 7)
    $stripe.Fill.ForeColor.RGB = $accent
    $stripe.Line.Visible = 0
    Add-Text $slide 239 ($top + 17) 181 23 '两层双向 GRU' 14 (Rgb 42 53 71) $true 2 | Out-Null
    Add-Text $slide 239 ($top + 45) 181 20 '保留有效窗口的时间依赖' 10 (Rgb 100 116 139) $false 2 | Out-Null
    foreach ($i in 0..1) {
        $x = 247 + ($i * 67)
        $mini = $slide.Shapes.AddShape(5, $x, $top + 66, 52, 10)
        $mini.Fill.ForeColor.RGB = $tint
        $mini.Line.ForeColor.RGB = $accent
        $mini.Line.Weight = 0.7
    }
}

function Add-AttentionCard($slide, [double]$top, [int]$accent, [int]$tint) {
    $card = Add-Box $slide 470 $top 177 82 (Rgb 255 255 255) $accent 1.7
    $stripe = $slide.Shapes.AddShape(1, 470, $top, 177, 7)
    $stripe.Fill.ForeColor.RGB = $accent
    $stripe.Line.Visible = 0
    Add-Text $slide 483 ($top + 17) 151 23 '时间注意力' 14 (Rgb 42 53 71) $true 2 | Out-Null
    Add-Text $slide 483 ($top + 45) 151 20 '聚合关键时间窗口' 10 (Rgb 100 116 139) $false 2 | Out-Null
    foreach ($i in 0..4) {
        $dot = $slide.Shapes.AddShape(9, 505 + ($i * 24), $top + 67 - (($i % 2) * 3), 10, 10)
        $dot.Fill.ForeColor.RGB = if ($i -eq 2) { $accent } else { $tint }
        $dot.Line.ForeColor.RGB = $accent
        $dot.Line.Weight = 0.7
    }
    Add-Arrow $slide 516 ($top + 72) 619 ($top + 72) $accent 0.7 | Out-Null
}

function Add-FusionCard($slide, [double]$left, [double]$top, [double]$width, [double]$height, [string]$title, [string]$subtitle, [int]$accent, [int]$fill) {
    $card = Add-Box $slide $left $top $width $height $fill $accent 2.0
    Add-Text $slide ($left + 10) ($top + 23) ($width - 20) 27 $title 16 (Rgb 42 53 71) $true 2 | Out-Null
    Add-Text $slide ($left + 10) ($top + 59) ($width - 20) 21 $subtitle 10 (Rgb 86 96 113) $false 2 | Out-Null
    $icon = $slide.Shapes.AddShape(9, ($left + ($width / 2) - 14), ($top + 8), 28, 12)
    $icon.Fill.ForeColor.RGB = $accent
    $icon.Line.Visible = 0
}

$powerPoint = $null
$presentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Add($false)
    $presentation.PageSetup.SlideWidth = 1280
    $presentation.PageSetup.SlideHeight = 466
    $slide = $presentation.Slides.Add(1, 12)

    $ink = Rgb 42 53 71
    $muted = Rgb 100 116 139
    $line = Rgb 148 163 184
    $orange = Rgb 215 112 76
    $orangeTint = Rgb 252 237 232
    $teal = Rgb 24 151 139
    $tealTint = Rgb 228 247 244
    $blue = Rgb 55 103 166
    $blueTint = Rgb 232 241 252
    $gold = Rgb 184 132 37
    $goldTint = Rgb 255 247 225
    $purple = Rgb 111 82 171
    $purpleTint = Rgb 242 237 251
    $bg = Rgb 255 255 255

    $background = $slide.Shapes.AddShape(1, 0, 0, 1280, 466)
    $background.Fill.ForeColor.RGB = $bg
    $background.Line.Visible = 0

    $leftPanel = Add-Box $slide 20 67 640 390 (Rgb 248 250 252) (Rgb 226 232 240) 0.8
    $rightPanel = Add-Box $slide 677 67 582 390 (Rgb 251 250 253) (Rgb 229 231 235) 0.8
    $leftPanel.Fill.Transparency = 0.08
    $rightPanel.Fill.Transparency = 0.08
    Add-Stage $slide 713 101 '输入模态' $orange
    Add-Stage $slide 860 101 '模态内时序建模' $teal
    Add-Stage $slide 713 153 '样本级融合' $gold
    Add-Stage $slide 860 153 '双任务输出' $purple

    # Connectors are placed first so they remain behind the cards.
    foreach ($y in @(141, 276, 411)) {
        Add-Arrow $slide 182 $y 226 $y $line 1.6 | Out-Null
        Add-Arrow $slide 433 $y 470 $y $line 1.6 | Out-Null
    }
    Add-Arrow $slide 647 141 697 276 $line 1.7 | Out-Null
    Add-Arrow $slide 647 276 697 276 $line 1.7 | Out-Null
    Add-Arrow $slide 647 411 697 276 $line 1.7 | Out-Null
    Add-Arrow $slide 853 276 891 276 $line 1.8 | Out-Null
    Add-Arrow $slide 1059 276 1096 182 $line 1.7 | Out-Null
    Add-Arrow $slide 1059 276 1096 390 $line 1.7 | Out-Null

    Add-InputCard $slide 100 '文本' '768维' 'T' $orange $orangeTint
    Add-InputCard $slide 235 '语音' '74维' 'A' $teal $tealTint
    Add-InputCard $slide 370 '视觉' '35维' 'V' $blue $blueTint
    Add-EncoderCard $slide 100 $orange $orangeTint
    Add-EncoderCard $slide 235 $teal $tealTint
    Add-EncoderCard $slide 370 $blue $blueTint
    Add-AttentionCard $slide 100 $orange $orangeTint
    Add-AttentionCard $slide 235 $teal $tealTint
    Add-AttentionCard $slide 370 $blue $blueTint

    Add-FusionCard $slide 697 222 156 108 '模态门控' '学习样本级可靠性' $gold $goldTint
    Add-FusionCard $slide 891 222 168 108 '特征融合' '归一化与非线性变换' $purple $purpleTint

    $classCard = Add-Box $slide 1096 130 156 104 (Rgb 239 246 255) $blue 1.8
    Add-Text $slide 1107 148 134 27 '情感极性' 16 $blue $true 2 | Out-Null
    Add-Text $slide 1107 180 134 20 '三分类 · Accuracy / F1' 10 $muted $false 2 | Out-Null
    $regCard = Add-Box $slide 1096 338 156 104 (Rgb 255 244 239) $orange 1.8
    Add-Text $slide 1107 356 134 27 '情感强度' 16 $orange $true 2 | Out-Null
    Add-Text $slide 1107 388 134 20 '连续值 · MAE / Pearson' 10 $muted $false 2 | Out-Null

    $presentation.SaveAs($pptxPath, 24)
    if ($Export) {
        $slide.Export($pngPath, 'PNG', 2400, 874)
        $presentation.SaveAs($pdfPath, 32)
    }
    Write-Output $pptxPath
} finally {
    if ($null -ne $presentation) { $presentation.Close() }
    if ($null -ne $powerPoint) { $powerPoint.Quit() }
}
