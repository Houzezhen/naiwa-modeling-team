"""Apply the paper's Latin font to saved SVG plots without changing plot data."""

from pathlib import Path
from xml.etree import ElementTree as ET


FIGURES = (
    "fig_validation_regression",
    "fig_temporal_attention",
    "fig_ablation_waterfall",
    "fig_explanation_card",
)
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def main():
    figure_dir = Path(__file__).resolve().parent / "figures"
    for stem in FIGURES:
        source = figure_dir / f"{stem}.svg"
        tree = ET.parse(source)
        for element in tree.iter(f"{{{SVG_NS}}}text"):
            properties = element.get("style", "").split(";")
            properties = [
                "font-family:Times New Roman,SimHei"
                if item.startswith("font-family:") else item
                for item in properties
            ]
            element.set("style", ";".join(properties))
        tree.write(source, encoding="utf-8", xml_declaration=True)
        tree.write(figure_dir / f"{stem}_times.svg", encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main()
