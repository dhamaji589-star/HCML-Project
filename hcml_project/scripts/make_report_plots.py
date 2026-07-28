"""Create report-ready plots from summarized MAD22 results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create report plots.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/summary_method_results.csv"),
        help="Summary CSV created by summarize_method_results.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/success_rate_by_method.png"),
        help="Output plot path.",
    )
    return parser.parse_args()


def parse_success(value: str) -> float:
    successes, trials = value.split("/")
    return 100.0 * int(successes) / int(trials)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def main() -> None:
    args = parse_args()
    rows = read_rows(args.summary_csv)

    methods = [row["method"] for row in rows]
    neg_rates = [parse_success(row["negfacediff_success"]) for row in rows]
    adapt_rates = [parse_success(row["adaptdiff_success"]) for row in rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1200, 650
    margin_left, margin_right = 95, 45
    margin_top, margin_bottom = 70, 120
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    base_y = margin_top + plot_height
    scale = plot_height / 100.0

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
        small_font = ImageFont.truetype("arial.ttf", 20)
        tiny_font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        tiny_font = ImageFont.load_default()

    draw.text((margin_left, 25), "Hidden-identity recovery success rate", fill="black", font=font)
    draw.rectangle(
        (width - 330, 30, width - 305, 55),
        fill="#4c78a8",
        outline="#4c78a8",
    )
    draw.text((width - 295, 30), "NegFaceDiff", fill="black", font=small_font)
    draw.rectangle(
        (width - 160, 30, width - 135, 55),
        fill="#f58518",
        outline="#f58518",
    )
    draw.text((width - 125, 30), "AdaptDiff", fill="black", font=small_font)

    for tick in range(0, 101, 20):
        y = base_y - tick * scale
        draw.line((margin_left, y, width - margin_right, y), fill="#dddddd", width=1)
        draw.text((35, y - 12), f"{tick}", fill="black", font=small_font)
    draw.text((20, margin_top - 35), "Success (%)", fill="black", font=small_font)
    draw.line((margin_left, margin_top, margin_left, base_y), fill="black", width=2)
    draw.line((margin_left, base_y, width - margin_right, base_y), fill="black", width=2)

    group_width = plot_width / len(methods)
    bar_width = group_width * 0.25
    for index, method in enumerate(methods):
        center = margin_left + group_width * index + group_width / 2
        for offset, value, color in [
            (-bar_width * 0.65, neg_rates[index], "#4c78a8"),
            (bar_width * 0.65, adapt_rates[index], "#f58518"),
        ]:
            x0 = center + offset - bar_width / 2
            x1 = center + offset + bar_width / 2
            y0 = base_y - value * scale
            draw.rectangle((x0, y0, x1, base_y), fill=color)
            draw.text((x0 - 2, y0 - 25), f"{value:.1f}", fill="black", font=tiny_font)
        draw.text((center - 55, base_y + 18), method, fill="black", font=small_font)

    image.save(args.output)
    print(f"Plot written: {args.output}")


if __name__ == "__main__":
    main()
