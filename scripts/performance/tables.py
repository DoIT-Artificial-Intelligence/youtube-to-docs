"""Writes the benchmark CSVs as readable Markdown tables for the docs.

Run this file directly to rebuild both tables from the CSVs without
re-running the benchmarks.
"""

import os

import polars as pl

DOCS_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../docs/assets")
)
# time_to_*.py record failed or unconfigured models with this time.
FAILED_TIME = 999.0


def _time(seconds: float) -> str:
    return "failed" if seconds == FAILED_TIME else f"{seconds:.1f} s"


def _cost(dollars: float, seconds: float) -> str:
    if seconds == FAILED_TIME:
        return "–"
    # Cheap summaries cost fractions of a cent, so keep three places there.
    return f"${dollars:.3f}" if dollars < 0.1 else f"${dollars:.2f}"


def _tokens(count: int) -> str:
    return f"{count:,}" if count else "–"


def _size(size: str | None) -> str:
    return size.replace("x", " × ") if size else "–"


def _write(path: str, df: pl.DataFrame, headers: list[str], rows: list[list[str]]):
    dates = sorted({str(d) for d in df["date (today)"].to_list()})
    with open(path, "w") as f:
        f.write(f"Measured on {', '.join(dates)}.\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        # Left-align the model, right-align the numbers.
        f.write("| :--- | " + " | ".join(["---:"] * (len(headers) - 1)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")


def write_summary_table(df: pl.DataFrame, path: str) -> None:
    rows = [
        [
            f"`{r['model']}`",
            _time(r["time (seconds)"]),
            _cost(r["total_cost"], r["time (seconds)"]),
            _tokens(r["input_tokens"]),
            _tokens(r["output_tokens"]),
        ]
        for r in df.iter_rows(named=True)
    ]
    headers = ["Model", "Time", "Cost", "Tokens in", "Tokens out"]
    _write(path, df, headers, rows)


def write_infographic_table(df: pl.DataFrame, path: str) -> None:
    rows = [
        [
            f"`{r['model']}`",
            _time(r["time (seconds)"]),
            _cost(r["total_cost"], r["time (seconds)"]),
            _size(r["size"]),
        ]
        for r in df.iter_rows(named=True)
    ]
    _write(path, df, ["Model", "Time", "Cost", "Size (pixels)"], rows)


if __name__ == "__main__":
    write_summary_table(
        pl.read_csv(os.path.join(DOCS_ASSETS_DIR, "time_to_summarize.csv")),
        os.path.join(DOCS_ASSETS_DIR, "time_to_summarize.md"),
    )
    write_infographic_table(
        pl.read_csv(os.path.join(DOCS_ASSETS_DIR, "time_to_infographic.csv")),
        os.path.join(DOCS_ASSETS_DIR, "time_to_infographic.md"),
    )
