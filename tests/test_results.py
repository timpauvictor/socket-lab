import csv
from pathlib import Path

import pytest
from openpyxl import load_workbook

from reporting import build_board, plot_board


def test_missing_rate_is_not_zero_and_duplicate_rows_are_not_arbitrary():
    record = {"pair": "en-ro", "setup": "eos-tune", "notes": "not measured"}
    with pytest.raises(ValueError, match="missing repetition"):
        plot_board.loop_rate(record)
    record["notes"] = "0.0% looped"
    assert plot_board.loop_rate(record) == 0
    with pytest.raises(ValueError, match="found 2"):
        plot_board.row([record, record], "en-ro", "eos-tune")


def test_equal_scores_get_labels_on_distinct_bars(tmp_path, monkeypatch):
    rows = plot_board.load()
    for record in rows:
        if record["setup"] == "eos-tune" and record["pair"] in ("en-ro", "ro-en", "de-en"):
            record["chrf"] = "44.36"
    monkeypatch.setattr(plot_board, "load", lambda: rows)
    axes = []
    original = plot_board.plt.subplots
    def capture():
        figure, axis = original()
        axes.append(axis)
        return figure, axis
    monkeypatch.setattr(plot_board.plt, "subplots", capture)
    plot_board.main(tmp_path)
    assert len({text.get_position()[0] for text in axes[0].texts}) == 3


def test_regeneration_keeps_banked_figure_values_and_workbook_rows(tmp_path):
    plot_board.main(tmp_path)
    expected = Path(plot_board.DATA_PATH).read_text(encoding="utf-8")
    actual = (tmp_path / "figure-data.csv").read_text(encoding="utf-8")
    assert actual == expected
    assert all((tmp_path / name).stat().st_size > 0 for name in (
        "fig-g1-trained-pairs.png", "fig-g2-de-fr.png", "fig-g3-loops.png"))
    workbook_path = tmp_path / "board.xlsx"
    build_board.main(workbook_path)
    workbook = load_workbook(workbook_path, read_only=True)
    try:
        with build_board.CSV_PATH.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = list(workbook["chrF board"].values)
        assert len(values) == len(rows) + 2
        assert [row[3] for row in values[2:]] == [float(row["chrf"]) for row in rows]
    finally:
        workbook.close()
