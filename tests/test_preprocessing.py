import pytest

from model.config import COLUMNS, TARGET_COLUMN
from model.preprocessing import load_data


def _write_raw(path, status_values):
    """Writes rows in the space-separated, header-less upstream format."""
    template = (
        "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 "
        "A192 A201"
    )
    path.write_text("\n".join(f"{template} {s}" for s in status_values) + "\n")
    return path


def test_reads_the_raw_layout(tmp_path):
    source = _write_raw(tmp_path / "german.data", [1, 2, 1])

    df = load_data(str(source))

    assert list(df.columns) == COLUMNS
    assert len(df) == 3


def test_maps_the_target_so_bad_credit_is_the_positive_class(tmp_path):
    """Upstream encodes 1 = good, 2 = bad; we model risk."""
    source = _write_raw(tmp_path / "german.data", [1, 2, 2])

    df = load_data(str(source))

    assert df[TARGET_COLUMN].tolist() == [0, 1, 1]


def test_raises_on_an_unexpected_target_value(tmp_path):
    """A silent NaN here would poison training without any error."""
    source = _write_raw(tmp_path / "german.data", [1, 3])

    with pytest.raises(ValueError, match="Unexpected value"):
        load_data(str(source))
