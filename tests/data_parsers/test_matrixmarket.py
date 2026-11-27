import pytest

from biqbin import MaxCutFromMatrixMarket, QuboFromMatrixMarket


@pytest.mark.parametrize(
    "parser",
    [
        MaxCutFromMatrixMarket,
        QuboFromMatrixMarket,
    ],
)
def test_matrix_market_reader_rejects_non_square_matrix(tmp_path, parser):
    path = tmp_path / "bad.mtx"
    path.write_text("%%MatrixMarket matrix coordinate integer general\n2 3 1\n1 1 1\n")

    with pytest.raises(ValueError):
        parser(str(path)).read()


@pytest.mark.parametrize(
    "parser",
    [
        MaxCutFromMatrixMarket,
        QuboFromMatrixMarket,
    ],
)
def test_matrix_market_reader_rejects_non_integer_values(tmp_path, parser):
    path = tmp_path / "bad.mtx"
    path.write_text("%%MatrixMarket matrix coordinate real general\n2 2 1\n1 2 0.5\n")

    with pytest.raises(ValueError):
        parser(str(path)).read()
