import pytest
from types import SimpleNamespace

import biqbin.data_parsers as data_parsers
from biqbin import QuboFromQPLIB


def _fake_qplib_problem(cons_type, var_type):
    return SimpleNamespace(
        obj=SimpleNamespace(
            sense=data_parsers.pyqplib.types.Sense.MINIMIZE,
        ),
        description=SimpleNamespace(
            cons_type=cons_type,
            var_type=var_type,
            obj_type=data_parsers.pyqplib.ProblemObjType.LINEAR,
        ),
    )
    
def test_qplib_reader_rejects_constrained_problem(
    tmp_path, monkeypatch
):
    problem = _fake_qplib_problem(
        data_parsers.pyqplib.ProblemConsType.LINEAR,
        data_parsers.pyqplib.ProblemVarType.BINARY,
    )

    monkeypatch.setattr(
        data_parsers.pyqplib,
        'read_problem',
        lambda _: problem,
    )

    with pytest.raises(ValueError, match='unconstrained'):
        QuboFromQPLIB(str(tmp_path / 'unused.qplib')).read()
        

def test_qplib_reader_rejects_non_binary_problem(
    tmp_path, monkeypatch
):
    problem = _fake_qplib_problem(
        data_parsers.pyqplib.ProblemConsType.UNCONSTRAINED,
        data_parsers.pyqplib.ProblemVarType.INTEGER,
    )

    monkeypatch.setattr(
        data_parsers.pyqplib,
        'read_problem',
        lambda _: problem,
    )

    with pytest.raises(ValueError, match='not binary'):
        QuboFromQPLIB(str(tmp_path / 'unused.qplib')).read()