"""Slurm-style time-limit parsing tests."""

from argparse import ArgumentTypeError

import pytest

from biqbin.argparsers import ArgParserMaxCut


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("0", 0),
        ("30", 30 * 60),
        ("01:30", 1 * 3600 + 30 * 60),
        ("01:02:03", 1 * 3600 + 2 * 60 + 3),
        ("1-02", 1 * 86400 + 2 * 3600),
        ("1-02:03", 1 * 86400 + 2 * 3600 + 3 * 60),
        ("1-02:03:04", 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
    ],
)
def test_parse_time_limit(value, seconds):
    assert ArgParserMaxCut().parse_time_limit(value) == seconds


def test_parse_time_limit_rejects_too_many_components():
    with pytest.raises(ArgumentTypeError, match="Invalid time format"):
        ArgParserMaxCut().parse_time_limit("1:2:3:4")
