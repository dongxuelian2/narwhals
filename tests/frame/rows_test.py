from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import narwhals as nw
import narwhals.stable.v1 as nw_v1
import narwhals.stable.v2 as nw_v2
from tests.utils import PANDAS_VERSION, is_pd_na

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Literal

    from tests.utils import ConstructorEager

data = {"a": [1, 3, 2], "b": [4, 4, 6], "z": [7.0, 8.0, 9.0]}
data_na = {"a": [None, 3, 2], "b": [4, 4, 6], "z": [7.0, None, 9]}


@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0, 5), (3, 4, 8.0, 6), (2, 6, 9.0, 7)]),
        (
            True,
            [
                {"a": 1, "_b": 4, "z": 7.0, "1": 5},
                {"a": 3, "_b": 4, "z": 8.0, "1": 6},
                {"a": 2, "_b": 6, "z": 9.0, "1": 7},
            ],
        ),
    ],
)
def test_iter_rows(
    request: Any,
    constructor_eager: ConstructorEager,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    if "cudf" in str(constructor_eager):
        request.applymarker(pytest.mark.xfail)

    data = {"a": [1, 3, 2], "_b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "1": [5, 6, 7]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = list(df.iter_rows(named=named))
    assert result == expected


@pytest.mark.filterwarnings(
    "ignore:.*all arguments of to_dict except for the argument:FutureWarning"
)
@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0), (3, 4, 8.0), (2, 6, 9.0)]),
        (
            True,
            [
                {"a": 1, "b": 4, "z": 7.0},
                {"a": 3, "b": 4, "z": 8.0},
                {"a": 2, "b": 6, "z": 9.0},
            ],
        ),
    ],
)
def test_rows(
    constructor_eager: ConstructorEager,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.rows(named=named)
    assert result == expected


@pytest.mark.filterwarnings(
    r"ignore:.*Starting with pandas version 3\.0 all arguments of to_dict"
)
@pytest.mark.parametrize(
    ("named", "expected"),
    [
        (False, [(1, 4, 7.0, 5), (3, 4, 8.0, 6), (2, 6, 9.0, 7)]),
        (
            True,
            [
                {"a": 1, "_b": 4, "z": 7.0, "1": 5},
                {"a": 3, "_b": 4, "z": 8.0, "1": 6},
                {"a": 2, "_b": 6, "z": 9.0, "1": 7},
            ],
        ),
    ],
)
def test_rows_eager(
    constructor_eager: Any,
    named: bool,  # noqa: FBT001
    expected: list[tuple[Any, ...]] | list[dict[str, Any]],
) -> None:
    # posit-dev/py-shiny relies on `.rows(named=False)` to return unnamed rows
    data = {"a": [1, 3, 2], "_b": [4, 4, 6], "z": [7.0, 8.0, 9.0], "1": [5, 6, 7]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    result = df.rows(named=named)
    assert result == expected


def test_rows_with_nulls_unnamed(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # cudf intentionally doesn't support itertuples / iter_rows
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data_na), eager_only=True)
    result = list(df.iter_rows(named=False))
    expected = [(None, 4, 7.0), (3, 4, None), (2, 6, 9.0)]
    for i, row in enumerate(expected):
        for j, value in enumerate(row):
            value_in_result = result[i][j]
            if value is None:
                assert value_in_result is None or is_pd_na(
                    value_in_result
                )  # because float('nan') != float('nan')
            else:
                assert value_in_result == value


def test_rows_with_nulls_named(
    constructor_eager: ConstructorEager, request: pytest.FixtureRequest
) -> None:
    if "cudf" in str(constructor_eager):
        # cudf intentionally doesn't support itertuples / iter_rows
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data_na), eager_only=True)
    result = list(df.iter_rows(named=True))
    expected: list[dict[str, Any]] = [
        {"a": None, "b": 4, "z": 7.0},
        {"a": 3, "b": 4, "z": None},
        {"a": 2, "b": 6, "z": 9.0},
    ]
    for i, row in enumerate(expected):
        for col, value in row.items():
            value_in_result = result[i][col]
            if value is None:
                assert value_in_result is None or is_pd_na(
                    value_in_result
                )  # because float('nan') != float('nan')
            else:
                assert value_in_result == value


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_rows_python_scalars(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    method: Literal["rows", "iter_rows"],
    named: bool,  # noqa: FBT001
) -> None:
    if "cudf" in str(constructor_eager) and method == "iter_rows":
        # cuDF does not support row iteration.
        request.applymarker(pytest.mark.xfail)
    data = {"a": [1, 2], "_b": [True, False], "c": [1.5, 2.5], "1": ["x", "y"]}
    df = nw.from_native(constructor_eager(data), eager_only=True)
    rows = list(getattr(df, method)(named=named))
    expected = [(1, True, 1.5, "x"), (2, False, 2.5, "y")]
    for row, values in zip(rows, expected, strict=True):
        actual = tuple(row.values()) if named else row
        assert actual == values
        assert [type(value) for value in actual] == [int, bool, float, str]
    assert json.loads(json.dumps(rows))


@pytest.mark.parametrize("namespace", [nw, nw_v1, nw_v2])
@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("dtype_backend", ["numpy_nullable", "pyarrow"])
def test_rows_pandas_nullable(
    namespace: ModuleType,
    method: Literal["rows", "iter_rows"],
    named: bool,  # noqa: FBT001
    empty: bool,  # noqa: FBT001
    dtype_backend: Literal["numpy_nullable", "pyarrow"],
) -> None:
    pd = pytest.importorskip("pandas")
    native = pd.DataFrame(
        {
            "a": pd.array([1, None], dtype="Int64"),
            "_b": pd.array([True, None], dtype="boolean"),
            "c": pd.array([1.5, None], dtype="Float64"),
            "1": pd.array(["x", None], dtype="string"),
            "ordinary": [3, 4],
        },
        index=[10, 20],
    )
    if dtype_backend == "pyarrow":
        from packaging.version import Version as PkgVersion

        if PANDAS_VERSION < PkgVersion("2"):
            pytest.skip("pyarrow dtype backend requires pandas>=2")
        pytest.importorskip("pyarrow")
        native = native.convert_dtypes(dtype_backend="pyarrow")
    if empty:
        native = native.head(0)
    original = native.copy(deep=True)
    df = namespace.from_native(native, eager_only=True)
    rows = list(getattr(df, method)(named=named))
    values = [] if empty else [(1, True, 1.5, "x", 3), (None, None, None, None, 4)]
    expected = (
        [dict(zip(native.columns, row, strict=True)) for row in values]
        if named
        else values
    )
    assert rows == expected
    if not empty:
        values = list(rows[0].values()) if named else rows[0]
        assert [type(value) for value in values] == [int, bool, float, str, int]
    assert json.loads(json.dumps(rows, allow_nan=False)) == json.loads(
        json.dumps(expected, allow_nan=False)
    )
    pd.testing.assert_frame_equal(native, original)


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("data", [{}, {"a": []}])
def test_rows_empty(
    constructor_eager: ConstructorEager,
    request: pytest.FixtureRequest,
    method: Literal["rows", "iter_rows"],
    named: bool,  # noqa: FBT001
    data: dict[str, list[Any]],
) -> None:
    if "cudf" in str(constructor_eager) and method == "iter_rows":
        # cuDF does not support row iteration.
        request.applymarker(pytest.mark.xfail)
    df = nw.from_native(constructor_eager(data), eager_only=True)
    assert list(getattr(df, method)(named=named)) == []


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_rows_pandas_zero_columns(
    method: Literal["rows", "iter_rows"],
    named: bool,  # noqa: FBT001
) -> None:
    pd = pytest.importorskip("pandas")
    df = nw.from_native(pd.DataFrame(index=[10, 20]), eager_only=True)
    assert list(getattr(df, method)(named=named)) == []


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("buffer_size", [0, 1, 2, 512])
def test_iter_rows_pandas_nullable_buffer(
    named: bool,  # noqa: FBT001
    buffer_size: int,
) -> None:
    pd = pytest.importorskip("pandas")
    native = pd.DataFrame({"a": pd.array([1, None] * 257, dtype="Int64")})
    df = nw.from_native(native, eager_only=True)
    rows = df.iter_rows(named=named, buffer_size=buffer_size)
    assert iter(rows) is rows
    assert next(rows) == ({"a": 1} if named else (1,))
    expected = [({"a": value} if named else (value,)) for value in [None, 1] * 256]
    expected.append({"a": None} if named else (None,))
    assert list(rows) == expected


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
def test_rows_pandas_nullable_mixed_dtypes(
    method: Literal["rows", "iter_rows"],
    named: bool,  # noqa: FBT001
) -> None:
    pd = pytest.importorskip("pandas")
    native = pd.DataFrame(
        {
            "nullable": pd.array([2**64 - 1, None], dtype="UInt64"),
            "float": [float("nan"), 1.5],
            "datetime": [pd.Timestamp("2020-01-01"), pd.NaT],
        }
    )
    df = nw.from_native(native, eager_only=True)
    rows = list(getattr(df, method)(named=named))
    first, second = (tuple(row.values()) if named else row for row in rows)
    assert type(first[0]) is int
    assert first[0] == 2**64 - 1
    assert first[1] != first[1]  # NaN remains NaN.
    assert first[2] == pd.Timestamp("2020-01-01")
    assert second == (None, 1.5, pd.NaT)


@pytest.mark.parametrize("method", ["rows", "iter_rows"])
@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        ("Int8", 1),
        ("Int16", 1),
        ("Int32", 1),
        ("Int64", 1),
        ("UInt8", 1),
        ("UInt16", 1),
        ("UInt32", 1),
        ("UInt64", 1),
        ("Float32", 1.5),
        ("Float64", 1.5),
        ("boolean", True),
    ],
)
def test_rows_pandas_nullable_numeric_types(
    method: Literal["rows", "iter_rows"],
    named: bool,  # noqa: FBT001
    dtype: str,
    value: float | bool,  # noqa: FBT001
) -> None:
    pd = pytest.importorskip("pandas")
    df = nw.from_native(
        pd.DataFrame({"a": pd.array([value, None], dtype=dtype)}), eager_only=True
    )
    rows = list(getattr(df, method)(named=named))
    assert rows == ([{"a": value}, {"a": None}] if named else [(value,), (None,)])
    assert type(rows[0]["a"] if named else rows[0][0]) is type(value)
    json.dumps(rows, allow_nan=False)
