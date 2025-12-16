from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr, PolarsDataType


def parse_into_expr(
    expr: IntoExpr,
    *,
    str_as_lit: bool = False,
    list_as_lit: bool = True,
    dtype: PolarsDataType | None = None,
) -> pl.Expr:
    """Convert user input into a Polars expression."""
    if isinstance(expr, pl.Expr):
        return expr
    elif isinstance(expr, str) and not str_as_lit:
        return pl.col(expr)
    elif isinstance(expr, list) and not list_as_lit:
        return pl.lit(pl.Series(expr), dtype=dtype)
    else:
        return pl.lit(expr, dtype=dtype)


def parse_version(version: str | list[str | int]) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers."""
    if isinstance(version, str):
        version = version.split(".")
    return tuple(int(re.sub(r"\D", "", str(v))) for v in version)
