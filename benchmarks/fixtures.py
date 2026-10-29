"""Fixed synthetic tables used by tests and the benchmark."""

from __future__ import annotations

import pandas as pd


def sales_frame() -> pd.DataFrame:
    """Six-row sales fixture with one duplicate row and one missing revenue."""
    return pd.DataFrame(
        [
            {"order_date": "2024-01-05", "region": "North", "category": "Widget", "units": 10, "revenue": 100.0},
            {"order_date": "2024-01-05", "region": "North", "category": "Widget", "units": 10, "revenue": 100.0},
            {"order_date": "2024-02-10", "region": "South", "category": "Gadget", "units": 30, "revenue": 300.0},
            {"order_date": "2024-02-20", "region": "South", "category": "Gadget", "units": 20, "revenue": None},
            {"order_date": "2024-03-01", "region": "East", "category": "Widget", "units": 40, "revenue": 500.0},
            {"order_date": "2024-03-18", "region": "East", "category": "Gadget", "units": 50, "revenue": 800.0},
        ]
    )


def constant_frame() -> pd.DataFrame:
    """status is constant. amount has one observed value and one missing value."""
    return pd.DataFrame(
        [
            {"status": "open", "amount": 5},
            {"status": "open", "amount": 5},
            {"status": "open", "amount": None},
        ]
    )


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "units": pd.Series(dtype="float64"),
            "region": pd.Series(dtype="object"),
        }
    )


def outlier_frame() -> pd.DataFrame:
    return pd.DataFrame({"score": [1, 2, 3, 4, 100]})


def invalid_frame() -> pd.DataFrame:
    return pd.DataFrame({"score": ["10", "20", "n/a", "30"]})
