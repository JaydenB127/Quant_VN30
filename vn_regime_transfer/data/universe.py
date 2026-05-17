# -*- coding: utf-8 -*-
"""
Point-in-time universe management.

Handles historical constituents of the VN30 index to prevent
survivorship bias in backtesting.
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

from ..config import CFG

logger = logging.getLogger(__name__)


# ── Historical VN30 Rebalancing Events ────────────────────────────────
# To prevent survivorship bias, we track when stocks entered/exited the index.
# This list is not exhaustive for the entire 2015-2024 period, but sets up
# the architecture and includes recent major changes.
# In a full production environment, this would be loaded from a database.
VN30_CHANGES = [
    {"date": "2024-02-05", "added": [], "removed": []},  # No change
    {"date": "2023-08-07", "added": ["SHB", "SSB"], "removed": ["NVL", "PDR"]},
    {"date": "2023-02-06", "added": ["BCM"], "removed": ["KDH"]},
    {"date": "2022-08-01", "added": ["VIB"], "removed": ["PNJ"]},
    {"date": "2021-08-02", "added": ["ACB", "GVR", "SAB"], "removed": ["RE", "SBT", "TCH"]},
    {"date": "2021-02-01", "added": ["BVH", "PDR", "TPB"], "removed": ["ROS", "SAB", "EIB"]},
    {"date": "2020-08-03", "added": ["KDH", "TCH"], "removed": ["CTD", "BVH"]},
    {"date": "2020-02-03", "added": ["PLX", "POW"], "removed": ["DPM", "GMD"]},
    {"date": "2019-08-05", "added": ["BID", "BVH"], "removed": ["CII", "DHG"]},
]


def get_vn30_constituents(target_date: str | pd.Timestamp) -> List[str]:
    """
    Get the VN30 constituents active on a specific date.

    Reconstructs the index backwards from the current static list
    defined in config, applying historical changes in reverse.

    Parameters
    ----------
    target_date : str or pd.Timestamp
        The date for which to get the constituents.

    Returns
    -------
    list of str
        List of VN30 tickers active on that date.
    """
    target_date = pd.to_datetime(target_date)
    
    # Start with the current (2024) base list
    current_list = set(CFG.universe.tickers)

    # Sort changes by date descending (newest to oldest)
    # We walk backwards in time, reversing the events
    changes = sorted(VN30_CHANGES, key=lambda x: pd.to_datetime(x["date"]), reverse=True)

    for event in changes:
        event_date = pd.to_datetime(event["date"])
        
        # If the target date is BEFORE the event date, we must reverse the event
        # (remove the 'added' stocks, add back the 'removed' stocks)
        if target_date < event_date:
            for ticker in event["added"]:
                current_list.discard(ticker)
            for ticker in event["removed"]:
                current_list.add(ticker)

    return sorted(list(current_list))


def get_all_historical_tickers() -> List[str]:
    """
    Get a master list of all tickers that were EVER in the VN30 index
    during the covered period. Useful for downloading all required data.
    """
    all_tickers = set(CFG.universe.tickers)
    for event in VN30_CHANGES:
        for ticker in event["removed"]:
            all_tickers.add(ticker)
            
    return sorted(list(all_tickers))
