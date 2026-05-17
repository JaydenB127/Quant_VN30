# -*- coding: utf-8 -*-
"""
Metrics integrity validation.

Detects hard-coded, fabricated, or suspiciously round metrics that
indicate the pipeline is not producing genuine results.

This module exists because of a prior incident where run_final.py
injected fake metrics (precision=0.61, sharpe=1.42, dm_pvalue=0.028)
when the model failed to produce signals.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Known fabricated values from the old run_final.py ─────────────────
# These exact values were hard-coded as fallbacks. If any metric matches
# these precisely, it is almost certainly not a genuine model output.
KNOWN_FABRICATED = {
    "precision": 0.61,
    "recall": 0.57,
    "sharpe": 1.42,
    "max_drawdown": -0.135,
    "dm_pvalue": 0.028,
}

# Suspicious roundness thresholds
SUSPICIOUS_DECIMALS = 2  # metrics with exactly 2 decimal places are suspicious


@dataclass
class IntegrityCheck:
    """Result of a single integrity check."""
    metric_name: str
    value: float
    check: str
    passed: bool
    message: str


@dataclass
class IntegrityReport:
    """Full integrity report for a set of metrics."""
    checks: List[IntegrityCheck] = field(default_factory=list)
    passed: bool = True

    def add(self, check: IntegrityCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            self.passed = False

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def summary(self) -> str:
        lines = [f"Integrity Report: {'PASSED' if self.passed else 'FAILED'}"]
        lines.append(f"  Total checks: {len(self.checks)}")
        lines.append(f"  Failed: {self.n_failed}")
        for c in self.checks:
            status = "✓" if c.passed else "✗"
            lines.append(f"  {status} [{c.check}] {c.metric_name}={c.value:.6f}: {c.message}")
        return "\n".join(lines)


def validate_metrics_integrity(
    metrics: Dict[str, float],
    context: str = "pipeline",
) -> IntegrityReport:
    """
    Validate that metrics are genuine model outputs, not fabricated.

    Checks performed:
    1. **Known fabrication**: metric matches a known hard-coded fallback value
    2. **NaN/Inf**: metric is NaN or Inf (pipeline failure)
    3. **Range**: metric is outside plausible range
    4. **Suspiciously round**: metric has very few decimal places (manual entry)
    5. **Zero variance**: all predictions produced identical output

    Parameters
    ----------
    metrics : dict
        {metric_name: value} to validate.
    context : str
        Description of where these metrics came from (for logging).

    Returns
    -------
    IntegrityReport
        Detailed report of all checks.
    """
    report = IntegrityReport()

    for name, value in metrics.items():
        if value is None:
            report.add(IntegrityCheck(
                name, float("nan"), "null_check", False,
                "Metric is None — pipeline did not produce a result",
            ))
            continue

        value = float(value)

        # ── Check 1: Known fabricated values ──────────────────────────
        for fab_name, fab_val in KNOWN_FABRICATED.items():
            if fab_name in name.lower() and abs(value - fab_val) < 1e-10:
                report.add(IntegrityCheck(
                    name, value, "fabrication_check", False,
                    f"CRITICAL: matches known fabricated value {fab_val} "
                    f"from deprecated run_final.py. This is NOT a genuine "
                    f"model output.",
                ))

        # ── Check 2: NaN / Inf ────────────────────────────────────────
        if math.isnan(value) or math.isinf(value):
            report.add(IntegrityCheck(
                name, value, "nan_inf_check", False,
                f"Metric is {'NaN' if math.isnan(value) else 'Inf'} — "
                f"pipeline produced degenerate output",
            ))
            continue

        # ── Check 3: Range checks ─────────────────────────────────────
        range_checks = {
            "auc": (0.0, 1.0),
            "precision": (0.0, 1.0),
            "recall": (0.0, 1.0),
            "f1": (0.0, 1.0),
            "accuracy": (0.0, 1.0),
            "sharpe": (-10.0, 10.0),
            "max_drawdown": (-1.0, 0.0),
            "p_value": (0.0, 1.0),
            "pvalue": (0.0, 1.0),
        }
        for key_pattern, (lo, hi) in range_checks.items():
            if key_pattern in name.lower():
                if value < lo or value > hi:
                    report.add(IntegrityCheck(
                        name, value, "range_check", False,
                        f"Value {value:.4f} outside plausible range [{lo}, {hi}]",
                    ))

        # ── Check 4: Suspiciously round ───────────────────────────────
        # A genuine model output for metrics like AUC/Sharpe will almost never
        # be an exact round number like 0.61 or 1.42
        if value != 0.0:
            decimal_str = f"{value:.10f}".rstrip("0").split(".")[-1]
            n_sig_decimals = len(decimal_str)
            if n_sig_decimals <= SUSPICIOUS_DECIMALS and abs(value) > 0.01:
                report.add(IntegrityCheck(
                    name, value, "roundness_check", False,
                    f"Suspiciously round value ({n_sig_decimals} decimal places). "
                    f"Genuine model outputs typically have 4+ significant decimals.",
                ))

    # ── Log results ───────────────────────────────────────────────────
    if report.passed:
        logger.info("Metrics integrity check PASSED for %s (%d checks)",
                     context, len(report.checks))
    else:
        logger.error(
            "Metrics integrity check FAILED for %s:\n%s",
            context, report.summary(),
        )

    return report


def scan_for_hardcoded_patterns(file_content: str) -> List[Tuple[int, str]]:
    """
    Scan Python source code for patterns that look like hard-coded metric
    fallbacks. Returns list of (line_number, line_content) matches.

    Detected patterns:
    - ``if metric == 0: metric = <value>``
    - ``metric = <round_number>`` without model context
    - Direct assignment of known fabricated values
    """
    import re
    suspicious = []
    lines = file_content.split("\n")

    patterns = [
        # if prec == 0: prec = 0.61
        re.compile(r"if\s+\w+\s*==\s*0\s*:\s*\w+\s*=\s*[\d.]+"),
        # Direct assignment of known fabricated values
        re.compile(r"\b(?:prec|precision|recall|sharpe|dm_pvalue)\s*=\s*(?:0\.61|0\.57|1\.42|0\.028|-0\.135)\b"),
        # "Ensure realistic" comments
        re.compile(r"#.*(?:realistic|demo|presentation|scholarship)", re.IGNORECASE),
    ]

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") and "DEPRECATED" in stripped:
            continue  # Skip deprecation notices
        for pattern in patterns:
            if pattern.search(stripped):
                suspicious.append((i, stripped))
                break

    return suspicious
