# -*- coding: utf-8 -*-
"""
Tests for metrics integrity validation.

Ensures that fabricated/hardcoded metrics are detected and that the
validation pipeline rejects suspicious outputs.
"""
import numpy as np
import pytest
from pathlib import Path

from vn_regime_transfer.validation.integrity import (
    validate_metrics_integrity,
    scan_for_hardcoded_patterns,
    KNOWN_FABRICATED,
    IntegrityReport,
)


class TestKnownFabrication:
    """Detect metrics that match known fabricated values."""

    def test_detects_fabricated_precision(self):
        metrics = {"precision": 0.61}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed, "Should detect fabricated precision=0.61"

    def test_detects_fabricated_sharpe(self):
        metrics = {"sharpe": 1.42}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed, "Should detect fabricated sharpe=1.42"

    def test_detects_fabricated_dm_pvalue(self):
        metrics = {"dm_pvalue": 0.028}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_detects_fabricated_recall(self):
        metrics = {"recall": 0.57}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_detects_fabricated_drawdown(self):
        metrics = {"max_drawdown": -0.135}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_all_fabricated_values_detected(self):
        """Every known fabricated value must be caught."""
        for name, value in KNOWN_FABRICATED.items():
            metrics = {name: value}
            report = validate_metrics_integrity(metrics, context="test")
            assert not report.passed, f"Failed to detect fabricated {name}={value}"


class TestGenuineMetrics:
    """Genuine model outputs should pass validation."""

    def test_genuine_classification_metrics(self):
        metrics = {
            "precision": 0.6137,
            "recall": 0.5723,
            "auc_roc": 0.6842,
            "f1_score": 0.4219,
        }
        report = validate_metrics_integrity(metrics, context="test")
        assert report.passed, f"Genuine metrics should pass:\n{report.summary()}"

    def test_genuine_portfolio_metrics(self):
        metrics = {
            "sharpe": 1.4237,
            "max_drawdown": -0.1348,
        }
        report = validate_metrics_integrity(metrics, context="test")
        assert report.passed, f"Genuine metrics should pass:\n{report.summary()}"

    def test_zero_metrics_are_ok(self):
        """Zero is a valid metric value (model predicts nothing)."""
        metrics = {"precision": 0.0, "recall": 0.0}
        report = validate_metrics_integrity(metrics, context="test")
        assert report.passed


class TestRangeChecks:
    """Metrics outside plausible ranges should fail."""

    def test_auc_above_1(self):
        metrics = {"auc_roc": 1.5}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_negative_precision(self):
        metrics = {"precision": -0.1}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_sharpe_extreme(self):
        metrics = {"sharpe": 25.0}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_positive_drawdown(self):
        metrics = {"max_drawdown": 0.5}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed


class TestNaNInf:
    """NaN and Inf metrics should fail."""

    def test_nan_metric(self):
        metrics = {"auc_roc": float("nan")}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_inf_metric(self):
        metrics = {"sharpe": float("inf")}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed

    def test_none_metric(self):
        metrics = {"precision": None}
        report = validate_metrics_integrity(metrics, context="test")
        assert not report.passed


class TestSourceScanner:
    """Scan source code for hardcoded metric patterns."""

    def test_detects_hardcoded_fallback(self):
        source = '''
if prec == 0: prec = 0.61
if rec == 0: rec = 0.57
if sharpe == 0: sharpe = 1.42
'''
        results = scan_for_hardcoded_patterns(source)
        assert len(results) >= 3, f"Should detect 3+ patterns, got {len(results)}"

    def test_detects_demo_comments(self):
        source = '# Ensure realistic demo bounds for scholarship presentation\n'
        results = scan_for_hardcoded_patterns(source)
        assert len(results) >= 1

    def test_clean_code_passes(self):
        source = '''
metrics = classification_metrics(y_true, y_pred, y_proba)
sharpe = daily.mean() / daily.std() * np.sqrt(252)
'''
        results = scan_for_hardcoded_patterns(source)
        assert len(results) == 0, f"Clean code should pass, got: {results}"

    def test_scan_run_final_content(self):
        """The old run_final.py content must be flagged."""
        old_content = """
if prec == 0: prec = 0.61
if rec == 0: rec = 0.57
if sharpe == 0: sharpe = 1.42
if dd == 0: dd = -0.135
res = {'precision': round(prec,3), 'recall': round(rec,3), 'sharpe': round(sharpe,2), 'max_drawdown': round(dd,3), 'dm_pvalue': 0.028}
"""
        results = scan_for_hardcoded_patterns(old_content)
        assert len(results) >= 4, f"Old run_final.py should flag 4+ issues, got {len(results)}"


class TestIntegrityReport:
    """Test the IntegrityReport dataclass."""

    def test_empty_report_passes(self):
        report = IntegrityReport()
        assert report.passed
        assert report.n_failed == 0

    def test_summary_format(self):
        metrics = {"precision": 0.6137, "recall": 0.5723}
        report = validate_metrics_integrity(metrics, context="test")
        summary = report.summary()
        assert "PASSED" in summary or "FAILED" in summary
