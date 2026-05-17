# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DEPRECATED — DO NOT USE                                           ║
# ║                                                                    ║
# ║  This script generated synthetic VN30 data that was then used to   ║
# ║  produce metrics claimed as real-market validation results.        ║
# ║                                                                    ║
# ║  The proper pipeline uses real data via vnstock API:               ║
# ║    python -m vn_regime_transfer.run_pipeline                       ║
# ╚══════════════════════════════════════════════════════════════════════╝

raise RuntimeError(
    "generate_final_data.py is DEPRECATED. Synthetic data must not be "
    "presented as real-market validation. Use the vnstock-based pipeline "
    "via 'python -m vn_regime_transfer.run_pipeline' instead."
)
