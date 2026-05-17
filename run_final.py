# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DEPRECATED — DO NOT USE                                           ║
# ║                                                                    ║
# ║  This script contained hard-coded fallback metrics (lines 59-63)   ║
# ║  that fabricated precision=0.61, sharpe=1.42, dm_pvalue=0.028     ║
# ║  when the model failed to produce signals.                         ║
# ║                                                                    ║
# ║  Use the proper pipeline instead:                                  ║
# ║    python -m vn_regime_transfer.run_pipeline                       ║
# ║                                                                    ║
# ║  See: implementation_plan.md, Fix #1 for details.                 ║
# ╚══════════════════════════════════════════════════════════════════════╝

raise RuntimeError(
    "run_final.py is DEPRECATED due to research integrity violations. "
    "It contained hard-coded fallback metrics and a fabricated DM p-value. "
    "Use 'python -m vn_regime_transfer.run_pipeline' instead."
)
