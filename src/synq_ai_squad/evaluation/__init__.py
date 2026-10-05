"""The evaluation platform: dataset loading, scoring, statistics, quality gate, reports and judge calibration.

Everything in here except judge.py is plain Python with no AI calls, so it is unit-tested in CI.
The command-line runner that ties it together is synq_ai_squad/evals.py.
"""
