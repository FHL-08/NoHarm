"""MuJoCo reference backend.

Newton (harmless/newton_scene.py) is the primary physics backend; the
original MuJoCo pipeline stays runnable under this namespace for parity
checks and the interactive viewer (view.py). The Executor is the same
measured bridge the agent loop drives: a live MjModel/MjData pair on
kitchen.xml, the G1 stepped under its own actuators, the authored props
resting in the contact set, the person towed along her route until a
measured impact releases her, and contacts/forces reported through
contacts(), stats and sense() — all physics-measured, nothing scripted.
"""
from harmless.executor import EpisodeStats, Executor, ViewerClosed

__all__ = ["Executor", "EpisodeStats", "ViewerClosed"]
