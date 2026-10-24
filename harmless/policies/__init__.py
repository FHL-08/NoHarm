"""Concrete policy adapters filling the act(observation) contract.

Each module here pairs a controller with an embodiment's declared
action_space: `gr00t` serves NVIDIA's released Isaac GR00T N weights as a
VLA emitting joint_pos action chunks, and `serve` holds the checkpoint
acquisition and model-serving plumbing behind it.
"""
