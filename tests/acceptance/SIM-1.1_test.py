"""SIM-1.1@1 newton_scene.build('kitchen.xml') loads the scene with G1
self-collisions off and room/prop/person contacts on.

The collision setup is judged by the contacts newton actually generates
for the spawn state: the G1's feet must meet the room and the props must
rest on the room, while no generated contact may pair two G1 bodies. The
person starts the episode touching nothing she can collide with, so her
contacts are probed by moving her free joint onto the robot through the
model state — with person contacts enabled that overlap produces
measured contacts; with them off it produces none."""

import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
KITCHEN_XML = REPO / "kitchen.xml"

# the authored free props of kitchen.xml (named in its grasp welds)
PROP_NAMES = ("mug", "plate", "knife", "butterknife", "vase", "apple",
              "bowl", "tomato", "bread")


def _load_model():
    from harmless import newton_scene

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        built = newton_scene.build(str(KITCHEN_XML))
    return built[-1] if isinstance(built, tuple) else built


def _collide(model, state):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.collide(state)


def _classes(model):
    """Classify every model body: 'robot' is the pelvis-rooted G1 subtree,
    'person' the person hazard, 'prop' an authored free prop, 'room'
    everything else (fixtures, furniture, world-static shapes)."""
    classes = []
    for label in model.body_label:
        name = label.rsplit("/", 1)[-1]
        if "/pelvis/" in label or name == "pelvis":
            classes.append("robot")
        elif name in ("person", "person_driver"):
            classes.append("person")
        elif name.startswith(PROP_NAMES):
            classes.append("prop")
        else:
            classes.append("room")
    return classes


def _contact_body_pairs(model, contacts):
    """Body-index pairs of the rigid contacts generated for a state."""
    n = int(np.atleast_1d(contacts.rigid_contact_count.numpy())[0])
    shape_body = model.shape_body.numpy()
    first = shape_body[contacts.rigid_contact_shape0.numpy()[:n]]
    second = shape_body[contacts.rigid_contact_shape1.numpy()[:n]]
    return first, second


def test_sim_1_1_g1_self_collision_off_room_prop_person_contacts_on():
    model = _load_model()
    import newton

    classes = _classes(model)
    for kind in ("robot", "person", "prop", "room"):
        assert kind in classes, f"kitchen scene loaded no {kind} bodies"

    def side(body):
        return "room" if body < 0 else classes[body]

    state = model.state()
    first, second = _contact_body_pairs(model, _collide(model, state))
    assert len(first), "build() produced a scene that generates no contacts at all"

    kinds = {frozenset((side(a), side(b))) for a, b in zip(first, second)}
    assert frozenset(("robot",)) not in kinds, (
        "contacts are generated between two G1 bodies — G1 self-collisions are on")
    assert frozenset(("robot", "room")) in kinds, (
        "no G1-vs-room contacts at spawn — the feet must rest on the floor")
    assert frozenset(("prop", "room")) in kinds, (
        "no prop-vs-room contacts at spawn — props must rest on counters/floor")

    # The person stands clear of everything her collision group touches,
    # so probe her contacts by moving her free joint over the robot.
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    px, py = state.body_q.numpy()[classes.index("robot")][:2]
    joint = next(
        i for i, label in enumerate(model.joint_label)
        if label.rsplit("/", 1)[-1] == "person_free")
    start = int(model.joint_q_start.numpy()[joint])
    q = state.joint_q.numpy().copy()
    q[start:start + 7] = [px, py, 0.0, 0.0, 0.0, 0.0, 1.0]
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    first, second = _contact_body_pairs(model, _collide(model, state))
    person_pairs = [
        (a, b) for a, b in zip(first, second)
        if (side(a) == "person") != (side(b) == "person")
    ]
    assert person_pairs, (
        "the person produces no contacts even when overlapping the robot — "
        "person contacts are off")
