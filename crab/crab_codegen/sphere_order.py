"""Canonical per-robot sphere ordering, shared by codegen and the Python-side model
builder so both agree on which flat-array index corresponds to which sphere.

crab_kinematics.fk.forward_kinematics (used by crab_kinematics.derived to build
the SymForce FK function) iterates robot.traversal_order and appends each visited
joint's child link's primitives, in that order -- it never visits the root link's own
primitives, since the root link is never anyone's "child" in that loop. Any other
ordering (e.g. iterating robot.links.values() directly) will silently misalign with
the compiled FK output's flat vector. This module is the one place that ordering is
defined, so it can't drift.
"""

from crab_model.robot import Robot


def robot_sphere_list(robot: Robot) -> list[dict]:
    spheres = []
    for joint_name in robot.traversal_order:
        joint = robot.joints[joint_name]
        link = robot.links[joint.child]
        for idx, primitive in enumerate(link.primitives):
            spheres.append({
                "link_id": link.id,
                "link_name": link.name,
                "index": idx,
                "local_x": float(primitive.pose.t[0]),
                "local_y": float(primitive.pose.t[1]),
                "local_z": float(primitive.pose.t[2]),
                "radius": float(primitive.radius),
            })
    return spheres


def _min_enclosing_sphere(centers, radii):
    """Smallest-ish sphere enclosing balls (centers (m, 3), radii (m,)): Badoiu-Clarkson iteration toward the
    farthest point of the farthest ball, then the exact radius for the final centre (always conservative)."""
    import numpy as np

    centers, radii = np.asarray(centers, float), np.asarray(radii, float)
    c = centers[np.argmax(radii)].copy()
    for k in range(1, 4000):
        d = np.linalg.norm(centers - c, axis=1)
        i = int(np.argmax(d + radii))
        if d[i] < 1e-12:
            break
        far = centers[i] + radii[i] * (centers[i] - c) / d[i]
        c = c + (far - c) / (k + 1)
    return c, float((np.linalg.norm(centers - c, axis=1) + radii).max()) + 1e-5


def link_bounding_spheres(robot: Robot) -> list[dict]:
    """One bounding sphere per run of consecutive spheres of the same link (the spheres of robot_sphere_list
    are grouped by link already): {link_name, first, count, center (link frame), radius}, in sphere order.
    The centre is fixed in the link frame, so its world position is link_pose * center (CRICKET's scheme)."""
    import numpy as np

    spheres = robot_sphere_list(robot)
    groups, start = [], 0
    for i in range(1, len(spheres) + 1):
        if i == len(spheres) or spheres[i]["link_id"] != spheres[start]["link_id"]:
            part = spheres[start:i]
            c, r = _min_enclosing_sphere([[s["local_x"], s["local_y"], s["local_z"]] for s in part],
                                         [s["radius"] for s in part])
            groups.append({"link_name": part[0]["link_name"], "first": start, "count": len(part),
                           "center": np.asarray(c), "radius": r})
            start = i
    return groups
