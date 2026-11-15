import yourdfpy
from crab_model.link import Link, Sphere
from crab_model.joint import Joint
from crab_model.robot import Robot
import symforce.symbolic as sf


from typing import List

def _pose3(origin) -> sf.Pose3:
    """Pose3 of a 4x4 `<origin>` matrix; yourdfpy gives None when the tag is absent (identity)."""
    if origin is None:
        return sf.Pose3()
    return sf.Pose3(R=sf.Rot3.from_rotation_matrix(origin[:3, :3]), t=sf.V3(origin[:3, 3]))


def load_urdf(path: str) -> yourdfpy.URDF:
    """Load a URDF file and return a yourdfpy.URDF object."""
    return yourdfpy.URDF.load(path)

def urdf_to_robot(urdf: yourdfpy.URDF, end_effectors: List[str] = None) -> Robot:
    """Convert a yourdfpy.URDF object to a Robot object."""

    robot = Robot(name="robot")

    for i, (link_name, link) in enumerate(urdf.link_map.items()):
        collision_spheres = []
        for collision in link.collisions:
            if collision.geometry.sphere:
                collision_spheres.append(Sphere(
                    pose = _pose3(collision.origin),
                    radius = collision.geometry.sphere.radius,
                ))
        robot.links[link.name] = Link(
            name=link.name,
            id=i,
            primitives=collision_spheres,
        )


        # (TODO) -- extract out the collision spheres from the urdf here and add them to the link
        # robot.links[link.name] = Link(name=link.name, id=i)

    # Joint.id is the index into q (actuated joints, in URDF order); fixed joints get -1.
    # It must not count fixed joints, or q[joint.id] in fk misindexes any robot where a
    # fixed joint precedes an actuated one.
    q_index = 0
    for joint_name, joint in urdf.joint_map.items():
        parent_name = robot.links[joint.parent].name
        child_name = robot.links[joint.child].name

        if joint.type != "fixed":
            joint_id = q_index
            q_index += 1
        else:
            joint_id = -1

        robot.joints[joint.name] = Joint(
            name=joint.name,
            id=joint_id,
            parent=parent_name,
            child=child_name,
            origin=_pose3(joint.origin),
            axis=sf.V3(joint.axis if joint.axis is not None else [1.0, 0.0, 0.0]),
            type=joint.type,
            limits=(joint.limit.lower, joint.limit.upper) if joint.limit else None,
            velocity_limits=(joint.limit.velocity, -joint.limit.velocity) if joint.limit else None,
            is_actuated=joint.type != "fixed",
        )

        # update the parent and child links with the joint information
        robot.links[joint.parent].child_joints.append(joint.name)
        robot.links[joint.child].parent_joint = joint.name

    if end_effectors is not None:
        robot.end_effectors = end_effectors
    else:
        # pick the last link in the traversal order as the end effector
        robot.end_effectors = [list(robot.links.keys())[-1]]

    # needed to trigger the post init function to compute the traversal order
    # this is stupid i know yeah but if you have a better way to do be my guest
    new_robot = Robot(
        name=robot.name,
        links=robot.links,
        joints=robot.joints,
        allowed_collision_pairs=robot.allowed_collision_pairs,
        end_effectors=robot.end_effectors,
    )

    return new_robot