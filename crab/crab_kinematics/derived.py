"""Symbolic quantities derived from forward kinematics -- plain
`(robot, q, *extra_inputs) -> sf.Matrix` math, no codegen/JIT-specific code. Meant to
be handed as symbolic_fn to crusty_jit.simple.build_simple_jit_function (Path A) or
pasted into a hand-fused kernel (Path B, crab_codegen.collision_kernel).
"""

import symforce.symbolic as sf

from crab_model.robot import Robot
from crab_kinematics.fk import forward_kinematics, q_to_eeposes


def forward_kinematics_spheres(robot: Robot, q) -> sf.Matrix:
    """q -> flat [x, y, z, radius] per collision sphere, in
    crab_codegen.sphere_order's canonical ordering (same ordering
    forward_kinematics already produces)."""
    primitive_xyzr, _ = forward_kinematics(robot, q)
    flat_xyzr = []
    for x, y, z, r in primitive_xyzr:
        flat_xyzr.extend([x, y, z, r])
    return sf.Matrix(flat_xyzr)


def q_to_ee_pose(robot: Robot, q) -> sf.Matrix:
    """q -> flat 7-vector Pose3.to_storage() of robot.end_effectors[0]."""
    ee_pose = q_to_eeposes(robot, q, [robot.end_effectors[0]])[0]
    return sf.Matrix(ee_pose.to_storage())


def task_space_distance(robot: Robot, q, goal_pose_flat) -> sf.Matrix:
    """(q, goal_pose) -> [translation distance] between the end effector and a goal
    pose. goal_pose_flat is a flat 7-vector, Pose3.to_storage()'s layout -- same
    convention q_to_ee_pose returns, so a goal can come directly from a previous
    q_to_ee_pose call."""
    ee_pose = q_to_eeposes(robot, q, [robot.end_effectors[0]])[0]
    goal_pose = sf.Pose3.from_storage(list(goal_pose_flat))
    rel = goal_pose.inverse() * ee_pose
    return sf.Matrix([rel.t.norm()])


def q_to_ee_position_jacobian(robot: Robot, q) -> sf.Matrix:
    """q -> flat row-major 3 x nq Jacobian d(position)/dq of robot.end_effectors[0], columns in
    q order (zero for joints not on the chain). Equals the linear rows of a world-aligned frame
    Jacobian, so it is directly comparable across libraries."""
    ee_pose = q_to_eeposes(robot, q, [robot.end_effectors[0]])[0]
    jac = sf.Matrix(ee_pose.t).jacobian(sf.Matrix(q))
    return sf.Matrix(jac.to_flat_list())


def _ee_jacobian_columns(robot: Robot, q):
    """Geometric Jacobian columns of robot.end_effectors[0], world-aligned: (linear, angular), each
    a list of nq V3. For each joint on the chain to the end effector, a revolute column is
    (world axis) x (p_ee - joint origin) linear and the world axis angular; a prismatic column is
    the world axis linear and zero angular. The axis and origin come from the joint's frame before
    its own rotation (parent pose * joint.origin), so column j doesn't depend on q_j and reuses
    FK's intermediate terms (O(joints) work). Joints off the chain give zero columns."""
    _, link_poses = forward_kinematics(robot, q)
    ee = robot.end_effectors[0]
    p = link_poses[robot.link_name_to_index[ee]].t

    linear = [sf.V3.zero() for _ in range(robot.nq)]
    angular = [sf.V3.zero() for _ in range(robot.nq)]
    link = robot.links[ee]
    while link.parent_joint is not None:
        joint = robot.joints[link.parent_joint]
        if joint.is_actuated:
            pre = link_poses[robot.link_name_to_index[joint.parent]] * joint.origin
            axis = pre.R * joint.axis
            if joint.type == "revolute":
                linear[joint.id] = axis.cross(p - pre.t)
                angular[joint.id] = axis
            elif joint.type == "prismatic":
                linear[joint.id] = axis
            else:
                raise ValueError(f"Unsupported joint type: {joint.type}")
        link = robot.links[joint.parent]
    return linear, angular


def q_to_ee_position_jacobian_analytic(robot: Robot, q) -> sf.Matrix:
    """Same quantity as q_to_ee_position_jacobian, but the geometric formula instead of symbolic
    differentiation (see _ee_jacobian_columns). Flat row-major 3 x nq."""
    linear, _ = _ee_jacobian_columns(robot, q)
    return sf.Matrix([linear[j][i] for i in range(3) for j in range(robot.nq)])


def q_to_ee_jacobian_analytic(robot: Robot, q) -> sf.Matrix:
    """Flat row-major 6 x nq geometric Jacobian, rows (vx, vy, vz, wx, wy, wz) in the world frame."""
    linear, angular = _ee_jacobian_columns(robot, q)
    return sf.Matrix([cols[j][i] for cols in (linear, angular) for i in range(3) for j in range(robot.nq)])
