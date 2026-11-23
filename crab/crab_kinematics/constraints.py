"""Symbolic task-space-region (TSR) error: plain `(robot, q, *extra_inputs) -> sf.Matrix` math, like
crab_kinematics.derived (codegen lives in crab_codegen.constraint_header).

For each end-effector frame the error is the se(3) displacement of the offset frame e (rTe, expressed in the
end-effector frame) in the reference frame r (wTr, expressed in the world frame):

    r_T_e = wTr^-1 * wT_eef(q) * rTe            error = [ translation(r_T_e) ; log(rotation(r_T_e)) ]

six rows per end-effector, [x y z rx ry rz]. The rows are compared with the region's bounds [lb, ub] outside
this file (the hinge), so the generated kernel does not depend on them. Transforms are flat
[qw qx qy qz x y z] (wxyz quaternion first, as McVAMP's TSR constraint). The rotation log uses SymForce's
epsilon-regularised map, so the Jacobian stays finite at exactly-satisfied orientations.
"""

import symforce.symbolic as sf

from crab_model.robot import Robot
from crab_kinematics.fk import q_to_eeposes


def _pose_wxyz(p, k: int) -> sf.Pose3:
    """Pose3 of the flat [qw qx qy qz x y z] stored at p[7k : 7k + 7]."""
    o = 7 * k
    rot = sf.Rot3(sf.Quaternion(xyz=sf.V3(p[o + 1], p[o + 2], p[o + 3]), w=p[o]))
    return sf.Pose3(R=rot, t=sf.V3(p[o + 4], p[o + 5], p[o + 6]))


def tsr_error(robot: Robot, q, r_te, w_tr, eef_names=None) -> sf.Matrix:
    """(q, rTe, wTr) -> raw error, 6 rows per end-effector (robot.end_effectors, or eef_names). r_te and w_tr are
    flat 7 * n_eef vectors, one transform per end-effector in the same order."""
    names = list(eef_names or robot.end_effectors)
    rows = []
    for k, eef in enumerate(q_to_eeposes(robot, q, names)):
        rel = _pose_wxyz(w_tr, k).inverse() * eef * _pose_wxyz(r_te, k)
        rows += rel.t.to_flat_list() + list(rel.R.to_tangent(epsilon=sf.epsilon()))     # to_tangent returns a list
    return sf.Matrix(rows)


def tsr_error_and_jacobian(robot: Robot, q, r_te, w_tr, eef_names=None) -> sf.Matrix:
    """[error (6 n_eef); d(error)/dq (6 n_eef x nq, row-major, columns in CRAB q order)] as one column, so the
    FK terms are shared. The Jacobian is of the raw error, as the planner hinges the error but not the Jacobian."""
    err = tsr_error(robot, q, r_te, w_tr, eef_names)
    return sf.Matrix(err.to_flat_list() + err.jacobian(sf.Matrix(q)).to_flat_list())
