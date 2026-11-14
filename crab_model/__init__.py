import symforce
# Centralized here (rather than in whichever module happens to run first) since
# crab_model.robot.Robot is a dependency of nearly every module in this codebase
# that touches symforce.symbolic (crab_kinematics, crab_codegen, ...). SymForce's
# symbolic backend is process-global, one-time, order-sensitive configuration: it has
# to run before anything does `import symforce.symbolic`, regardless of which of
# those modules happens to be imported first by a given entry point.
symforce.set_symbolic_api("symengine")
# A numeric epsilon (not set_epsilon_to_symbol): a symbolic one would become an extra
# input to every generated function and break the fixed `void** inputs` ABI.
symforce.set_epsilon_to_number()
