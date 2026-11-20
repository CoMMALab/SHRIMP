from dataclasses import dataclass, field
import symforce.symbolic as sf


@dataclass(slots=True)
class Sphere:
    pose: sf.Pose3
    radius: float


@dataclass(slots=True)
class Link:
    name: str
    id: int

    parent_joint: str | None = None
    child_joints: list[str] = field(default_factory=list)

    primitives: list[Sphere] = field(default_factory=list)
