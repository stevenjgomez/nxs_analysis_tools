from _typeshed import Incomplete
from nxs_analysis_tools import plot_slice as plot_slice

class Padder:
    def __init__(self, data: Incomplete | None = ...) -> None: ...
    data: Incomplete
    steps: Incomplete
    maxes: Incomplete
    def set_data(self, data) -> None: ...
    padding: Incomplete
    padded: Incomplete
    def pad(self, padding): ...
    def save(self, fout_name: Incomplete | None = ...) -> None: ...
    def unpad(self, data): ...

class Symmetrizer2D:
    def __init__(self, **kwargs) -> None: ...
    theta_min: Incomplete
    theta_max: Incomplete
    skew_angle: Incomplete
    mirror: Incomplete
    transform: Incomplete
    rotations: Incomplete
    symmetrization_mask: Incomplete
    wedges: Incomplete
    symmetrized: Incomplete
    def set_parameters(self, theta_min, theta_max, skew_angle: int = ..., mirror: bool = ...) -> None: ...
    wedge: Incomplete
    def symmetrize_2d(self, data): ...
    def test(self, data): ...

class Symmetrizer3D:
    data: Incomplete
    q1: Incomplete
    q2: Incomplete
    q3: Incomplete
    plane1symmetrizer: Incomplete
    plane2symmetrizer: Incomplete
    plane3symmetrizer: Incomplete
    plane1: Incomplete
    plane2: Incomplete
    plane3: Incomplete
    def __init__(self, data) -> None: ...
    symmetrized: Incomplete
    def symmetrize(self): ...
    def save(self, fout_name: Incomplete | None = ...) -> None: ...
