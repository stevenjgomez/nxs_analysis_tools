from _typeshed import Incomplete

def load_data(path): ...
def plot_slice(data, X: Incomplete | None = ..., Y: Incomplete | None = ..., transpose: bool = ..., vmin: Incomplete | None = ..., vmax: Incomplete | None = ..., skew_angle: int = ..., ax: Incomplete | None = ..., xlim: Incomplete | None = ..., ylim: Incomplete | None = ..., xticks: Incomplete | None = ..., yticks: Incomplete | None = ..., cbar: bool = ..., logscale: bool = ..., symlogscale: bool = ..., cmap: str = ..., linthresh: int = ..., title: Incomplete | None = ..., mdheading: Incomplete | None = ..., cbartitle: Incomplete | None = ..., **kwargs): ...

class Scissors:
    data: Incomplete
    center: Incomplete
    window: Incomplete
    axis: Incomplete
    integration_volume: Incomplete
    integrated_axes: Incomplete
    linecut: Incomplete
    integration_window: Incomplete
    def __init__(self, data: Incomplete | None = ..., center: Incomplete | None = ..., window: Incomplete | None = ..., axis: Incomplete | None = ...) -> None: ...
    def set_data(self, data) -> None: ...
    def get_data(self): ...
    def set_center(self, center) -> None: ...
    def set_window(self, window) -> None: ...
    def get_window(self): ...
    def cut_data(self, center: Incomplete | None = ..., window: Incomplete | None = ..., axis: Incomplete | None = ...): ...
    def highlight_integration_window(self, data: Incomplete | None = ..., label: Incomplete | None = ..., highlight_color: str = ..., **kwargs): ...
    def plot_integration_window(self, **kwargs): ...
