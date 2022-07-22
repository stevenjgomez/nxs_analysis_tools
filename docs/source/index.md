# MagentroPy

```{admonition} References
Please cite the following in any published work that makes use of this package:

> [1] J. D. Bocarsly et al.,
> [Phys. Rev. B 97, 100404(R) (2018)](https://doi.org/10.1103/PhysRevB.97.100404)
>
> [2] J. J. Stickel,
> [Comput. Chem. Eng. 34, 467 (2010)](https://doi.org/10.1016/j.compchemeng.2009.10.007)

The first version of the `magentropy` code was included as supplementary
material in [1]. The Tikhonov regularization procedure was described
in [2] and was originally implemented by Stickel in the package
[scikit.datasmooth](https://github.com/jjstickel/scikit-datasmooth).
```

## Overview

MagentroPy provides a class, {{ MagentroData }},
that can be used to calculate magnetocaloric quantities from DC magnetization
data supplied as magnetic moment vs. temperature sweeps (monotonic) taken under
several different magnetic fields. The class is set up to work out-of-the-box
with `.dat` data files produced by a Quantum Design Vibrating Sample
Magnetometer or a
[Quantum Design MPMS3 SQUID Magnetometer](https://www.qdusa.com/products/mpms3.html).
However, {class}`pandas.DataFrame`s or delimited files such as `.csv`
are also acceptable inputs.

During data processing, the magnetic moment is differentiated with respect to
temperature and integrated with respect to the magnetic field to calculate
the entropy. Smoothing is performed on the magnetization data using Tikhonov
regularization in order to reduce noise in the derivative. See [1] and [2]
for additional information.

Plotting methods are provided for creating line plots and heat maps. These
methods can be used with {mod}`matplotlib` for flexible and extensive plotting
functionality.

An experimental {{ bootstrap }} method is implemented to estimate the error
in the smoothed moment.

See the {doc}`quickstart` for installation, logging, and usage.

## Just-in-time compilation

Certain methods, such as {{ process_data }}, are accelerated
using {{ numba }}'s just-in-time ({term}`JIT`) compilation. {term}`JIT`
functions will have a noticeable compilation overhead the first time they are
run; however, the results are cached on disk, so all subsequent runs will be
faster.

If desired, {term}`JIT` compilation can be disabled globally as follows:

```python
from numba import config

config.DISABLE_JIT = True
```

```{toctree}
---
caption: Contents
---
```

## Contents

```{toctree}
---
maxdepth: 2
---
quickstart
examples/index
api
license
genindex
```

## License

This project is {doc}`licensed <license>` under the [MIT License](https://opensource.org/licenses/MIT).
