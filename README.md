# MagentroPy

```{admonition} References
Please cite the following in any published work that makes use of this package:

> [1] J. D. Bocarsly et al.,
> [Phys. Rev. B 97, 100404(R) (2018)](https://doi.org/10.1103/PhysRevB.97.100404)
>
> [2] J. J. Stickel,
> [Comput. Chem. Eng. 34, 467 (2010)](https://dx.doi.org/10.1016/j.compchemeng.2009.10.007)

The first version of the `magentropy` code was included as supplementary
material in [1]. The Tikhonov regularization procedure was described
in [2] and was originally implemented by Stickel in the package
[scikit.datasmooth](https://github.com/jjstickel/scikit-datasmooth).
```

## Overview

MagentroPy provides a class, `MagentroData`,
that can be used to calculate magnetocaloric quantities from DC magnetization
data supplied as magnetic moment vs. temperature sweeps (monotonic) taken under
several different magnetic fields. The class is set up to work out-of-the-box
with `.dat` data files produced by a Quantum Design Vibrating Sample
Magnetometer or a
[Quantum Design MPMS3 SQUID Magnetometer](https://www.qdusa.com/products/mpms3.html).
However, `pandas.DataFrame`s or delimited files such as `.csv`
are also acceptable inputs.

View the documentation

## Installation

Install MagentroPy with ``pip``:

```{code-block} console
pip install magentropy
```

Or, with ``conda``:

```{code-block} console
conda install -c conda-forge magentropy
```

## License

This project is licensed under the MIT License.
