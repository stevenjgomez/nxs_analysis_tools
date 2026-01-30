import os
import pooch

GOODBOY = pooch.create(
    path=pooch.os_cache("nxs_analysis_tools/cubic"),
    base_url="https://raw.githubusercontent.com/stevenjgomez/dataset-cubic/main/data/",
    registry={
        "cubic_15.nxs": None,
        "15/transform.nxs": None,
        "cubic_25.nxs": None,
        "25/transform.nxs": None,
        "cubic_35.nxs": None,
        "35/transform.nxs": None,
        "cubic_45.nxs": None,
        "45/transform.nxs": None,
        "cubic_55.nxs": None,
        "55/transform.nxs": None,
        "cubic_65.nxs": None,
        "65/transform.nxs": None,
        "cubic_75.nxs": None,
        "75/transform.nxs": None,
        "cubic_80.nxs": None,
        "80/transform.nxs": None,
        "cubic_104.nxs": None,
        "104/transform.nxs": None,
        "cubic_128.nxs": None,
        "128/transform.nxs": None,
        "cubic_153.nxs": None,
        "153/transform.nxs": None,
        "cubic_177.nxs": None,
        "177/transform.nxs": None,
        "cubic_202.nxs": None,
        "202/transform.nxs": None,
        "cubic_226.nxs": None,
        "226/transform.nxs": None,
        "cubic_251.nxs": None,
        "251/transform.nxs": None,
        "cubic_275.nxs": None,
        "275/transform.nxs": None,
        "cubic_300.nxs": None,
        "300/transform.nxs": None,
    }
)

def fetch_cubic(temperatures=None):
    """
    Load the cubic dataset.
    """
    fnames = []
    temperatures = [15, 25, 35, 45, 55, 65, 75, 80, 104, 128,
                    153, 177, 202, 226, 251, 275, 300] if temperatures is None else temperatures
    for T in temperatures:
        fnames.append(GOODBOY.fetch(f"cubic_{T}.nxs"))
        fnames.append(GOODBOY.fetch(f"{T}/transform.nxs"))
    return fnames

def cubic(temperatures=None):
    fnames = fetch_cubic(temperatures)
    dirname = os.path.dirname(fnames[0])
    return dirname

POOCH = pooch.create(
    path=pooch.os_cache("nxs_analysis_tools/hexagonal"),
    base_url="https://raw.githubusercontent.com/stevenjgomez/dataset-hexagonal/main/data/",
    registry={
        "hexagonal_15.nxs": "850d666d6fb0c7bbf7f7159fed952fbd53355c3c0bfb40410874d3918a3cca49",
        "15/transform.nxs": "45c089be295e0a5b927e963540a90b41f567edb75f283811dbc6bb4a26f2fba5",
        "hexagonal_300.nxs": "c6a9ff704d1e42d9576d007a92a333f529e3ddf605e3f76a82ff15557b7d4a43",
        "300/transform.nxs": "e665ba59debe8e60c90c3181e2fb1ebbce668a3d3918a89a6bf31e3563ebf32e",
    }
)

def fetch_hexagonal(temperatures=None):
    """
    Load the hexagonal dataset.
    """
    fnames = []
    temperatures = [15, 300] if temperatures is None else temperatures
    for T in temperatures:
        fnames.append(POOCH.fetch(f"hexagonal_{T}.nxs"))
        fnames.append(POOCH.fetch(f"{T}/transform.nxs"))
    return fnames

def hexagonal(temperatures=None):
    fnames = fetch_hexagonal(temperatures)
    dirname = os.path.dirname(fnames[0])
    return dirname

LARGEBOI = pooch.create(
    path=pooch.os_cache("nxs_analysis_tools/orthorhombic"),
    base_url="https://raw.githubusercontent.com/stevenjgomez/dataset-orthorhombic/main/data/",
    registry={
        "orthorhombic_15.nxs": None,
        "15/transform.nxs": None,
        "orthorhombic_100.nxs": None,
        "100/transform.nxs": None,
        "orthorhombic_300.nxs": None,
        "300/transform.nxs": None,
    }
)

def fetch_orthorhombic(temperatures=None):
    """
    Load the orthorhombic dataset.
    """
    fnames = []
    temperatures = [15, 100, 300] if temperatures is None else temperatures
    for T in temperatures:
        fnames.append(LARGEBOI.fetch(f"orthorhombic_{T}.nxs"))
        fnames.append(LARGEBOI.fetch(f"{T}/transform.nxs"))
    return fnames

def orthorhombic(temperatures=None):
    fnames = fetch_orthorhombic(temperatures)
    dirname = os.path.dirname(fnames[0])
    return dirname

BONES = pooch.create(
    path=pooch.os_cache("nxs_analysis_tools/vacancies"),
    base_url="https://raw.githubusercontent.com/stevenjgomez/dataset-vacancies/main/",
    registry={
        "vacancies.nxs": "39eaf8df84a0dbcacbe6ce7c6017da4da578fbf68a6218ee18ade3953c26efb5",
        "fft.nxs": "c81178eda0ec843502935f29fcb2b0b878f7413e461612c731d37ea9e5e414a9",
    }
)

def vacancies():
    """
    Load the vacancies dataset.
    """
    return BONES.fetch(f"vacancies.nxs")

def vacanciesfft():
    """
    Load the vacancies fft dataset.
    """
    return BONES.fetch(f"fft.nxs")