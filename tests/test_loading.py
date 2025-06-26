import pytest
from nexusformat.nexus import NXdata, NXfield, NXentry, NXroot, NXlink
from nexusformat.nexus import nxsave
import numpy as np

from nxs_analysis_tools.datareduction import load_transform
from nxs_analysis_tools.chess import TempDependence

# @pytest.fixture
# def nxrefine_nxs_file(tmp_path):
#     # Load original data
#     x = NXfield(np.linspace(0, 1, 10))
#     y = NXfield(np.linspace(0, 1, 10))
#     z = NXfield(np.linspace(0, 1, 10))
#     v = NXfield(np.random.rand(10, 10))
#     data = NXdata(v, (x, y, z))
#     raw_data = data.nxsignal.nxdata

#     # Save intermediate transformed data
#     out_data = NXdata(NXfield(raw_data.transpose(2, 1, 0), name='v'))
#     # Create subfolder '15' under the temp directory
#     transform_dir = tmp_path / '15'
#     transform_dir.mkdir()
#     transform_path = transform_dir / 'transform.nxs'
#     nxsave(str(transform_path), out_data)

#     # Construct final NXroot structure
#     main_file = NXroot()
#     main_file['entry'] = NXentry()
#     newH = NXfield(data.nxaxes[0].nxdata, name='Qh')
#     newK = NXfield(data.nxaxes[1].nxdata, name='Qk')
#     newL = NXfield(data.nxaxes[2].nxdata, name='Ql')
    
#     main_file['entry']['transform'] = NXdata(
#         NXlink(name='data', target='/entry/data/v', file=transform_path.name),
#         [newL, newK, newH]
#     )
#     main_file.entry.transform.attrs['angles'] = [90., 90., 90.]
#     main_file.entry.transform.attrs['signal'] = 'data'
#     main_file.entry.transform['title'] = '15.000K Transform'

#     final_file_path = tmp_path / 'CsV3Sb5_15.nxs'
#     nxsave(str(final_file_path), main_file)

#     # Return path for test usage
#     return final_file_path