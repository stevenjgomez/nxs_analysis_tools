from nxs_analysis_tools.datareduction import load_data
from nxs_analysis_tools.pairdistribution import *
import matplotlib.pyplot as plt

data = load_data('../docs/source/examples/example_data/pairdistribution_data/test_hkli.nxs')
# s = Symmetrizer3D(data)

# s.plane1symmetrizer.set_parameters(theta_min=0, theta_max=90, mirror=True)
# s.plane1symmetrizer.test(data[:,:,len(data.L)//2])
#
# s.plane2symmetrizer.set_parameters(theta_min=45, theta_max=90, mirror=True)
# s.plane2symmetrizer.test(data[:,len(data.K)//2,:])
#
# s.plane3symmetrizer.set_parameters(theta_min=0, theta_max=90, mirror=False)
# s.plane3symmetrizer.test(data[len(data.H)//2,:,:])
#
# s.symmetrize()

p = Puncher()
p.set_data(data)
p.set_lattice_params((1,1,1,90,90,90))
bm = p.generate_bragg_mask(punch_radius=0.25)
p.add_mask(bm)
p.punch()
plot_slice(p.punched[:,:,0.0])
plt.show()

m = p.generate_mask_at_coord(coordinate=(0.33, 0.33, 0.0), punch_radius=0.25)
p.add_mask(m)
p.punch()
plot_slice(p.punched[:,:,0.0])
plt.show()

m = p.generate_mask_at_coord(coordinate=(-0.1, -0.1, 0.0), punch_radius=0.2)
p.subtract_mask(m)
p.punch()
plot_slice(p.punched[:,:,0.0])
plt.show()

# SUCCESS