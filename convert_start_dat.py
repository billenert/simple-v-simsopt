import numpy as np
import matplotlib.pyplot as plt
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry, vmec_splines
import booz_xform as bx
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from scipy.optimize import root
from simsopt.util.mpi import MpiPartition
from mpi4py import MPI
from simsopt._core.util import parallel_loop_bounds
from simsopt.util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY

comm = MPI.COMM_WORLD
mpi = MpiPartition(comm_world=comm)

Ekin=FUSION_ALPHA_PARTICLE_ENERGY
mass=ALPHA_PARTICLE_MASS
charge=ALPHA_PARTICLE_CHARGE # Alpha particle charge
init_v=np.sqrt(2*Ekin/mass)

'''
“1) zstart(1,ipart)=r (normalized toroidal flux s)
2) zstart(2,ipart)=theta_vmec
3) zstart(3,ipart)=varphi_vmec
4) normalized velocity module z(4) = v / v_0:
zstart(4,ipart)=1.d0  Means all particles start with 3.5 Mev energies
5) starting pitch z(5)=v_\parallel / v:”
'''
data = np.loadtxt('./initial conditions/SIMPLE initial/start.dat')
npart = len(data)
s = data[:npart, 0]
theta0 = np.mod(data[:npart, 1], 2*np.pi)
phi = np.mod(data[:npart, 2], 2*np.pi)
v0 = init_v * data[:npart, 3]
#5) starting pitch z(5)=v_\parallel / v: -> v_parallel = v * z(5)
vpar0 = v0 * data[:npart, 4]





'''Load VMEC and booz_xform data'''
vmec = Vmec('./initial conditions/SIMPLE initial/wout.nc')
b = bx.Booz_xform()
b.read_boozmn('./initial conditions/simsopt initial/boozmn_LandremanPaul2021_QA_reactorScale_lowres_reference.nc')
bri = BoozerRadialInterpolant(b,order=3,no_K=True)
vs = vmec_splines(vmec)

'''Compute vartheta from VMEC data'''
mnmax = vs.mnmax
xm = vs.xm
xn = vs.xn

ns = len(s)

lmns = np.zeros((ns, mnmax))
for jmn in range(mnmax):
    lmns[:, jmn] = vs.lmns[jmn](s)

angle = xm[:, None] * theta0[None, :] - xn[:, None] * phi[None, :]
sinangle = np.sin(angle)

lambd = np.einsum('ij,ji->i', lmns, sinangle)
vartheta = theta0 + lambd

'''Perform root solve to obtain start.dat data in Boozer coordinates'''

def vartheta_phi_vmec(s,theta_b,zeta_b):
    points = np.zeros((1,3))
    points[:,0] = s
    points[:,1] = theta_b
    points[:,2] = zeta_b
    bri.set_points(points)
    nu = bri.nu()[0,0]
    iota = bri.iota()[0,0]
    vartheta = theta_b - iota*nu
    phi = zeta_b - nu
    return vartheta, phi

def func_root(x,s,vartheta_target,phi_target):
    theta_b = x[0]
    zeta_b = x[1]
    vartheta, phi = vartheta_phi_vmec(s,theta_b,zeta_b)
    return [vartheta - vartheta_target,phi - phi_target]


'''Split data over procs'''
first, last = parallel_loop_bounds(comm, len(s))

theta_b_chunk = []
zeta_b_chunk = []
for i in range(first, last):
    sol = root(func_root, [vartheta[i],phi[i]], args=(s[i],vartheta[i],phi[i]))
    if (sol.success):
        theta_b_chunk.append(sol.x[0])
        zeta_b_chunk.append(sol.x[1])
    else:
        theta_b_out = sol.x[0]
        zeta_b_out = sol.x[1]
        vartheta_out, phi_out = vartheta_phi_vmec(s[i],theta_b_out,zeta_b_out)
        print('vartheta: ',vartheta[i])
        print('vartheta_out: ',vartheta_out)
        print('phi: ',phi[i])
        print('phi_out: ',phi_out)

data_out_theta_b = comm.gather(theta_b_chunk)
data_out_zeta_b  = comm.gather(zeta_b_chunk)

if comm.rank == 0:
    theta_b = [i for o in data_out_theta_b for i in o]
    zeta_b  = [i for o in data_out_zeta_b for i in o]
    print(theta_b)
    print(zeta_b)
    np.savetxt('./initial conditions/simsopt initial/theta0.txt',theta_b)
    np.savetxt('./initial conditions/simsopt initial/zeta0.txt',zeta_b)
    np.savetxt('./initial conditions/simsopt initial/s0.txt', s)
    np.savetxt('./initial conditions/simsopt initial/vpar0.txt', vpar0)
