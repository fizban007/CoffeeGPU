import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import numpy as np
import h5py
import sys
import os
import toml
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
matplotlib.rc('text',usetex=True)

datadir = 'Data/'
plotdir = 'plot/'
if len(sys.argv) > 1:
    datadir = os.path.join(sys.argv[1],datadir)
    plotdir = os.path.join(sys.argv[1],plotdir)

import os
if not os.path.exists(plotdir):
    os.makedirs(plotdir)

conf = toml.load(os.path.join(datadir, 'config.toml'))

omega0 = conf['omega']
c = 1.0
dn1 = conf['data_interval']
dt = conf['dt']
b0 = conf['b0']
#T=2*np.pi/omega0
lw=0.5

def plotslice(n):
    s=os.path.join(datadir, 'fld.%05d.h5' % n)
    #f = h5py.File(s,'r',swmr=True)
    f = h5py.File(s,'r')
    Bx = f["Bx"][()]
    By = f["By"][()]
    Bz = f["Bz"][()]
    Ex = f["Ex"][()]
    Ey = f["Ey"][()]
    Ez = f["Ez"][()]
    f.close()
    
    print(Bx.shape)
    ny,nx = Bx.shape
    xmin=conf['lower'][0]
    ymin=conf['lower'][1]
    zmin=conf['lower'][2]
    sizex=conf['size'][0]
    sizey=conf['size'][1]
    sizez=conf['size'][2]
    xmax=xmin+sizex
    ymax=ymin+sizey
    dx=sizex/nx
    dy=sizey/ny
    x0=np.arange(xmin,xmax,dx)
    y0=np.arange(ymin,ymax,dy)
    #z=np.arange(zmin,zmax,dz)
    x,th=np.meshgrid(x0,y0)
    r=np.exp(x)
    
    R=r*np.sin(th)
    dphi=Bx*r**3*np.sin(th)
    Br=r*Bx
    Bth=r*By
    Bph=R*Bz
    Er=r*Ex
    Eth=r*Ey
    Eph=R*Ez
    Aph=np.cumsum(dphi, axis=0)
    maxAph=np.amax(Aph)
    print(maxAph)

    B2=Br**2+Bth**2+Bph**2
    B=np.sqrt(B2)
    
    xcoord,ycoord=r*np.sin(th),r*np.cos(th)
    Rmax=50.0
    #Rmax=25.0
    #Rmax=np.amax(R)
    index=np.argmin(np.fabs(np.exp(x0)-1.5*Rmax))
    print(index)

    fig=plt.figure(figsize=(11,10))
    step=2
    ax=fig.add_subplot(121)
    vlim=0.05
    #vlim=0.1
    #vlim=0.01
    #im=plt.pcolormesh(xcoord[:,0:index],ycoord[:,0:index],(Bph/B)[:,0:index],cmap='jet',vmin=-vlim,vmax=vlim)
    im=plt.pcolormesh(xcoord[::step,0:index:step],ycoord[::step,0:index:step],(Bph*r/b0)[::step,0:index:step],cmap='bwr',vmin=-vlim,vmax=vlim)
    #im=plt.pcolor(xcoord[:,0:index],ycoord[:,0:index],(Bph/B)[:,0:index],cmap='jet',vmin=-vlim,vmax=vlim)
    levels=np.linspace(0,0.2*maxAph,10)
    plt.contour(xcoord[:,0:index],ycoord[:,0:index],Aph[:,0:index],colors='k',levels=levels,linewidths=lw)
    ax.set_xlim([0,Rmax])
    ax.set_ylim([-Rmax,Rmax])
    ax.set_aspect('equal')
    plt.xlabel("R")
    plt.ylabel("z")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(r'$B_{\phi}r$')
    ax=fig.add_subplot(122)
    vlim=0.05
    #vlim=0.1
    #vlim=1e-4
    #im=plt.pcolormesh(xcoord[:,0:index],ycoord[:,0:index],(Eph/B)[:,0:index],cmap='jet',vmin=-vlim,vmax=vlim)
    im=plt.pcolormesh(xcoord[::step,0:index:step],ycoord[::step,0:index:step],(Eph*r/b0)[::step,0:index:step],cmap='bwr',vmin=-vlim,vmax=vlim)
    plt.contour(xcoord[:,0:index],ycoord[:,0:index],Aph[:,0:index],levels=levels,colors='k',linewidths=lw)
    ax.set_xlim([0,Rmax])
    ax.set_ylim([-Rmax,Rmax])
    ax.set_aspect('equal')
    plt.xlabel("R")
    plt.ylabel("z")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.title(r'$E_{\phi}r$')
    #t=n*dn1*dt/T
    t=n*dn1*dt
    plt.suptitle("t/T=%.3f" % t)
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir,'nEBplot%03d.png' % (n)))
    #plt.savefig(os.path.join(plotdir,'smallEBplot%03d.png' % (n)))
    #plt.savefig(os.path.join(plotdir,'largeEBplot%03d.png' % (n)))
    plt.close()
    #plt.show()

max_num = conf['max_steps'] // conf['data_interval'] + 1
#for n in range(0, max_num,1):
for n in range(0, 201,1):
    if os.path.exists(os.path.join(datadir, 'fld.%05d.h5' % n)) and not os.path.exists(os.path.join(plotdir,'nEBplot%03d.png' % (n))):
#    if os.path.exists(os.path.join(datadir, 'fld.%05d.h5' % n)) and not os.path.exists(os.path.join(plotdir,'largeEBplot%03d.png' % (n))):
        print(n)
        plotslice(n)
