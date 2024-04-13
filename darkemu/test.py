import numpy as np
import matplotlib.pyplot as plt

from dark_emulator import darkemu

#import plotparams
#plotparams.buba()

z = 0

emu = darkemu.base_class()
cparam = np.array([0.02225,0.1198,0.6844,3.094,0.9645,-1.])
emu.set_cosmology(cparam)

rs = np.logspace(-2,2.5,200)

plt.figure(figsize=(10,6))
for i, Mmin in enumerate(np.logspace(12,15,7)):
    xihm = emu.get_xicross_massthreshold(rs,Mmin,z)
    plt.loglog(rs,xihm,color="C{}".format(i),label='$M_\mathrm{th}=%0.2g$' %Mmin)
    plt.loglog(rs,-xihm,':',color="C{}".format(i))
plt.legend(fontsize=12)
plt.ylim(0.00001,1000000)
plt.xlabel("$x\,[h^{-1}\mathrm{Mpc}]$")
plt.ylabel("$\\xi_\mathrm{hm}(x)$")
plt.text(0, 0.5)#, s='$\xi_\mathrm{hm}(x)$')
plt.show()
