import sys
import numpy as np
from pylab import *
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
matplotlib.rcParams.update({'font.size': 13})

folder = '/home/eli/Documentos/Astronomia/posdoc/halo-elongation/redMapper/member_distribution/profiles/'
out = np.loadtxt(folder+'table_mcmc_700_5000.out',dtype='str').T

folder2 = '/home/eli/Documentos/Astronomia/proyectos/HALO-SHAPE/BCG_orientation/profiles/'
out_bcg = np.loadtxt(folder2+'table_bcg.out',dtype='str').T

samples = out[0]
M200    = out[1].astype(float)
zmean   = out[2].astype(float)
rin     = out[3].astype(float)
rout    = out[4].astype(float)

# '''
et      = out[5].astype(float)
e_et    = np.array([out[6].astype(float),out[7].astype(float)])
chi_t   = out[8].astype(float)
ex      = out[9].astype(float)
e_ex    = np.array([out[10].astype(float),out[11].astype(float)])
chi_x   = out[12].astype(float)
eb      = out[13].astype(float)
e_eb    = np.array([out[14].astype(float),out[15].astype(float)])
chi_b   = out[16].astype(float)
eraw     = out[17].astype(float)
e_eraw   = out[18].astype(float)
# '''

samples = np.append(out_bcg[0],samples)
rin     = np.append(out_bcg[3].astype(float),rin)
rout    = np.append(out_bcg[4].astype(float),rout)
eb      = np.append(out_bcg[5].astype(float),eb)
e_eb2    = np.array([out_bcg[6].astype(float),out_bcg[7].astype(float)])

e_eb = np.concatenate((e_eb,e_eb2),axis=1)

mtotal   = []
mmbin1   = []
mmbin2   = []
mmz1     = []
mmz2     = []
mt       = []
mtwl     = []
mtwd     = []
mtp      = []
mtpwl    = []
mtpwd    = []
mcontrol = []
mbcg     = []
distributions = []

angles = ['bcg',u'$\phi_1$',u'$\phi_L$',u'$\phi_d$',
          r'$\phi^*_1$',r'$\phi^*_L$',r'$\phi^*_d$',
          'control']

ang_ind = np.arange(len(angles))

for i in range(len(samples)):
    j = samples[i]
    mtotal += ['total' in j]
    mmbin1 += ['bin1' in j]
    mmbin2 += ['bin2' in j]
    mmz1   += ['z1' in j]
    mmz2   += ['z2' in j]
    mt       +=  ['t.cat' in j]     
    mtwl     +=  ['twl.cat' in j]
    mtwd     +=  ['twd.cat' in j]    
    mtp      +=  ['tp.cat' in j]
    mtpwl    +=  ['tpwl.cat' in j]
    mtpwd    +=  ['tpwd.cat' in j]
    mcontrol +=  ['control.cat' in j]
    mbcg      +=  ['bcg' in j]
    
    if mbcg[-1]:
        distributions.append(np.loadtxt(folder2+'quadrupole_bcg_both_'+j[4:]+'.out')[1000:])
    else:
        distributions.append(np.loadtxt(folder+'quadrupole_both_'+j[:-4]+'_'+str(int(rin[i]))+'_'+str(int(rout[i]))+'.out')[500:])
	
distributions = np.array(distributions)
mmz1 = np.array(mmz1)
mmz2 = np.array(mmz2)
mmbin1 = np.array(mmbin1)
mmbin2 = np.array(mmbin2)


mt    = np.array(mt)   
mtwl  = np.array(mtwl) 
mtwd  = np.array(mtwd) 
mtp   = np.array(mtp)  
mtpwl = np.array(mtpwl)
mtpwd = np.array(mtpwd)
# mint  = (rout == 1)
# mext  = (rin == 1)
# mrtot = (rin == 0)*(rout == 5)

mint  = (rout == 700)
mext  = (rin == 700)
mrtot = (rin == 0)*(rout == 5000)


##################################
# COMPARE BINS
##################################
f, axtot = plt.subplots(2,3, figsize=(11,7), sharex=True, sharey=True)
f.subplots_adjust(hspace=0,wspace=0)
ax = axtot[0,0]
bp = ax.violinplot((distributions[mmbin1*mrtot]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C4')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmbin2*mrtot]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C1')
     pc.set_edgecolor('k')  
bp = ax.violinplot((distributions[mtotal*mrtot]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C3')
     pc.set_edgecolor('k')     
plt.rc('font', family='serif', size='12.0') 
ax.plot(ang_ind+1,eb[mtotal*mrtot],'C3o',label='Total sample')
ax.plot(ang_ind+1,eb[mmbin1*mrtot],'C4v',label=u'$\lambda < 27.982$')
ax.plot(0.0,0.02,'w.',label='$r_{0.1}^{5.0}$')
ax.plot(ang_ind+1,eb[mmbin2*mrtot],'C1^',label=u'$\lambda \geq 27.982$')
ax.errorbar(ang_ind+1,eb[mtotal*mrtot],yerr=e_eb[:,mtotal*mrtot],c = 'C3',ecolor='C3',lw=2,fmt = 'none')      
ax.errorbar(ang_ind+1,eb[mmbin1*mrtot],yerr=e_eb[:,mmbin1*mrtot],c = 'C4',ecolor='C4',lw=2,fmt = 'none')       
ax.errorbar(ang_ind+1,eb[mmbin2*mrtot],yerr=e_eb[:,mmbin2*mrtot],c = 'C1',ecolor='C1',lw=2,fmt = 'none')       
ax.legend(fontsize=12,ncol=2,columnspacing=0.2,frameon=False)

ax = axtot[0,1]
bp = ax.violinplot((distributions[mtotal*mint]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C3')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmbin1*mint]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C4')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmbin2*mint]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C1')
     pc.set_edgecolor('k')  
plt.rc('font', family='serif', size='14.0') 
ax.plot(0.0,0.02,'w.',label='$r_{0.1}^{0.7}$')
ax.plot(ang_ind+1,eb[mtotal*mint],'C3o')
ax.errorbar(ang_ind+1,eb[mtotal*mint],yerr=e_eb[:,mtotal*mint],c = 'C3',ecolor='C3',lw=2,ls='--',fmt = 'none')       
ax.plot(ang_ind+1,eb[mmbin1*mint],'C4v')
ax.errorbar(ang_ind+1,eb[mmbin1*mint],yerr=e_eb[:,mmbin1*mint],c = 'C4',ecolor='C4',lw=2,ls='--',fmt = 'none')       
ax.plot(ang_ind+1,eb[mmbin2*mint],'C1^')
ax.errorbar(ang_ind+1,eb[mmbin2*mint],yerr=e_eb[:,mmbin2*mint],c = 'C1',ecolor='C1',lw=2,ls='--',fmt = 'none')       
ax.legend(fontsize=12,ncol=2,columnspacing=0.2,frameon=False)

ax = axtot[0,2]
bp = ax.violinplot((distributions[mtotal*mext]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C3')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmbin1*mext]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C4')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmbin2*mext]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C1')
     pc.set_edgecolor('k')  
plt.rc('font', family='serif', size='13.0') 
ax.plot(0.0,0.02,'w.',label='$r_{0.7}^{5.0}$')
ax.plot(ang_ind+1,eb[mtotal*mext],'C3o')
ax.errorbar(ang_ind+1,eb[mtotal*mext],yerr=e_eb[:,mtotal*mext],c = 'C3',ecolor='C3',lw=2,ls='-.',fmt = 'none')       
ax.plot(ang_ind+1,eb[mmbin1*mext],'C4v')
ax.errorbar(ang_ind+1,eb[mmbin1*mext],yerr=e_eb[:,mmbin1*mext],c = 'C4',ecolor='C4',lw=2,ls='-.',fmt = 'none')       
ax.plot(ang_ind+1,eb[mmbin2*mext],'C1^')
ax.errorbar(ang_ind+1,eb[mmbin2*mext],yerr=e_eb[:,mmbin2*mext],c = 'C1',ecolor='C1',lw=2,ls='-.',fmt = 'none')       
ax.legend(fontsize=12,ncol=2,columnspacing=0.2,frameon=False)


ax = axtot[1,0]
bp = ax.violinplot((distributions[mtotal*mrtot]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C3')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmz1*mrtot]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C0')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmz2*mrtot]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C2')
     pc.set_edgecolor('k')  
plt.rc('font', family='serif', size='13.0') 
ax.plot(ang_ind+1,eb[mtotal*mrtot],'C3o',label='Total sample')
ax.errorbar(ang_ind+1,eb[mtotal*mrtot],yerr=e_eb[:,mtotal*mrtot],c = 'C3',ecolor='C3',lw=2,fmt = 'none')       
ax.plot(ang_ind+1,eb[mmz1*mrtot],'C0v',label=u'$z_c < 0.313$')
ax.errorbar(ang_ind+1,eb[mmz1*mrtot],yerr=e_eb[:,mmz1*mrtot],c = 'C0',ecolor='C0',lw=2,fmt = 'none')       
ax.plot(ang_ind+1,eb[mmz2*mrtot],'C2^',label=u'$z_c \geq 0.313$')
ax.errorbar(ang_ind+1,eb[mmz2*mrtot],yerr=e_eb[:,mmz2*mrtot],c = 'C2',ecolor='C2',lw=2,fmt = 'none')       
ax.legend(fontsize=12,ncol=2,columnspacing=0.2,frameon=False)

ax = axtot[1,1]
bp = ax.violinplot((distributions[mtotal*mint]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C3')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmz1*mint]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C0')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmz2*mint]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C2')
     pc.set_edgecolor('k')  
plt.rc('font', family='serif', size='13.0') 
ax.plot(ang_ind+1,eb[mtotal*mint],'C3o')
ax.errorbar(ang_ind+1,eb[mtotal*mint],yerr=e_eb[:,mtotal*mint],c = 'C3',ecolor='C3',lw=2,ls='--',fmt = 'none')       
ax.plot(ang_ind+1,eb[mmz1*mint],'C0v')
ax.errorbar(ang_ind+1,eb[mmz1*mint],yerr=e_eb[:,mmz1*mint],c = 'C0',ecolor='C0',lw=2,ls='--',fmt = 'none')       
ax.plot(ang_ind+1,eb[mmz2*mint],'C2^')
ax.errorbar(ang_ind+1,eb[mmz2*mint],yerr=e_eb[:,mmz2*mint],c = 'C2',ecolor='C2',lw=2,ls='--',fmt = 'none')       

ax = axtot[1,2]
bp = ax.violinplot((distributions[mtotal*mext]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C3')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmz1*mext]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C0')
     pc.set_edgecolor('k')
bp = ax.violinplot((distributions[mmz2*mext]).T,showextrema=False)
for pc in bp['bodies']:
     pc.set_facecolor('C2')
     pc.set_edgecolor('k')  
plt.rc('font', family='serif', size='13.0') 
ax.plot(ang_ind+1,eb[mtotal*mext],'C3o',label='Total sample')
ax.errorbar(ang_ind+1,eb[mtotal*mext],yerr=e_eb[:,mtotal*mext],c = 'C3',ecolor='C3',lw=2,ls='-.',fmt = 'none')       
ax.plot(ang_ind+1,eb[mmz1*mext],'C0v',label=u'$\lambda < 27.982$')
ax.errorbar(ang_ind+1,eb[mmz1*mext],yerr=e_eb[:,mmz1*mext],c = 'C0',ecolor='C0',lw=2,ls='-.',fmt = 'none')      
ax.plot(ang_ind+1,eb[mmz2*mext],'C2^',label=u'$\lambda \geq 27.982$')
ax.errorbar(ang_ind+1,eb[mmz2*mext],yerr=e_eb[:,mmz2*mext],c = 'C2',ecolor='C2',lw=2,ls='-.',fmt = 'none')       

ax.axis([0.0,9,-0.04,0.6])
plt.xticks(ang_ind+1,angles)
axtot[0,0].set_ylabel(r'$\epsilon$',fontsize=18)
axtot[1,0].set_ylabel(r'$\epsilon$',fontsize=18)
# ax.legend()
plt.savefig(folder2+'ellipticities_medians.pdf',bbox_inches='tight')

# COMPARE ELLIPTICITIES


f, ax = plt.subplots(1,3, figsize=(12,4), sharex=True, sharey=True)
f.subplots_adjust(hspace=0,wspace=0)
pcut = mtp+mtpwl+mtpwd
nop = mt+mtwl+mtwd

ax[0].errorbar(eb[mint*pcut*mtotal],eb[mint*nop*mtotal],yerr=e_eb[:,mint*nop*mtotal],xerr=e_eb[:,mint*pcut*mtotal],fmt='none',label = 'Total sample',ecolor='C3',alpha=0.6)
ax[0].errorbar(eb[mint*pcut*mmbin1],eb[mint*nop*mmbin1],yerr=e_eb[:,mint*nop*mmbin1],xerr=e_eb[:,mint*pcut*mmbin1],fmt='none',label = u'$\lambda < 27.982$',ecolor='C4',alpha=0.6)
ax[0].errorbar(eb[mint*pcut*mmbin2],eb[mint*nop*mmbin2],yerr=e_eb[:,mint*nop*mmbin2],xerr=e_eb[:,mint*pcut*mmbin2],fmt='none',label = u'$\lambda \geq 27.982$',ecolor='C1',alpha=0.6)
ax[0].errorbar(eb[mint*pcut*mmz1],eb[mint*nop*mmz1],yerr=e_eb[:,mint*nop*mmz1],xerr=e_eb[:,mint*pcut*mmz1],fmt='none',label = u'$z < 0.313$',ecolor='C0',alpha=0.6)
ax[0].errorbar(eb[mint*pcut*mmz2],eb[mint*nop*mmz2],yerr=e_eb[:,mint*nop*mmz2],xerr=e_eb[:,mint*pcut*mmz2],fmt='none',label = u'$z \geq 0.313$',ecolor='C2',alpha=0.6)
ax[0].plot(eb[mint*pcut*mtpwd],eb[mint*nop*mtwd],'ko')
ax[0].plot(eb[mint*pcut*mtpwl],eb[mint*nop*mtwl],'C7o')
ax[0].plot(eb[mint*pcut*mtp],eb[mint*nop*mt],'kx')
ax[0].plot([0.05,0.4],[0.05,0.4],'C7--')
plt.rc('font', family='serif', size='12.0') 
ax[0].legend(fontsize=11,frameon=False,fancybox=False)

ax[1].errorbar(eb[mext*pcut*mtotal],eb[mext*nop*mtotal],yerr=e_eb[:,mext*nop*mtotal],xerr=e_eb[:,mext*pcut*mtotal],fmt='none',ecolor='C3',alpha=0.6)
ax[1].errorbar(eb[mext*pcut*mmbin1],eb[mext*nop*mmbin1],yerr=e_eb[:,mext*nop*mmbin1],xerr=e_eb[:,mext*pcut*mmbin1],fmt='none',ecolor='C4',alpha=0.6)
ax[1].errorbar(eb[mext*pcut*mmbin2],eb[mext*nop*mmbin2],yerr=e_eb[:,mext*nop*mmbin2],xerr=e_eb[:,mext*pcut*mmbin2],fmt='none',ecolor='C1',alpha=0.6)
ax[1].errorbar(eb[mext*pcut*mmz1],eb[mext*nop*mmz1],yerr=e_eb[:,mext*nop*mmz1],xerr=e_eb[:,mext*pcut*mmz1],fmt='none',ecolor='C0',alpha=0.6)
ax[1].errorbar(eb[mext*pcut*mmz2],eb[mext*nop*mmz2],yerr=e_eb[:,mext*nop*mmz2],xerr=e_eb[:,mext*pcut*mmz2],fmt='none',ecolor='C2',alpha=0.6)
ax[1].plot(eb[mext*pcut*mtpwd],eb[mext*nop*mtwd],'ko',label = '$d$')
ax[1].plot(eb[mext*pcut*mtpwl],eb[mext*nop*mtwl],'C7o',label = '$L$')
ax[1].plot(eb[mext*pcut*mtp],eb[mext*nop*mt],'kx',label = 'uniform')
ax[1].plot([0.05,0.4],[0.05,0.4],'C7--')
ax[1].legend(loc=4,frameon=False)

ax[2].errorbar(eb[mint*pcut*mtotal],eb[mext*nop*mtotal],yerr=e_eb[:,mext*nop*mtotal],xerr=e_eb[:,mint*pcut*mtotal],fmt='none',label = 'Total',ecolor='C3',alpha=0.6)
ax[2].errorbar(eb[mint*pcut*mmbin1],eb[mext*nop*mmbin1],yerr=e_eb[:,mext*nop*mmbin1],xerr=e_eb[:,mint*pcut*mmbin1],fmt='none',label = 'low mass',ecolor='C4',alpha=0.6)
ax[2].errorbar(eb[mint*pcut*mmbin2],eb[mext*nop*mmbin2],yerr=e_eb[:,mext*nop*mmbin2],xerr=e_eb[:,mint*pcut*mmbin2],fmt='none',label = 'high mass',ecolor='C1',alpha=0.6)
ax[2].errorbar(eb[mint*pcut*mmz1],eb[mext*nop*mmz1],yerr=e_eb[:,mext*nop*mmz1],xerr=e_eb[:,mint*pcut*mmz1],fmt='none',label = 'low z',ecolor='C0',alpha=0.6)
ax[2].errorbar(eb[mint*pcut*mmz2],eb[mext*nop*mmz2],yerr=e_eb[:,mext*nop*mmz2],xerr=e_eb[:,mint*pcut*mmz2],fmt='none',label = 'high z',ecolor='C2',alpha=0.6)
ax[2].plot(eb[mint*pcut*mtpwd],eb[mext*nop*mtwd],'ko',label = '$d$')
ax[2].plot(eb[mint*pcut*mtpwl],eb[mext*nop*mtwl],'C7o',label = '$L$')
ax[2].plot(eb[mint*pcut*mtp],eb[mext*nop*mt],'kx',label = 'uniform')
ax[2].plot([0.05,0.4],[0.05,0.4],'C7--')

ax[0].set_ylabel('$\epsilon$',fontsize=18)
ax[0].set_xlabel('$\epsilon^*$',fontsize=18)
ax[0].yaxis.set_ticks([0.1,0.2,0.3,0.4])
ax[0].set_yticklabels([0.1,0.2,0.3,0.4])


ax[1].set_xlabel('$\epsilon^*$',fontsize=18)
ax[2].set_xlabel('$\epsilon^*$',fontsize=18)
# plt.savefig(folder+'plot_results_centred_700_5000/ellipticities_rrange.pdf',bbox_inches='tight')


