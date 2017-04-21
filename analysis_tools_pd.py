import numpy as N
import pandas as pd
import scipy.interpolate as I
import scipy.special as S
import scipy.optimize as O

#Linear correlation function with Planck cosmology at z=0.1
xifile = '/gpfs/data/nmccull/emocks/xilin_planck_z0.1_113016.txt'

wp_sdss=pd.DataFrame({'wp m1':[2615, 1189, 728, 491.4, 272.8, 154.4, 111.5, 94.5, 56.8, 35.1, 22.0, 11.4, 5.89],
    'wp m2':[1028, 731.7, 392.6, 228.6, 144.6, 94.3, 70.5, 48.6, 33.1, 20.9, 11.6, 6.04, 3.28],
    'wp m3':[586.2, 402.9, 258.7, 163.2, 105.5, 68.9, 50.2, 35.5, 24.5, 15.3, 8.54, 4.11, 2.73],
    'wp m4':[455.7, 296.9, 197.0, 134.1, 89.4, 61.1, 44.0, 31.2, 21.3, 13.7, 7.65, 4.09, 3.21],
    'wp m5':[366.1, 264.3, 184.0, 128.6, 84.7, 59.4, 42.9, 30.9, 21.9, 14.6, 8.24, 4.88, 3.58],
    'wp m6':[307.0, 228.5, 159.3, 110.4, 72.9, 49.8, 34.6, 24.6, 16.7, 10.7, 5.73, 2.82, 1.39],
    'wp m7':[322.5, 231.1, 162.4, 114.6, 75.5, 50.6, 35.0, 24.2, 15.3, 9.2, 4.11, 1.81, 0.72],
    'wp m8':[313.3, 230.2, 165.4, 118.3, 79.7, 53.8, 37.4, 25.9, 17.4, 10.6, 5.31, 3.56, 0.96],
    'wp m9':[294.3, 221.5, 161.4, 114.7, 75.5, 48.6, 32.4, 19.7, 10.8, 6.35, 3.62, 2.14, 0.56],
    'err m1':[491, 202, 96.3, 55.3, 23.2, 14.5, 10.4, 5.6, 3.8, 3.2, 2.2, 1.6, 1.21],
    'err m2':[68,  34,   17.1, 10.9, 6.4, 3.7,  2.7,  2.3, 1.8, 1.5, 1.2, .95, .64],
    'err m3':[19.5, 11.7, 6.7, 4.7, 3.0, 2.2, 2.1, 1.8, 1.6, 1.3, 0.94, 0.71, 0.54],
    'err m4':[11.3, 6.9, 5.1, 4.1, 3.3, 2.6, 2.3, 2.0, 1.8, 1.5, 1.07, 0.88, 0.70],
    'err m5':[9.3, 7.6, 6.6, 5.5, 4.3, 3.6, 3.3, 3.1, 2.7, 2.1, 1.32, 1.06, 0.85],
    'err m6':[9.2, 8.3, 7.2, 5.6, 4.2, 3.4, 2.9, 2.5, 2.4, 1.9, 1.28, 1.13, 0.91],
    'err m7':[17.0, 15.3, 12.8, 10.3, 7.7, 6.0, 4.7, 3.6, 2.9, 1.78, 1.29, 1.39, 1.24],
    'err m8':[25.9, 24.9, 21.1, 17.5, 13.2, 10.5, 7.8, 5.8, 4.5, 2.6, 1.42, 1.76, 1.02],
    'err m9':[34.7, 32.1, 27.6, 22.0, 16.5, 11.5, 7.7, 4.4, 2.8, 1.93, 1.34, 1.23, 1.26],
    'rp': [0.17, 0.27, 0.42, 0.67, 1.1, 1.7, 2.7, 4.2, 6.7, 10.6, 16.9, 26.8, 42.3]})

hod_params_sdss_lowsigmaM = {'m1': {'logMmin':14.06, 'sigmaLogM':0.71, 'logM0':13.72, 'logM1':14.80, 'alpha':1.35},
    'm2': {'logMmin':13.38, 'sigmaLogM':0.69, 'logM0':13.35, 'logM1':14.20, 'alpha':1.09},
    'm3': {'logMmin':12.78, 'sigmaLogM':0.68, 'logM0':12.71, 'logM1':13.76, 'alpha':1.15}, 
    'm4': {'logMmin':12.11, 'sigmaLogM':0.01, 'logM0':11.86, 'logM1':13.41, 'alpha':1.13},
    'm5': {'logMmin':11.78, 'sigmaLogM':0.02, 'logM0':12.32, 'logM1':12.98, 'alpha':1.01},
    'm6': {'logMmin':11.56, 'sigmaLogM':0.003, 'logM0':12.15, 'logM1':12.79, 'alpha':1.01},
    'm7': {'logMmin':11.44, 'sigmaLogM':0.01, 'logM0':10.31, 'logM1':12.64, 'alpha':1.03},
    'm8': {'logMmin':11.29, 'sigmaLogM':0.03, 'logM0':9.64, 'logM1':12.48, 'alpha':1.01},
    'm9': {'logMmin':11.14, 'sigmaLogM':0.02, 'logM0':9.84, 'logM1':12.40, 'alpha':1.04}}
hod_params_sdss = {'m1': {'logMmin':14.06, 'sigmaLogM':0.71, 'logM0':13.72, 'logM1':14.80, 'alpha':1.35},
    'm2': {'logMmin':13.38, 'sigmaLogM':0.69, 'logM0':13.35, 'logM1':14.20, 'alpha':1.09},
    'm3': {'logMmin':12.78, 'sigmaLogM':0.68, 'logM0':12.71, 'logM1':13.76, 'alpha':1.15}, 
    'm4': {'logMmin':12.14, 'sigmaLogM':0.17, 'logM0':11.62, 'logM1':13.43, 'alpha':1.15},
    'm5': {'logMmin':11.83, 'sigmaLogM':0.25, 'logM0':12.35, 'logM1':12.98, 'alpha':1.00},
    'm6': {'logMmin':11.57, 'sigmaLogM':0.17, 'logM0':12.23, 'logM1':12.75, 'alpha':0.99},
    'm7': {'logMmin':11.45, 'sigmaLogM':0.19, 'logM0':9.77, 'logM1':12.63, 'alpha':1.02},
    'm8': {'logMmin':11.33, 'sigmaLogM':0.26, 'logM0':8.99, 'logM1':12.50, 'alpha':1.02},
    'm9': {'logMmin':11.18, 'sigmaLogM':0.19, 'logM0':9.81, 'logM1':12.42, 'alpha':1.04}}

NMIN={'m1':-2.0, 'm2':-2.5, 'm3':-3.0, 'm8':-3.0}
XMAX=14.9

# Compute the r-band magnitude with a dust extinction tapered to a given threshold value
def taper_magr(data, threshold):
    # r-band extinction
    
    
    data['Mr tapered'] = N.arctan(N.pi*(data['Mr dust'] - data['Mr no dust'])/(2*threshold))*2*threshold / N.pi
    
    data.loc[data['Mr tapered']<0, 'Mr tapered']=0
    data['Mr tapered'] = data['Mr tapered'] + data['Mr no dust']
    
    return data

# Compute the r-band and g-band magnitudes with a dust extinction tapered to a given threshold value
def taper_magr_magg(data, threshold):
    
    #data['Mr tapered'] = N.arctan(N.pi*(data['Mr dust'] - data['Mr no dust'])/(2*threshold))*2*threshold / N.pi
    data['Mr tapered'] = data['Mr dust'] - data['Mr no dust']
    data['Mg tapered'] = data['Mg dust'] - data['Mg no dust']
    data.loc[data['Mr tapered']<0, 'Mr tapered']=0
    data.loc[data['Mg tapered']<0, 'Mg tapered']=0
    subset = (data['Mr tapered']>0.1) & (data['Mg tapered']>0)
    fit = N.polyfit(N.log10(data.loc[subset, 'Mr tapered']), N.log10(data.loc[subset, 'Mg tapered']), 1)
    
    data['Mr tapered'] = N.arctan(N.pi*(data['Mr tapered'])/(2*threshold))*2*threshold / N.pi
    data['Mg tapered'] = (data['Mr tapered']**fit[0])*10**fit[1]
    
    data['Mr tapered'] = data['Mr tapered'] + data['Mr no dust']
    data['Mg tapered'] = data['Mg tapered'] + data['Mg no dust']
    
    return data

# Compute the projected correlation function from the 3D correlation function in a periodic box
def compute_wp_box(r, cf3D):
    cfi=I.interp1d(r, cf3D)
    # Extrapolate the given correlation function with the theoretical linear correlation function
    # Multiplied by a linear bias
    xilin=N.loadtxt(xifile)
    cfLin=I.interp1d(xilin[:,0], xilin[:,1])
    
    bias = N.mean(cfi(N.arange(20, 40, 1.0)) / cfLin(N.arange(20, 40, 1.0)))

    rp = N.logspace(-1, 2, 100)
    
    rnew = N.arange(r[-4], 200, 0.5)
    rr = N.concatenate([r[0:-4], rnew])
    cfnew = N.concatenate([cf3D[0:-4], cfLin(rnew)*bias])
    
    
    cfi=I.interp1d(rr, cfnew)
    wp=N.zeros_like(rp)
    
    for i in N.arange(wp.size):
        dlogr=0.0001
        logrr=N.arange(N.log(rp[i])+dlogr, N.log(rnew[-2]), dlogr)
        rr=N.exp(logrr)
        wp[i]=N.sum(2*cfi(rr)*rr**2/N.sqrt(rr**2-rp[i]**2)*dlogr)
    return rp, wp

# Compute the HOD from a given set of HOD parameters
# Parameters is a dict with samples, each of which is a dict
# with the following 5 keys:
# alpha, logMmin, sigmaLogM, logM0, logM1
# Also must supply the halo masses to compute the HOD at (mhalo)
def compute_hod_model(mhalo, hod_params):
    hod = {}
    for sample in hod_params.keys():
        hod[sample] = hod_model(mhalo, hod_params[sample])
    return hod

def hod_model(mhalo, params):
    arg1=(N.log10(mhalo)-params['logMmin'])/params['sigmaLogM']
    arg2=(mhalo-10.**(params['logM0']))/10.**(params['logM1'])
    arg3=1+(arg2)**(params['alpha'])
    arg3[N.isnan(arg3)]=1.0
    hod=0.5*(1.0+S.erf(arg1))*arg3
    hod[N.where(hod==0.0)]=10**-15
    return hod

# Maps mhalo to the masses in massfn using the cumulative mass function
def map_halo_mass(data, massfn, vol=542.16**3):
    mh=massfn[:,0]
    hmf_cum=massfn[:,8]
    hmf = massfn[:,7]

    subset = N.where(N.isnan(hmf)==False)
    mh=mh[subset]
    hmf_cum=hmf_cum[subset]
    isort=N.argsort(hmf_cum)
    hmf_cum=hmf_cum[isort]
    mh=mh[isort]

    
    bins=N.logspace(9.75, 15.5, 200)
    bw=N.diff(N.log10(bins))
    
    numhalo, bins=data.groupby('galaxy type')['mhhalo'].aggregate(N.histogram, bins)['central']
    bc=(bins[1:]+bins[:-1])/2.0

    ncum=N.cumsum(numhalo[::-1])[::-1]/vol

    cmf2=I.interp1d(bc, ncum, bounds_error=False, fill_value="extrapolate")
    mhi=I.interp1d(hmf_cum, mh, bounds_error=False, fill_value="extrapolate")
    
    cmf_all=cmf2(data['mhhalo'])
    
    data['mhhalo wmap7'] = mhi(cmf_all)
    
    return data

# Computes the HOD for a catalogue (mhalo, central, magr)
# at the specified mass bins (mhbins) for the number densities given in nbar
def compute_hod(mhbins, nbar, mhalo, central, magr, stellar_mass = False, massfn=None, vol=542.16**3, centsat=False, redblue=False, magg=None, cutA = 0.55, cutB = -0.047):
    # If massfn is set to something, we map the halo masses onto this mass function
    if (massfn!=None):
        mhalo=map_halo_mass(mhalo, central, massfn, vol=vol)
        
    numhalo, bins=N.histogram(mhalo[central==1], bins=mhbins)
    bc=(bins[1:]+bins[:-1])/2.0
    numhalo = numhalo.astype(N.float)
    
    if (stellar_mass):
        magr = -magr
    
    magrsort = N.sort(magr)    
    hod = {}
    for sample in nbar.keys():
        num = N.int(nbar[sample]*vol)
        subset = {}
        if (centsat==True):
            subset['central'] = ((magr < magrsort[num]) & (central == 1))
            subset['satellite'] = ((magr < magrsort[num]) & (central == 0))
        if (redblue==True):
            gmr = magg - magr
            subset['red'] = ((magr < magrsort[num]) & (gmr >= cutB*(magr+20)+cutA))
            subset['blue'] = ((magr < magrsort[num]) & (gmr < cutB*(magr+20)+cutA))
        subset['all'] = (magr < magrsort[num])
        
        hod[sample] = {}
        for split in subset.keys():
            numgal, bins=N.histogram(mhalo[subset[split]], bins=mhbins)
            numgal = numgal.astype(N.float)
            hod[sample][split] = numgal/numhalo
            hod[sample][split][N.isnan(hod[sample][split])]=0.0
    return bc, hod, numhalo
        
def add_color(data, cutA = 0.55, cutB = -0.047):
    data['color'] = 'red'
    data.loc[data['Mg tapered'] - data['Mr tapered'] < cutB*(data['Mr tapered']+20.)+cutA, 'color'] = 'blue'
    return data
    
# Returns the theoretical halo bias as a function of halo mass
# given sigma
def halo_bias(mh, sigma):
    a=0.707
    b=0.5
    c=0.6
    deltac=1.686
    nu=deltac/sigma
    bh=1.0+1.0/(N.sqrt(a)*deltac)*(N.sqrt(a)*(a*nu**2)+N.sqrt(a)*b*(a*nu**2)**(1-c)-(a*nu**2)**c/((a*nu**2)**c+b*(1-c)*(1-c/2)))
    return 0.95*bh

# Parses the mass function data
def get_massfn_bias(massfunction):
    mh=massfunction[:,0]
    sigma=massfunction[:,1]
    dndlogM=massfunction[:,7]
    subset = N.where(N.isnan(dndlogM)==False)
    mh=mh[subset]
    MASSFN_MH = mh
    sigma=sigma[subset]
    dndlogM=dndlogM[subset]
    MASSFN_DNDLOGM = dndlogM
    bias=halo_bias(mh, sigma)
    I.interp1d(mh, dndlogM, bounds_error=False, fill_value="extrapolate")
    return I.interp1d(mh, dndlogM, bounds_error=False, fill_value="extrapolate"), I.interp1d(mh, bias, bounds_error=False, fill_value="extrapolate")


def read_bootstrap_hods(bootdir, nbootstraps):
    boot_hod = pd.DataFrame()
    mh = N.array([], dtype=N.float)
    
    for i in N.arange(nbootstraps):
        tmp = N.load('%s/hod_%i.npy'%(bootdir, i))
        
        boot_hod = boot_hod.append(pd.DataFrame({'mh': tmp[:,0], 'm1': tmp[:,1], 'm2': tmp[:,2], 'm3': tmp[:,3], 'm8': tmp[:,4]}))
    return boot_hod

def numdens(x, hod, dnI):
    bw=N.diff(N.log10(x))
    return N.sum(hod[:-1]*dnI(x[:-1])*bw)

def chisq(params, *args):
    logmmin=params[0]
    sigmam=params[1]
    logm0=params[2]
    logm1=params[3]
    alpha=params[4]
    #log mh (measured)
    x=args[0]
    #log <n> (measured)
    y=args[1]
    #nbar
    nn=args[2]
    mm = args[3]
    dnI = args[4]
    LAM=1
    sigma_n=0.05*nn
    subset=N.where((y>NMIN[mm]) & (x<XMAX))
    sigma_m=1.0
    params = {'logMmin': logmmin, 'sigmaLogM': sigmam, 'logM0': logm0, 'logM1': logm1, 'alpha': alpha}
    hod=N.log10(hod_model(10.**x[subset], params))
    nmodel=numdens(10.**x[subset], 10.**hod, dnI)
    return N.sum((y[subset]-hod)**2/sigma_m**2) + (nn-nmodel)**2/sigma_n**2*LAM

def fit_HOD_params(mh, hod, nbar, massfunction, lowsigmaM=False):
    
    dnI, dnbI = get_massfn_bias(massfunction)
    
    if (lowsigmaM):
        sdss_params = hod_params_sdss_lowsigmaM
    else:
        sdss_params = hod_params_sdss
    bestfit_params={}
    bestfit_hod=pd.DataFrame()
    p0={}
    success = {}
    
    bestfit_hod['mh'] = mh
    
    for sample in nbar.keys():
        p0[sample]=[sdss_params[sample]['logMmin'], sdss_params[sample]['sigmaLogM'], sdss_params[sample]['logM0'], sdss_params[sample]['logM1'], sdss_params[sample]['alpha']]
        
        ans=O.minimize(chisq, p0[sample], args=(N.log10(mh), N.log10(hod[sample].values), nbar[sample], sample, dnI), method='SLSQP', bounds=((None, None), (0, None), (None, None), (None, None), (0, None)))
        success[sample] = ans.success
        bestfit_params[sample]={}
        if (ans.success==False):
            bestfit_params[sample]['logMmin']=-1
            bestfit_params[sample]['sigmaLogM']=-1
            bestfit_params[sample]['logM0']=-1
            bestfit_params[sample]['logM1']=-1
            bestfit_params[sample]['alpha']=-1
            bestfit_hod[sample]=N.ones_like(mh)
        else:
            bestfit_params[sample]['logMmin']=ans.x[0]
            bestfit_params[sample]['sigmaLogM']=ans.x[1]
            bestfit_params[sample]['logM0']=ans.x[2]
            bestfit_params[sample]['logM1']=ans.x[3]
            bestfit_params[sample]['alpha']=ans.x[4]
            bestfit_hod[sample] = hod_model(mh, bestfit_params[sample])
    return success, bestfit_params, bestfit_hod
    
    
    