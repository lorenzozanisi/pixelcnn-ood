# coding: utf-8
import galsim
import numpy as np
from astropy.modeling.functional_models import Sersic2D
from scipy.special import gammainc
import os
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

def bn(n):
    return 2*n-1/3+0.009876/n # Prugniels&Simien 97

def Ie_(n,flux,re):
    
    b = bn(n)
    nn = 2*n
    one  = flux*b**(nn)/(2*re**2)
    two = 2*np.pi*n*np.e**b
    three = gammainc(nn, b)

    return one*two*three

def ellipticity(x):
    return np.sqrt(1-x**2)

def make_blob_astropy(params):
    print(params)
    m_bulge,r_bulge,n_bulge,ba_bulge,m_disk,r_disk,n_disk, ba_disk = params

    x,y= np.meshgrid(np.arange(128), np.arange(128))

    flux_nMaggie_bulge = 10**((22.5-m_bulge)*0.4)
#    Ie_bulge = Ie_(n_bulge,flux_nMaggie_bulge,r_bulge)
    e_bulge = ellipticity(ba_bulge)
    bulge = Sersic2D(amplitude=1,r_eff = r_bulge,n=n_bulge,x_0=64,y_0=64,ellip=e_bulge)

    bulge = bulge(x,y)
    tot = np.sum(bulge)
    image = bulge/tot*flux_nMaggie_bulge
    image = image/0.396**2

    if m_disk != 999:
        flux_nMaggie_disk = 10**((22.5-m_disk)*0.4)
#        Ie_disk = Ie_(n_disk,flux_nMaggie_disk,r_disk)
        e_disk = ellipticity(ba_disk)
        disk = Sersic2D(amplitude=1, r_eff = r_disk, n=n_disk, x_0=64,y_0=64, ellip=e_disk)

        disk = disk(x,y)
        tot = np.sum(disk)
        disk = disk/tot*flux_nMaggie_disk

        gal = bulge+disk       
        image = gal/0.396**2

    surfaceBrightness_nMaggie = 22.5 -2.5*np.log10(image)
    surfaceBrightness_nMaggie[np.isnan(surfaceBrightness_nMaggie)] = 999

    return surfaceBrightness_nMaggie


def make_blob(params, Sersic=True):

    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
    arcsec_to_rad = 4.848e-6
    scale_rad_pix = 0.396*arcsec_to_rad

#    m_bulge,n_bulge,r_bulge,pa_bulge,ba_bulge,m_disk,r_disk,n_disk, pa_disk, ba_disk = params

    if Sersic:
        m_bulge,r_bulge,n_bulge,ba_bulge, objid,z = params
    else:
        m_bulge,r_bulge,n_bulge,ba_bulge,m_disk,r_disk,n_disk, ba_disk, objid,z = params
        
        
    try: 
        
        if Sersic:
            flux_nMaggie_bulge = 10**((22.5-m_bulge)*0.4)
            bulge = galsim.Sersic(n=n_bulge, half_light_radius=r_bulge)
            bulge = bulge.withFlux(flux_nMaggie_bulge)
            bulge_shape = galsim.Shear(q=ba_bulge, beta=90*galsim.degrees)
            gal = bulge.shear(bulge_shape)
       

        else:
            if m_disk==999:
                flux_nMaggie_bulge = 10**((22.5-m_bulge)*0.4)
                bulge = galsim.Sersic(n=n_bulge, half_light_radius=r_bulge)
                bulge = bulge.withFlux(flux_nMaggie_bulge)
                bulge_shape = galsim.Shear(q=ba_bulge, beta=90*galsim.degrees)
                gal = bulge.shear(bulge_shape)
            else:
                flux_nMaggie_bulge = 10**((22.5-m_bulge)*0.4)
                flux_nMaggie_disk = 10**((22.5-m_disk)*0.4)

                bulge = galsim.Sersic(n=n_bulge, half_light_radius=r_bulge)
                disk = galsim.Sersic(n=1, scale_radius=r_disk)

                bulge = bulge.withFlux(flux_nMaggie_bulge)
                bulge_shape = galsim.Shear(q=ba_bulge, beta=90*galsim.degrees)
                bulge = bulge.shear(bulge_shape)

                disk = bulge.withFlux(flux_nMaggie_disk)
                disk_shape = galsim.Shear(q=ba_disk, beta=90*galsim.degrees) 
                disk = disk.shear(disk_shape)

                gal = bulge + disk
    except:
        size = 4096
        pars = galsim.GSParams(maximum_fft_size=4*size)
        psf = galsim.Gaussian(fwhm=0.1)
        total = galsim.Convolve([psf,gal], gsparams=pars)
        image = galsim.ImageF(256,256)
        total.drawImage(image=image, scale=0.396) 

        image = image.array/(0.396)**2
        image = image[64:192,64:192]
        image[image<0] = 0

        d_ang = cosmo.angular_diameter_distance(z).value*1.e6
        scale_pc_pix = scale_rad_pix*d_ang

        hdu = fits.PrimaryHDU()
        hdu.header['REDSHIFT'] = z
        hdu.header['CDELT1'] = round(scale_pc_pix,3)
        hdu.header['PSF_FWHM'] = 0.1
        surfaceBrightness_nMaggie = 22.5 -2.5*np.log10(image)
        surfaceBrightness_nMaggie[np.isnan(surfaceBrightness_nMaggie)] = 999
        surfaceBrightness_nMaggie[np.isinf(surfaceBrightness_nMaggie)] = 999
        hdu.data = surfaceBrightness_nMaggie
        hdu.writeto('./forConnor_SerOnly/'+str(objid)+'_r.fits', clobber=True)

        return surfaceBrightness_nMaggie
    except:
        os.system('echo '+str(objid)+' >> failed_SerOnly.txt')
        return #np.nan
