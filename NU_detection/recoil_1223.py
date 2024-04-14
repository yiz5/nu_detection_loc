import numpy as np
import os
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from read import *

path='../'

def get_sin2theta(flavor, correction = False ):
    #https://arxiv.org/abs/1810.05606
    sin2theta =0.231 # 0.2223#
    r2_nue = -0.83e-32 * u.cm**2
    r2_numu = -0.48e-32 * u.cm**2
    r2_nutau = -0.3e-32 * u.cm**2
    m_w = 80.385 * u.GeV/const.c**2
    if correction and flavor == 'nue':
        return sin2theta * (1+ 1/3 * (m_w**2 * r2_nue/(const.hbar*const.c)**2 * const.c**4).decompose())
    elif correction and flavor == 'numu':
        return sin2theta * (1+ 1/3 * (m_w**2 * r2_numu/(const.hbar*const.c)**2 * const.c**4).decompose())
    elif correction and flavor == 'nutau':
        return sin2theta * (1+ 1/3 * (m_w**2 * r2_nutau/(const.hbar*const.c)**2 * const.c**4).decompose())
    else:
        return sin2theta

def setup_cdfpdf_ax(ax, title, xlabel, ylabel, gridTF, cdfpdf, label_size, font_size, 
                    vlines = [0], hlines = [0], xlims = [0,0], ylims = [0,0], log = [0,0]):
    ax.yaxis.set_tick_params(labelsize=label_size) 
    ax.xaxis.set_tick_params(labelsize=label_size)  
    ax.set_title(title, fontsize = font_size)
    ax.grid(gridTF)
    if cdfpdf in ['cdf']:
        ylabel = 'Event rate '+ r'[ton$^{-1}$ yr$^{-1}$]'
    elif cdfpdf in ['pdf']:
        ylabel = r'$\dfrac{dR}{dE_{r}}$  '+ '[ton$^{-1}$ yr$^{-1}$ '+'keV'+'$^{-1}$]'
    ax.set_xlabel(xlabel, fontsize = label_size)
    ax.set_ylabel(ylabel, fontsize = label_size)
    
    if sum(xlims)>0:
        ax.set_xlim(xlims[0], xlims[1])
    if sum(ylims)>0:
        ax.set_ylim(ylims[0], ylims[1])
    if sum(vlines)>0:
        for vline in vlines:
            ax.axvline(x = vline, lw = 3, ls = '--', color = 'black')
    if sum(hlines)>0:
        for hline in hlines:
            ax.axhline(y = hline, lw = 3, ls = '--', color = 'black')
    if log[0]!=0:
        ax.set_xscale('log')
    if log[1]!=0:
        ax.set_yscale('log')
    return ax 


def get_Enu_min_Nrecoil(Er, M):
    #https://arxiv.org/abs/0903.3630
    Enu_GeV_min = np.sqrt(M*Er/2).to(u.GeV)
    return Enu_GeV_min

def get_Enu_min_erecoil(Er):
    #https://arxiv.org/pdf/1307.5458.pdf
    m_e_GeV = (const.m_e*const.c**2).to(u.GeV)
    Er_GeV = Er.to(u.GeV)
    Enu_GeV_min = 0.5 * (Er_GeV + np.sqrt( Er_GeV *(Er_GeV+2*m_e_GeV) ) )
    return Enu_GeV_min


def calculate_dsigma_dEr_erecoil(Er, Enu, correction = False):
    #https://arxiv.org/abs/1307.5458 eqn5
    
    sin2theta = get_sin2theta('nue', correction = correction)
    #print('sin2theta = ', sin2theta)
        
    g_v = 2 * sin2theta - 0.5
    g_a = -0.5
    m_e = (const.m_e*const.c**2).to(u.GeV)
    Gf = 1.1663787e-5*u.GeV**(-2)
    
    Er_GeV = Er.to(u.GeV)
    Enu_GeV = Enu.to(u.GeV)
    
    g_v_ee = g_v + 1
    g_a_ee = g_a + 1 #https://arxiv.org/abs/1307.5458
    dsigma_dEr_nue = (Gf**2) * m_e / 2/np.pi * ((g_v_ee + g_a_ee)**2 + 
                                            (g_v_ee - g_a_ee)**2 * (1-Er_GeV/Enu_GeV)**2 + 
                                            (g_a_ee**2 - g_v_ee**2)*m_e*Er_GeV/Enu_GeV**2
                                           )#  1/GeV**3
    
    dsigma_dEr_nutaumu = (Gf**2) * m_e / 2/np.pi * ((g_v + g_a)**2 + 
                                            (g_v - g_a)**2 * (1-Er_GeV/Enu_GeV)**2 + 
                                            (g_a**2 - g_v**2)*m_e*Er_GeV/Enu_GeV**2
                                           )#  1/GeV**3
    
    #https://arxiv.org/pdf/1308.0443.pdf
    #print('nue: ', dsigma_dEr_nue, ', numu: ', dsigma_dEr_nutaumu, 'ratio: ', dsigma_dEr_nutaumu/dsigma_dEr_nue )
    
    
    #dsigma_dEr = dsigma_dEr_nue#+dsigma_dEr_nutaumu
    return dsigma_dEr_nue.to(1/u.GeV**3), dsigma_dEr_nutaumu.to(1/u.GeV**3)#　dsigma_dEr.to(1/u.GeV**3)# 1/GeV**3

def get_dR_dEr_pertonyrkeV(Ers, nucleus, Enu, dN_dEnus, recoil_type, pcle, ebinding = True, corr = False):
    if nucleus in ['Xenon']:
        A = 131.293
        Z = 54
        N = A-Z
        
        #http://www.chemistry.uoguelph.ca/educmat/atomdata/bindener/grp18num.htm
        #http://www.prvky.com/elements/xenon-electron-configuration.html
        e_bind = np.array([34563]*2 + [5454.8]*2 + [4891.4]*6 + [1148]*2 + [959.78]*6 + [681.02]*10 + [214.63]*2 + [153.48]*6 + [
    68.146]*10 +[23.39]*2 + [12.563]*6)*(u.eV)
    elif nucleus in ['Argon']:
        A = 39.948# 40
        Z = 18
        N = A-Z        
        e_bind = np.array([3206.2]*2 + [324.2]*2 + [247.74]*6 + [29.24]*2 + [15.76]*6)*(u.eV)
    elif nucleus in ['Helium']:
        A = 4
        N = 2
        Z = 2
        
    M = A* (const.u*const.c**2).to(u.GeV)
    print('nucleon mass: ', M)
    #print('ebind: ', e_bind)
    
    
    #Na*m_amu = 1e-3, m_amu * A = m_Atom, so 1/m_atom =Na/A ;(const.N_A.value)/u.g*const.u).decompose() = 1
    Norm = (const.N_A.value)/u.g /A

    #print(len(Emins))
    dR_dEr_pertonyrkeVs = []
    print(const.N_A.value, A, Norm, M)
   
    Enu_GeV = Enu.to(u.GeV)
    dEnu_GeV = np.diff(Enu_GeV)
    
    for Er in Ers.to(u.GeV):
        atomic_binding = np.heaviside(Er.to(u.keV).value - e_bind.to(u.keV).value,1) 
        if recoil_type in ['NR']:
            Enu_min_GeV = get_Enu_min_Nrecoil(Er, M)
            dsigma_dEr = calculate_dsigma_dEr_Nrecoil (Er, Enu_GeV, nucleus, correction = corr) 
            
        
            #dR_dEr = sum( dsigma_dEr[:-1] * dN_dEnus[:-1]*  dEnu_GeV * heviside_fn_Enu[:-1])
            #print(dsigma_dEr.shape, dN_dEnus.shape,dEnu_GeV.shape, heviside_fn_Enu.shape )
            #dR_dEr = sum( dsigma_dEr* dN_dEnus*  dEnu_GeV * heviside_fn_Enu)

            heviside_fn_Enu = np.heaviside(Enu_GeV - Enu_min_GeV,1)    
            if pcle in ['Be7_384',  'Be7_861','Be7_384FE', 'Be7_384FEmno', 'Be7_861FE','Be7_861FEmno', 'pep']:  
                dR_dEr = sum( dsigma_dEr* dN_dEnus*  heviside_fn_Enu)#prob_ee* 
                
            elif pcle in ['pp','ppFE', 'ppFEmno', '8B', 'hep', 'N13', 'O15', 'F17'] or pcle[:5] in ['atmNu']:  
                
                dR_dEr = sum( dsigma_dEr[1:] * dN_dEnus[1:]*  dEnu_GeV * heviside_fn_Enu[1:])
                
                #print( Er,Enu_min_GeV, sum(heviside_fn_Enu))#,dsigma_dEr,dEnu_GeV,heviside_fn_Enu#prob_ee,
            elif pcle in ['simatmNu']:    
                print(len(dsigma_dEr),len(dN_dEnus), len(dEnu_GeV), len(heviside_fn_Enu))
                dR_dEr = sum( dsigma_dEr[1:]* dN_dEnus*  dEnu_GeV * heviside_fn_Enu[1:])    
            dR_dEr_pertonyrkeV = (Norm*dR_dEr*(const.hbar*const.c)**2).to(1/u.tonne/u.yr/u.keV ) 

        
            #print(dsigma_dEr.shape, dN_dEnus.shape, dEnu_GeV.shape, heviside_fn_Enu.shape)
        elif recoil_type in ['ER']:
            prob_ee = 0.553 #https://arxiv.org/abs/1610.04177
            Enu_min_GeV = get_Enu_min_erecoil(Er)
            dsigma_dEr_nue, dsigma_dEr_nutaumu = calculate_dsigma_dEr_erecoil(Er, Enu, correction = corr) 
            heviside_fn_Enu = np.heaviside(Enu_GeV - Enu_min_GeV,1)    
            #print(Enu_min_GeV,Enu_GeV,  heviside_fn_Enu )
            if pcle in ['Be7_384', 'Be7_861', 'Be7_384FE', 'Be7_384FEmno', 'Be7_861FE','Be7_861FEmno', 'pep']:  
                
                
                dR_dEr = sum( dsigma_dEr_nue*dN_dEnus*  prob_ee*  heviside_fn_Enu)+ sum(
                    dsigma_dEr_nutaumu*dN_dEnus*  (1-prob_ee)*  heviside_fn_Enu)

            elif pcle in ['pp','ppFE', 'ppFEmno', '8B', 'hep','N13', 'O15', 'F17']:    
                
                dR_dEr = sum( dsigma_dEr_nue[1:] * dN_dEnus[1:]*  prob_ee* dEnu_GeV * heviside_fn_Enu[1:])+ sum( 
                    dsigma_dEr_nutaumu[1:] * dN_dEnus[1:]*  (1-prob_ee)* dEnu_GeV * heviside_fn_Enu[1:])

                #print(prob_ee, Er,Enu_min_GeV, sum(heviside_fn_Enu))#,dsigma_dEr,dEnu_GeV,heviside_fn_Enu
            if ebinding:
                dR_dEr_pertonyrkeV = (sum(atomic_binding)*Norm*dR_dEr*(const.hbar*const.c)**2).to(1/u.tonne/u.yr/u.keV ) 
            else:
                dR_dEr_pertonyrkeV = (Z*Norm*dR_dEr*(const.hbar*const.c)**2).to(1/u.tonne/u.yr/u.keV ) 
        
        dR_dEr_pertonyrkeVs.append(dR_dEr_pertonyrkeV.value)

    return dR_dEr_pertonyrkeVs*dR_dEr_pertonyrkeV.unit


def calculate_dsigma_dEr_Nrecoil (Er, Enus, nucleus, correction = False): # all in GeV
    #https://arxiv.org/pdf/1307.5458.pdf EQN3
    if nucleus in ['Xenon']:
        A = 131.293
        Z = 54
        N = A-Z
    elif nucleus in ['Argon']:
        A = 40
        N = 22
        Z = 18
    elif nucleus in ['Helium']:
        A = 4
        N = 2
        Z = 2
    Gf = 1.1663787e-5*u.GeV**(-2)

    M = A*0.9315*u.GeV
    
    sin2theta = get_sin2theta('numu', correction = correction)
    #print('sin2theta = ', sin2theta)
    
  
    Q = N - (1-4*sin2theta)* Z
    formfactor2 = get_form_factor(A, Z, N, Er) #number
    
    dsigma_dEr = Gf**2 /(4*np.pi)* Q**2 * formfactor2 * M * (1- (M* Er)/(2*Enus**2))
    dsigma_dEr_GeV = dsigma_dEr.to(1/u.GeV**3)
    return dsigma_dEr # GeV^(-3)#length of Enus


def get_form_factor(A, Z, N, Er):
    #https://www.tir.tw/phys/hep/dm/amidas/equations/eq-FQ_Helm.html
    s = 0.9*u.fm  #fm
    c = (1.23* A**(1/3) - 0.6)*u.fm #fm
    a = 0.52 *u.fm #fm
    R0 = np.sqrt(c**2 +(7/3)*(np.pi*a)**2 - 5*s**2) 
    
    #M_P = 0.938272*u.GeV
    #M_N = 0.939565*u.GeV
    M = A * 0.9315*u.GeV 
    
    q = np.sqrt(2*M*Er)#GeV 
    qR0_dim = (q*R0/(const.hbar*const.c)).decompose().value
    qs_dim = (q*s/(const.hbar*const.c)).decompose().value
    #R0 * q is dimensionless
    #print((q*R0/(const.hbar*const.c)).decompose().unit)
    j1 = np.sin(qR0_dim) / (qR0_dim)**2 - np.cos(qR0_dim) / (qR0_dim) #dimensionless
    form_fac2 = (3* j1/(qR0_dim))**2 * np.exp(-qs_dim**2) #dimensionless
    
    return np.array(form_fac2)#dimensionless


def get_norm(pcle, nuclei, ebinding = True):
    Er_keV, pdf = read_pcle_pdf(pcle, nuclei, ebinding = ebinding, plot = False)
    norm = max(get_cdf(pdf[:-1]/u.tonne/u.yr/u.keV, np.diff(Er_keV)*u.keV, 'survival'))
    return norm

def get_atm_loc_avg_fluc(detector, nuclei):
    atm_2009 = 'atmNu'+'_'+detector+'_'+'2009'
    atm_2014 = 'atmNu'+'_'+detector+'_'+'2014'
    norm_2009 = get_norm(atm_2009, nuclei)
    norm_2014 = get_norm(atm_2014, nuclei)
    norm_avg = (norm_2009+norm_2014)/2
    max_fluc = (norm_2009/norm_2014)-1
    A = max_fluc/2
    return norm_avg, max_fluc.value, A.value


def get_recoil_type(pcle):
    if pcle in ['pp','Be7_384', 'Be7_861', 'ppBe', 'CNO','pep','nubb','N13','O15','F17', 'Rn222', 'Kr85']:
        recoil_type = 'ER'
    elif pcle in [ 'hep','8B', 'dsnb', 'dsnbtot'] or pcle[:5] in ['atmNu']:
        recoil_type = 'NR'
    return recoil_type


#neutrino flux
#https://arxiv.org/pdf/1208.5723.pdf
def flux_norm(pcle, norm_unit = (u.cm**2 * u.s)**(-1), metallicity_model = 'low'):
    print(pcle)
    #AGSS09-SFII
    if metallicity_model == 'low':
        print('AGSS09-SFII')
        if pcle[:2] in ['pp']:
            types = '_MeV_nu_spectrum'
            normalization = 6.03e10*norm_unit
        elif pcle in ['hep']:
            types = '_MeV_nu_spectrum'
            normalization = 8.31e3*norm_unit
        elif pcle in ['8B']:
            types = '_MeV_nu_spectrum'
            normalization =  4.59e6*norm_unit
        elif pcle in ['N13']:
            types = ''
            normalization =   2.17e8*norm_unit
        elif pcle in ['O15']:
            types = ''
            normalization =  1.56e8*norm_unit
        elif pcle in ['F17']:
            types = ''
            normalization =  3.40e6*norm_unit
        elif pcle in ['Be7_384','Be7_384FE', 'Be7_384FEmno']:
            normalization = 4.56e9*norm_unit * 0.1
            Enu_MeV = [0.38447]*u.MeV 
        elif pcle in ['Be7_861','Be7_861FE','Be7_861FEmno']:
            normalization = 4.56e9*norm_unit * 0.9
            Enu_MeV = [0.86227]*u.MeV  #0.86227
        elif pcle in ['pep']:
            normalization = 1.47e8*norm_unit # 1.39e8
            Enu_MeV = [1.442]*u.MeV
            
    #GS98-SFII
    if metallicity_model == 'high':
        print('GS98-SFII')
        if pcle[:2] in ['pp']:
            types = '_MeV_nu_spectrum'
            normalization = 5.98e10*norm_unit
        elif pcle in ['hep']:
            types = '_MeV_nu_spectrum'
            normalization = 8.04e3*norm_unit
        elif pcle in ['8B']:
            types = '_MeV_nu_spectrum'
            normalization = 5.58e6*norm_unit
        elif pcle in ['N13']:
            types = ''
            normalization = 2.96e8*norm_unit
        elif pcle in ['O15']:
            types = ''
            normalization = 2.23e8*norm_unit
        elif pcle in ['F17']:
            types = ''
            normalization = 5.52e6*norm_unit
        elif pcle in ['Be7_384','Be7_384FE', 'Be7_384FEmno']:
            normalization = 5.00e9*norm_unit * 0.1
            Enu_MeV = [0.38447]*u.MeV 
        elif pcle in ['Be7_861','Be7_861FE','Be7_861FEmno']:
            normalization = 5.00e9*norm_unit * 0.9
            Enu_MeV = [0.86227]*u.MeV  #0.86227
        elif pcle in ['pep']:
            normalization = 1.44e8*norm_unit # 1.39e8
            Enu_MeV = [1.442]*u.MeV
              
        
    if pcle in ['pep', 'Be7_384', 'Be7_861', 'Be7_384FE', 'Be7_384FEmno', 'Be7_861FE', 'Be7_861FEmno']: 
        print(pcle, 'mono energy' )
        return Enu_MeV, qua2arr([normalization])
    
    elif pcle[:5] in ['atmNu']:#FLUKA
        print(pcle)
        detector = pcle.split('_')[1]
        year = pcle.split('_')[-1]
        atmflux_folder = os.path.join(os.path.join(path, 'real_data_nest'),
                            'atmflux')
        if year in ['FLUKA']:
            atmflux_file = os.path.join(atmflux_folder, 
                               detector+'_'+year+'_perGeVm2s.txt')
        else:
            atmflux_file = os.path.join(os.path.join(atmflux_folder, detector),
                               detector+'_'+year+'_Trackback_perGeVcm2s.txt')
              
        Enu_GeV, atm_flux = np.array(read_file_dataline(atmflux_file)).T

        Enu_MeV = (Enu_GeV*u.GeV).to(u.MeV)
        Enu_MeV_centers = (Enu_MeV[:-1]+Enu_MeV[1:])/2
        dEnu_MeV = np.append(np.diff(Enu_MeV), np.diff(Enu_MeV)[0])
        if year in ['FLUKA']:
            dN_dEnu_percm2sMeV = (atm_flux/u.GeV/u.m**2/u.s).to(1/u.MeV/u.cm**2/u.s) 
        else:
            unit_atm_flux = 1/(u.GeV * u.s * u.cm**2)
        
            dN_dEnu_percm2sMeV = (atm_flux/u.GeV/u.cm**2/u.s).to(1/u.MeV/u.cm**2/u.s) 

        return Enu_MeV,dN_dEnu_percm2sMeV
    
    else:    
        print(pcle)
        data = read_file_data(os.path.join(os.path.join(path,'real_data_nest'),pcle+types+'.txt'))
        Enu_MeV = data[0]*u.MeV
        Enu_MeV_centers = (Enu_MeV[:-1]+Enu_MeV[1:])/2
        dEnu_MeV = np.append(np.diff(Enu_MeV), np.diff(Enu_MeV)[0])
    
        dN_dEnu_percm2sMeV = normalization * data[1]/u.MeV #counts per cm2 per second per MeV

        return Enu_MeV,dN_dEnu_percm2sMeV
    
    

    
def get_thinNEST_pdf(pcle):
    if pcle in ['nubb_ER' , 'atmTot_NR', 'B8_NR', 'dsnb_NR', 'hep_NR', 'ppBe_ERl', 'ppBe_ERmno']:
        data = read_file_dataline(os.path.join(os.path.join(os.path.join(path,'thinNEST'),'spectra'),pcle+'_xe.dat'))
        E_R = data[0]*u.keV
        #print(E_R)
        #E_R_center = (E_R[:-1]+E_R[1:])/2
        dN_dEr =data[1]/u.tonne/u.yr/u.keV
        #dE_R = np.append(np.diff(E_R), np.diff(E_R)[0])*u.keV
        #y = list(data[1])/u.tonne/u.yr/u.keV
    elif pcle in [ 'dsnb']:
        data = read_file_dataline(os.path.join(os.path.join(os.path.join(path,'thinNEST'),'spectra'),pcle+'.dat'))
        E_R = data[0]*u.keV
        #print(E_R)
        #E_R_center = (E_R[:-1]+E_R[1:])/2
        dN_dEr =data[1]/u.tonne/u.yr/u.keV
        #dE_R = np.append(np.diff(E_R), np.diff(E_R)[0])*u.keV
        #y = list(data[1])/u.tonne/u.yr/u.keV
    else:
        print('pcles are nubb_ER , atmTot_NR, B8_NR, dsnb_NR, hep_NR, ppBe_ERl, ppBe_ERmno')
    return E_R, dN_dEr


#recoil energy range for calculation
def get_fittingcurve_eMax_keV(pcle, nuclei, recoil_type):
    if pcle in ['pp']:
        if recoil_type in ['ER']:
            if nuclei in ['Argon']:
                eMax = 2000
            elif nuclei in ['Xenon']:
                eMax = 300
        elif recoil_type in ['NR']:
            if nuclei in ['Argon']:
                eMax = 0.05
            elif nuclei in ['Xenon']:
                eMax = 4e-3
            
            
    elif pcle in ['Be7_384']:
        if recoil_type in ['ER']:
            if nuclei in ['Argon']:
                eMax = 2000
            elif nuclei in ['Xenon']:
                eMax = 260
        elif recoil_type in ['NR']:
            if nuclei in ['Argon']:
                eMax = 0.05
            elif nuclei in ['Xenon']:
                eMax = 4e-3
            
    elif pcle in ['Be7_861']:
        if recoil_type in ['ER']:
            if nuclei in ['Argon']:
                eMax = 2000
            elif nuclei in ['Xenon']:
                eMax = 850
        elif recoil_type in ['NR']:
            if nuclei in ['Argon']:
                eMax = 0.1
            elif nuclei in ['Xenon']:
                eMax = 0.02

    elif pcle in ['pep']:
        if recoil_type in ['ER']:
            eMax = 2000
        elif recoil_type in ['NR']:
            if nuclei in ['Xenon']:
                eMax = 0.05
                
    elif pcle in ['F17']:
        if recoil_type in ['ER']:
            eMax = 2000
        elif recoil_type in ['NR']:
            eMax = 0.07
            
    elif pcle in ['O15']:
        if recoil_type in ['ER']:
            eMax = 2000
        elif recoil_type in ['NR']:
            eMax = 0.07
            
    elif pcle in ['N13']:
        if recoil_type in ['ER']:
            if nuclei in ['Argon']:
                eMax = 2000
            elif nuclei in ['Xenon']:
                eMax = 1000
        elif recoil_type in ['NR']:
            eMax = 0.05
            
    elif pcle in ['nubb']:
        eMax = 2000
    
    elif pcle in ['dsnbtot']:
        eMax = 20
    elif pcle in ['hep']:
         eMax = 50
    elif pcle in ['8B']:
        if recoil_type in ['ER']:
            eMax = 100000
        elif recoil_type in ['NR']:
            eMax = 50
    elif pcle[:5] in ['atmNu']:
        if nuclei == 'Xenon':
            eMax = 100
        elif nuclei == 'Argon':
            eMax = 800
 
    return eMax*u.keV

#recoil energy bins
def get_Er(pcle, nuclei, recoil_type, start=-9, binnum= 200):
    #binnum = 200
    #start = -9
    emax_keV = get_fittingcurve_eMax_keV(pcle, nuclei, recoil_type)
    emax_GeV = emax_keV.to(u.GeV).value
    Er_GeV = np.logspace(start, np.log10(emax_GeV), binnum)*u.GeV    
    return Er_GeV 


def get_Ebind(pcle,recoil_type, ebinding):
    if ebinding:
        bind = '_Ebind_'
    else:
        bind = '_FE_'
    if (pcle in ['8B', 'hepNR','hep', 'dsnbtotNR','dsnb', 'dsnbtot'] or pcle[:5] in ['atmNu']) and  recoil_type =='NR':
        bind = '_'
    if pcle in ['nubbER', 'nubb', 'Rn222', 'Kr85']:
        bind = '_'
    if recoil_type =='NR':
        bind = '_'
    return bind


def get_pmf(dR_dEr, dEr, norm = True):
    dR_dEe_pertonyrs = (dR_dEr*dEr).to(1/u.tonne/u.yr)
    if norm:
        return dR_dEe_pertonyrs/sum(dR_dEe_pertonyrs)
    else:
        return dR_dEe_pertonyrs.value
    
def get_cdf(dR_dEr, dEr, types):
    dR_dEe_pertonyrs = (dR_dEr*dEr).to(1/u.tonne/u.yr)
    if types == 'survival':
        print('survival')
        dR_dEe_pertonyr = dR_dEe_pertonyrs[::-1].cumsum()[::-1].to(1/u.tonne/u.yr)
    else:
        dR_dEe_pertonyr = (dR_dEe_pertonyrs).cumsum()
   
    return dR_dEe_pertonyr

def read_pcle_pdf(pcle, nuclei, eff_type, recoil_type = '', metallicity_model = '', ebinding = True, plot = True, corr = ''):
    if recoil_type == '':
        recoil_type = get_recoil_type(pcle)
    print(recoil_type)
    ebind = get_Ebind(pcle,recoil_type, ebinding)
    
    if pcle in ['dsnb', 'dsnbtot', 'nubb'] and nuclei == 'Xenon' and corr == '':
        if pcle in ['dsnb']:
            Er_keV, pdf = [i.value for i in get_thinNEST_pdf(pcle)]    
        else:
            Er_keV, pdf = [i.value for i in get_thinNEST_pdf(pcle+'_'+recoil_type)]    

        
    else:
        if pcle in ['dsnb'] and nuclei =='Argon' and corr == '':
            pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                'dsnb_Ar.txt')
        elif pcle in ['dsnb'] and nuclei =='Argon' and corr != '':
            pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                'dsnb_NR_Argon_pdf'+corr+'.txt')
            
        elif pcle in ['dsnb'] and nuclei == 'Xenon' and corr !='':
            pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                pcle+'_'+recoil_type+ebind+nuclei+'_pdf'+corr+'.txt')
            
        elif pcle in ['Rn222'] and nuclei == 'Argon' and corr=='':
            pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                pcle+'_'+recoil_type+ebind+nuclei+'_per400tyr10keV_pdf.txt')
        
        elif pcle in ['Rn222'] and nuclei == 'Argon' and corr !='':
            pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                pcle+'_'+recoil_type+ebind+nuclei+'_pdf'+corr+'.txt')
            
        elif pcle == '8B' and nuclei == 'Xenon' and 'deteff' in eff_type:
            print('using 8B pdf from experiment', eff_type)
            pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                '8B_NR_Xenon_pdf_'+eff_type+'.txt')
        else:
            if pcle not in ['pp' ,'Be7_861', 'Be7_384', 'CNO', 'N13','O15' ,'F17', 'pep','hep', '8B']:
                pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                pcle+'_'+recoil_type+ebind+nuclei+'_pdf'+corr+'.txt')
        
            else:
                print('else')
                pdf_path = os.path.join(os.path.join(os.path.join(path, 'real_data_nest'),'pdf'),
                                pcle+'_'+recoil_type+ebind+nuclei+'_pdf_'+metallicity_model+corr+'.txt')
        print(corr, pdf_path)

        Er_keV, pdf = np.array(read_file_dataline(pdf_path))
        if pcle in ['Rn222'] and nuclei == 'Argon' and corr=='':
            pdf = pdf/4000
            Er_keV = Er_keV*1000
    if len(eff_type)>0:
        print(eff_type)
        if pcle == '8B' and nuclei =='Xenon' and 'deteff' in eff_type:
            pdf_deteff = pdf
        else:
            #print(Er_keV)
            pdf_deteff =pdf * get_detector_efficiency(Er_keV, eff_type)
        
        if nuclei == 'Xenon' and 'Argon' in eff_type:
            print('Wrong detector efficiency, Argon eff for Xenon')
            pdf_deteff = pdf * 0
        elif nuclei == 'Argon' and ('Xe' in eff_type or 'LZ' in eff_type):
            print('Wrong detector efficiency, Xenon eff for Argon')
            pdf_deteff = pdf * 0
         
    if plot:
        fig, ax = plt.subplots()
        ax = setup_cdfpdf_ax(ax, pcle, 'ErkeV', '/tyrkeV', True, 'pdf', 20, 20, 
                    vlines = [1], hlines = [0], xlims = [0,0], ylims = [max(pdf)*1e-5, max(pdf)*1.1], log = [1,1])
        ax.loglog(Er_keV, pdf)
        if len(eff_type)>0:
            print(recoil_type, 'add detector efficiency')
            ax.loglog(Er_keV, pdf_deteff)
    if len(eff_type)>0:
        return Er_keV, pdf_deteff
    else:
        return Er_keV, pdf

    
rand = np.random.default_rng(42)
def read_pcle_cdf(pcle, nuclei, E_threshold_keV, eff_type, recoil_type = '',
                  read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, metallicity = '', ebind = True, plot_pdf = True, corr = ''):
    if read_pdf:
        Er_keV, pdf = read_pcle_pdf(pcle, nuclei, eff_type, recoil_type = recoil_type, metallicity_model = metallicity, 
                                    ebinding = ebind, plot = plot_pdf, corr = corr)
        

    Er_threshold = Er_keV>=E_threshold_keV
    
    Er_keV = Er_keV[Er_threshold]
    pdf = pdf[Er_threshold]
    
    if endpt >0:
        good = Er_keV[:-1]<= endpt
    else:
        good = pdf[:-1]>0
        
    #if pcle[:5] == 'atmNu' and corr == '':
    #    good = [pdf[i]>pdf[i+1] for i in range(0, len(pdf)-1)]
        
    dEr = np.diff(Er_keV)[good]
    cdf = get_cdf(pdf[:-1][good]/u.keV/u.tonne/u.yr, dEr*u.keV, 'survival')
    pmf = get_pmf(pdf[:-1][good]/u.keV/u.tonne/u.yr, dEr*u.keV)
    try:
        exposure = max(cdf)
    except ValueError:
        exposure = 0*cdf.unit
    
    try:
        E_endpt = max(Er_keV[:-1][good])
    except ValueError:
        E_endpt = 0*u.keV
        
    return exposure, E_endpt, Er_keV[:-1][good], pdf[:-1][good], pmf, cdf

def get_deteff_name(eff_type):
    if eff_type == 'ER_Xe100_S1S2':
        return 'Xe_ER NEST_S1S2'
    if eff_type == 'ER_Xe1t':
        return 'Xe_ER 2006.09721'
    if eff_type == 'ER_Xe100_S1':
        return 'Xe_ER NEST_S1'
    
    if eff_type == 'NR_Xe100_S1S2':
        return 'Xe_NR NEST_S1S2'
    if eff_type == 'NR_Argon_tot':
        return 'Ar_NR Argon_tot'# 1510.00702v3
    else:
        return eff_type

    
    
def get_detector_efficiency(keVnr, eff_type, 
                            A = 17.106, B = 1.8223, C = 0.65911, D = 18.292, E = 20869, F = -2.35):
    if eff_type == 'LUX03':
        Efficiency_fn  = 10 ** ( 2 - A * np.exp ( -B * keVnr ** C) - D * np.exp( -E * keVnr ** F ) ) /100
        #where A,B,C,D,E,F are free parameters
    else:
        effieicney_keV, EFF = read_file_dataline(os.path.join(path,os.path.join('real_data_nest', 'det_eff_'+eff_type+'.txt')))
        f = interp1d(effieicney_keV, EFF, fill_value=(0, EFF[-1]), bounds_error=False)
        Efficiency_fn = f(keVnr)

    return Efficiency_fn    
'''
def get_detector_efficiency(keVnr, eff_type, A = 17.106, B = 1.8223, C = 0.65911, D = 18.292, E = 20869, F = -2.35):
    if eff_type == 'LUX03':
        Efficiency_fn  = 10 ** ( 2 - A * np.exp ( -B * keVnr ** C) - D * np.exp( -E * keVnr ** F ) ) /100
        #where A,B,C,D,E,F are free parameters
    else:
        effieicney_keV, EFF = read_file_dataline(os.path.join(path,os.path.join('real_data_nest', 'det_eff_'+eff_type+'.txt')))
        efficiency_Es = [np.interp(x_test, effieicney_keV, EFF) for x_test in keVnr[keVnr<=max(effieicney_keV)]]
        
        
        if eff_type in ['LZ']:#'Xe1t', 'Xepast', 'PANDAS_tot','PANDAS_ROI', 
            efficiency_100 = [int(~e) for e in keVnr>max(effieicney_keV)]
            Efficiency_fn = efficiency_Es +[i for i in efficiency_100 if i == 0]
        
        elif 'S1S2' in eff_type:
            
            efficiency_100 = [int(~e) for e in keVnr>max(effieicney_keV)]
            Efficiency_fn = efficiency_Es +[max(efficiency_Es) for i in efficiency_100 if i == 0]
        elif 'Argon' in eff_type or 'ER' in eff_type:
            efficiency_Es= np.array(efficiency_Es)
            efficiency_Es[efficiency_Es<0]=0
            efficiency_100 = np.array([int(e) for e in keVnr>max(effieicney_keV)])
            Efficiency_fn = list(efficiency_Es) +list(efficiency_100[efficiency_100>0] * efficiency_Es[-1])
            
        else:
            efficiency_100 = [int(e) for e in keVnr>max(effieicney_keV)]
            Efficiency_fn = efficiency_Es +[i for i in efficiency_100 if i != 0]
     
        if len(keVnr) != len(Efficiency_fn):
            print('wrong linear itp detector s1s2 efficiency function ')
            Efficiency_fn =  [0]*len(keVnr)
       
    return Efficiency_fn           
'''
def generate_accept(xMin, xMax, yMax, fitfn_num):
    rdm_xy = [xMin + ( xMax - xMin )*np.random.uniform() , yMax* np.random.uniform()]
    while(rdm_xy[1]>use_choise_fun(rdm_xy[0], fitfn_num)):
        rdm_xy = [xMin + ( xMax - xMin )*np.random.uniform() , yMax* np.random.uniform()]
    return rdm_xy



def linear_Itp_pdf(start, end, binnum, original_x, original_y):
      
    x_itp = np.logspace(start, end, binnum)*original_x.unit
    y_itp = [linear_intp(x_test, original_x.value, original_y.value) for x_test in x_itp[x_itp>=min(original_x)].value]
    print(y_itp[0])
    if y_itp[0] <0:
        y_itp[0] = 0
    y_end = np.ones(x_itp[x_itp<min(original_x)].shape)*y_itp[0]
    print(y_end)
    ys = list(y_end)+list(y_itp)
    return x_itp, ys*original_y.unit


def add_smearing(Er, pdf, nuclei = 'Xenon', sigma_percentage = 0.07, plot = False):
    Er_keV = Er.to(u.keV)
    smear_pdfs = []
  
    for i, (Er_position, dEr_position) in enumerate(zip(Er_keV, np.diff(Er_keV))):
        if nuclei == 'Xenon':
            sigma = get_smear_sigma(Er_position)
        elif nuclei == 'Argon':
            sigma = get_smear_sigma_Argon(Er_position, sigma_percentage)
        kernel = np.exp(-(Er_keV - Er_position) ** 2 / (2 * sigma ** 2)) / np.sqrt(2*np.pi)/sigma
       
        smear_pdfs.append(sum(pdf * kernel * dEr_position))
    smear_pdfs = qua2arr(smear_pdfs)
    if plot:
        fig, ax = plt.subplots()
        ax.loglog(Er_keV, pdf, label = '')
        ax.loglog(Er_keV[:-1], smear_pdfs, label = 'smear')
    return Er_keV[:-1], smear_pdfs


def get_official_pcle(pcle):
    if pcle == 'Be7_861':
        return r'$^{7}$Be 861keV'
    elif pcle == 'Be7_384':
        return r'$^{7}$Be 384keV'
    elif pcle == 'Be7':
        return r'$^{7}$Be'
    elif pcle == 'N13':
        return r'$^{13}$N'
    elif pcle == 'O15':
        return r'$^{15}$O'
    elif pcle == 'F17':
        return r'$^{17}$F'
    elif pcle == 'nubb':
        return r'2$\nu\beta\beta$'
    elif pcle == 'atmNu_SURF_avg':
        return 'Atm SURF'
    elif pcle == 'Kr85':
        return r'$^{85}$Kr'
    elif pcle == 'Rn222':
        return r'$^{222}$Rn'
    elif pcle == '8B':
        return r'$^{8}$B'
    else:
        return pcle
    
    
def get_solar_components(pcle):
    if pcle in ['pp' ,'Be7_861', 'Be7_384', 'N13','O15' ,'F17', 'pep', 'CNO', '8B', 'hep']:
        return True
    else:
        return False

def gaussian(x, x0, sigma):
    return np.exp(-0.5 * (x-x0)**2 / sigma**2) / np.sqrt(2*np.pi) / sigma

def get_smear_sigma(Er):
    #https://arxiv.org/pdf/1807.07169.pdf
    Er = Er.to(u.keV)
    return (0.31 * np.sqrt(Er/u.keV) + 0.0035 * Er/u.keV) * u.keV

def get_smear_sigma_Argon(Er, sigma_percentage):
    Er = Er.to(u.keV)
    return sigma_percentage * Er