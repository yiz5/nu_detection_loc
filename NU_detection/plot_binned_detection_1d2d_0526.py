import numpy as np
import os

import random
import pandas as pd
import math
import astropy.units as u
import astropy.constants as const
import scipy.stats
import itertools
from scipy.stats import poisson
from scipy import interpolate
import sys
from s1s2_0311 import *
from read import *
from recoil_1223 import *

unit_pdf = (u.tonne*u.yr*u.keV)**(-1)
unit_Er = u.keV


def check_consecutive(l):
    return sorted(l)==list(range(min(l), max(l)+1))

def find_consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def get_grid_index(dataf, start_index, sum_thrd, plot=True, plot_contour = False, print_check = True, plot_maxDFsize = 150):
   
    startx, starty = start_index
    
    previous_index = [start_index]
    summing_test_df = dataf.iat[start_index]
    step = 0
    max_xtest, max_ytest = dataf.shape
    if np.prod(dataf.shape)<=plot_maxDFsize and plot == True:
        
        fig, ax = plt.subplots(figsize = (6,6))
        ax = setup_cdfpdf_ax(ax,'test', 'bin_index_x', 'bin_index_y', True, '', 10, 10, 
                vlines = [0, max_xtest-1], hlines = [0, max_ytest-1], 
                     xlims = [-1,max_xtest], ylims = [-1,max_ytest], log = [0,0])
        ax.scatter([startx], [starty], color ='black')
        if plot_contour:
            cp =ax.contour(dataf.to_numpy().T)
            ax.clabel(cp, inline=True, fontsize=12)
        ax.set_aspect('equal')
    while summing_test_df < sum_thrd:
        step+=1
        #print(step)
        lists1 = list(np.arange(startx-step, startx+step+1, 1))
        lists2 = list(np.arange(starty-step, starty+step+1, 1))
        #print(lists1, lists2)

        all_index = list(itertools.product(*[lists1, lists2]))
        valid_all_index, _, _ = filter_outDF(dataf, all_index)

        new_index = list(set(all_index) ^ set(previous_index) )
        valid_new_index, valid_cell_values, invalid_new_index = filter_outDF(dataf, new_index)
        
        if print_check:
            print('all_index = ', all_index , 'len = ', len(all_index), 
              '\nvalid_all_index = ', valid_all_index, 'len = ', len(valid_all_index), 
              '\nprevious_index = ', previous_index,'len = ', len(previous_index), 
              '\nnew_index = ', new_index,'len = ', len(new_index), 
              '\nvalid_new_index = ', valid_new_index, 'len = ', len(valid_new_index)
        )


        previous_index = valid_all_index



        #print('len(new_index) = ', len(new_index), '\n',new_index)
        summing_test_df += sum(valid_cell_values)
        print('start at', start_index, summing_test_df, '\n')
        if np.prod(dataf.shape)<=plot_maxDFsize and plot == True:

            ax.scatter(np.array(valid_new_index).T[0], np.array(valid_new_index).T[1], s = 15, 
                       color =tuple(np.random.rand(3)), alpha = 0.8)
        if len(valid_all_index) == np.prod(dataf.shape):
            print('finished scanning all ', np.prod(dataf.shape), ' cells')
            break
    if np.prod(dataf.shape)<=plot_maxDFsize and plot == True:        
        ax.axvline(x = min(np.array(valid_all_index).T[0]))
        ax.axvline(x = max(np.array(valid_all_index).T[0]))
        ax.axhline(y = min(np.array(valid_all_index).T[1]))
        ax.axhline(y = max(np.array(valid_all_index).T[1]))
    return summing_test_df, (min(np.array(valid_all_index).T[0]), max(np.array(valid_all_index).T[0])),(min(np.array(valid_all_index).T[1]), max(np.array(valid_all_index).T[1]))


def filter_outDF(dataf, index):
    valid_index, valid_cell_values, invalid_index = [], [], []
    for idc in index:
        if all(i >=0 for i in idc):
            try:
                a = dataf.iat[idc]
                #print(idc, dataf.iat[idc])
                valid_index.append(idc)
                valid_cell_values.append(dataf.iat[idc])
                
            except BaseException:
                invalid_index.append(idc)
                #print(0)
                continue 
        else:
            continue
    return valid_index, valid_cell_values, invalid_index


def check_equality(x1, x2):
    if round(x1/x2, 10)==1:
        return True
    else:
        return False
    
    
def get_1dNR(det_pcle, pcles, binned_data_pd, metallicity, E_threshold_keV, nuclei, eff_type, 
            detector = '', corr = '', atm = False, print_check = False):
    
    if atm:
        alter_pcles = [n for n in pcles if 'atm' not in n] + [n for n in pcles if 'atm' in n and n in det_pcle] 
        null_pcles = [n for n in pcles if 'atm' not in n]
    else:
        alter_pcles = [n for n in pcles if 'atm' not in n] + [n for n in pcles if 'atm' in n and detector in n] 
        null_pcles = [n for n in alter_pcles if n not in det_pcle]

    if print_check:
        print('null pcles: ',null_pcles, 
             '\nalter_pcles and observed pcles: ', alter_pcles,
             '\ndet_pcle: ', det_pcle)
    if len(corr) > 0:
        select_null = [col for col in binned_data_pd if any([null_pcle in col for null_pcle in null_pcles]) and
             '_'+eff_type+'_' in col and 'thrd'+str(E_threshold_keV)+'keV' in col
                       and metallicity in col and nuclei in col and corr in col]

        select_alter = [col for col in binned_data_pd if any([alter_pcle in col for alter_pcle in alter_pcles]) and
             '_'+eff_type+'_' in col and 'thrd'+str(E_threshold_keV)+'keV' in col
                        and metallicity in col and nuclei in col and corr in col]
        
        select_det = [col for col in binned_data_pd if any([det in col for det in det_pcle]) and
             '_'+eff_type+'_' in col and 'thrd'+str(E_threshold_keV)+'keV' in col
                        and metallicity in col and nuclei in col and corr in col]
    else:
        select_null = [col for col in binned_data_pd if any([null_pcle in col for null_pcle in null_pcles]) and
             '_'+eff_type+'_' in col and 'thrd'+str(E_threshold_keV)+'keV' in col
                       and metallicity in col and nuclei in col and 'smear' not in col]

        select_alter = [col for col in binned_data_pd if any([alter_pcle in col for alter_pcle in alter_pcles]) and
             '_'+eff_type+'_' in col and 'thrd'+str(E_threshold_keV)+'keV' in col
                        and metallicity in col and nuclei in col and 'smear' not in col]
        
        select_det = [col for col in binned_data_pd if any([det in col for det in det_pcle]) and
             '_'+eff_type+'_' in col and 'thrd'+str(E_threshold_keV)+'keV' in col
                        and metallicity in col and nuclei in col and 'smear' not in col]
        
    if print_check:
        print('select_null = ',select_null,'\nselect_alter = ',select_alter,'\n')
    
    if len(select_null)!=len(null_pcles) or len(select_alter)!=len(alter_pcles):

        print('wrong selection')
        print(select_alter,select_null)
        return 0,0
    exposure_det_pcle = [col for col in binned_data_pd if any([det in col for det in det_pcle]) and
         '_'+eff_type+'_' in col and 'thrd'+str(E_threshold_keV)+'keV' in col
                         and metallicity in col and nuclei in col and corr in col]

    if len(exposure_det_pcle)!=len(det_pcle):
        print(exposure_det_pcle)
        print('wrong det pcle selection')
        return 0,0
    det_pcle_exposure = round(np.sum(np.sum(binned_data_pd[exposure_det_pcle], axis = 0)), 5)*unit_pdf*unit_Er
    if print_check:
        print(exposure_det_pcle, det_pcle_exposure)
    null_df = binned_data_pd[select_null]#*unit_pdf*unit_Er
    alter_df = binned_data_pd[select_alter]#*unit_pdf*unit_Er
    det_df = binned_data_pd[select_det]
    
    null_binned = np.array(np.sum(null_df, axis = 1))
    alter_binned = np.array(np.sum(alter_df, axis = 1))
    det_binned = np.array(np.sum(det_df, axis = 1))
    if len(null_binned)!=len(null_df) or len(alter_binned)!=len(alter_df):
        print('wrong Er_bin number after summation')
        return 0,0
    #qs = get_qs(exposures , alter_binned, null_binned)
    return null_binned, alter_binned, det_binned, null_pcles, alter_pcles, det_pcle_exposure#qs, det_pcle_exposure



def get_2dNR(det_pcle, pcles, binned_datas_dict, pcle_dict, metallicity, nubbscale, 
            detector = '', atm = False, print_check = False):
    if atm:
        alter_pcles = [n for n in pcles if 'atm' not in n] + [n for n in pcles if 'atm' in n and n in det_pcle] 
        null_pcles = [n for n in pcles if 'atm' not in n]
        
    else:
        alter_pcles = [n for n in pcles if 'atm' not in n] + [n for n in pcles if 'atm' in n and detector in n] 
        null_pcles = [n for n in alter_pcles if n not in det_pcle]

    
    if print_check:
        print('null pcles: ',null_pcles, 
             '\nalter_pcles and observed pcles: ', alter_pcles,
             '\ndet_pcle: ', det_pcle)
    null_binned = get_binned_data(null_pcles, nubbscale, binned_datas_dict, pcle_dict, 
                                  metallicity = metallicity, print_check = False)
    alter_binned = get_binned_data(alter_pcles, nubbscale, binned_datas_dict, pcle_dict, 
                                   metallicity = metallicity, print_check = False)
    det_binned = get_binned_data(det_pcle, nubbscale, binned_datas_dict, pcle_dict, 
                                   metallicity = metallicity, print_check = False)

    
    return null_binned, alter_binned, det_binned, null_pcles, alter_pcles


def get_1dER(det_pcle, pcles, binned_data_pd, metallicity, nuclei, nubb_fraction, eff_type = '', corr = '', 
               print_check = True):

    alter_pcles = pcles
    null_pcles = [n for n in alter_pcles if n not in det_pcle]
    
    if print_check:
        print('nubb: ', nubb_fraction, nuclei,'null pcles: ',null_pcles, 
             '\nalter_pcles and observed pcles: ', alter_pcles,'\n')

    if corr == '':
        select_null = [col for col in binned_data_pd if any([null_pcle in col for null_pcle in null_pcles]) and
             '_'+eff_type+'_' in col and metallicity in col and nuclei in col and 'smear' not in col]

        select_alter = [col for col in binned_data_pd if any([alter_pcle in col for alter_pcle in alter_pcles]) and
             '_'+eff_type+'_' in col and metallicity in col and nuclei in col and 'smear' not  in col]
        
        select_det = [col for col in binned_data_pd if any([det in col for det in det_pcle]) and
             '_'+eff_type+'_' in col and metallicity in col and nuclei in col and 'smear' not  in col]
        
    else:
        select_null = [col for col in binned_data_pd if any([null_pcle in col for null_pcle in null_pcles]) and
                 '_'+eff_type+'_' in col and metallicity in col and nuclei in col and corr in col]

        select_alter = [col for col in binned_data_pd if any([alter_pcle in col for alter_pcle in alter_pcles]) and
                 '_'+eff_type+'_' in col and metallicity in col and nuclei in col and corr in col]
        
        select_det = [col for col in binned_data_pd if any([det in col for det in det_pcle]) and
                 '_'+eff_type+'_' in col and metallicity in col and nuclei in col and corr in col]
        
    if print_check:
        print('select_alter:',select_alter, '\nselect_null:',select_null)

    if len(select_null)!=len(null_pcles) or len(select_alter)!=len(alter_pcles):
        print('wrong selection')
        return 0,0 
    if corr == '':
        exposure_det_pcle = [col for col in binned_data_pd if any([det in col for det in det_pcle]) and
             '_'+eff_type+'_' in col and metallicity in col and nuclei in col and 'smear' not in col]
    else:
        exposure_det_pcle = [col for col in binned_data_pd if any([det in col for det in det_pcle]) and
             '_'+eff_type+'_' in col and metallicity in col and nuclei in col and corr in col]
    if len(exposure_det_pcle)!=len(det_pcle):
        print('wrong det pcle selection')
        return 0,0 
    det_pcle_exposure = round(np.sum(np.sum(binned_data_pd[exposure_det_pcle], axis = 0)), 5)*unit_pdf*unit_Er
    null_df = binned_data_pd[select_null]#*unit_pdf*unit_Er
    alter_df = binned_data_pd[select_alter]#*unit_pdf*unit_Er
    det_df = binned_data_pd[select_det]
    
    if nuclei == 'Xenon':
        nubb_col = [col for col in null_df if 'nubb' in col]
        null_df[nubb_col] = null_df[nubb_col]*nubb_fraction
        
        alter_col = [col for col in alter_df if 'nubb' in col]
        alter_df[nubb_col] = alter_df[nubb_col]*nubb_fraction


    null_binned = np.array(np.sum(null_df, axis = 1))
    alter_binned = np.array(np.sum(alter_df, axis = 1))
    det_binned = np.array(np.sum(det_df, axis = 1))
    
    if len(null_binned)!=len(null_df) or len(alter_binned)!=len(alter_df):
        print('wrong Er_bin number after summation')
        return 0,0 

    return null_binned, alter_binned, det_binned, det_pcle_exposure#qs, det_pcle_exposure

def get_2dER(det_pcle, pcles, binned_datas_dict, pcle_dict, metallicity, nubb_fraction, print_check = False):
    alter_pcles = pcles
    null_pcles = [n for n in alter_pcles if n not in det_pcle]
    
    if print_check:
        print('null pcles: ',null_pcles, 
             '\nalter_pcles and observed pcles: ', alter_pcles,'\n')
    null_binned = get_binned_data(null_pcles, nubb_fraction, binned_datas_dict, pcle_dict, 
                                  metallicity = metallicity, print_check = print_check)
    alter_binned = get_binned_data(alter_pcles, nubb_fraction, binned_datas_dict, pcle_dict, 
                                   metallicity = metallicity, print_check = print_check)
    detect_binned = get_binned_data(det_pcle, nubb_fraction, binned_datas_dict, pcle_dict, 
                                   metallicity = metallicity, print_check = print_check)

    return null_binned, alter_binned, detect_binned#, qs



def get_all_components_ER(ER_pcles, nuclei, bg_ideal = ''):
    if nuclei == 'Argon':
        if bg_ideal == 'ideal':
            pcles= [p for p in ER_pcles if p not in ['nubb', 'Kr85', 'Rn222']]
        else:
            pcles= [p for p in ER_pcles if p not in ['nubb', 'Kr85']]
        E_end = 3000
    elif nuclei == 'Xenon':
        print(bg_ideal)
        if bg_ideal == 'ideal':
            pcles= [p for p in ER_pcles if p not in ['nubb', 'Kr85', 'Rn222']] 
        elif bg_ideal == 'nubb_only':
            pcles= [p for p in ER_pcles if p not in ['Kr85', 'Rn222']] 
        else:
            pcles= ER_pcles
        E_end = 2500
    return pcles,E_end


def get_all_components_NR(nuclei):
    if nuclei == 'Argon':
        E_end = 500
        eff_types = ['', 'Argon_tot']
    elif nuclei == 'Xenon':
        E_end = 100
        eff_types = ['','Xe100_5_eff_eh_S1S2']
        

    return E_end, eff_types


def get_1d_pd_ER(binned_data_pd, ER_pcles, eff_types, nuclei, metallicities, bg_ideal = '', binnum = 20, 
              folder_1d = 'unbinned_Er_pdf_itp', size = 100000, corr ='_smear', corr_sim = '_smear', 
                 E_threshold_keV = 1, unit_pdf = (u.tonne*u.yr*u.keV)**(-1), unit_Er = u.keV, log = True, 
                 print_check = True, plot = False):
    
    if corr_sim == '_smear0.013' and eff_types == ['ER_Ar'] and nuclei == 'Argon':
        ER_pcles = [p for p in ER_pcles if p not in ['pp', 'Be7_384']] 
    pcles,E_end = get_all_components_ER(ER_pcles, nuclei, bg_ideal = bg_ideal)
    print(nuclei, pcles)
    bin_ = np.logspace(np.log10(E_threshold_keV), np.log10(E_end), binnum)*unit_Er
    bin_centers = (bin_[:-1]+bin_[1:])/2
    binned_data_pd[nuclei+'_Er_bin ['+str(unit_Er)+']'] = [str(bin_[i].value)+'_'+str(bin_[i+1].value) for i in range(0, len(bin_)-1)]
    binned_data_pd[nuclei+'_Er_bincenters ['+str(unit_Er)+']'] = bin_centers.value

    for metallicity in metallicities:
        for eff_type in eff_types:
            binned_data_pd = binning_1dhist(binned_data_pd, folder_1d, nuclei, pcles, size, bin_, eff_type, 
                                    metallicity, ebinding = True, corr = corr, corr_sim =corr_sim, 
                   unit_pdf = (u.tonne*u.yr*u.keV)**(-1), unit_Er = u.keV, E_threshold_keV = 1, 
                   plot= plot, log = log, print_check = print_check)
    return binned_data_pd


def get_1d_pd_NR(binned_data_pd, NR_pcles, nuclei, metallicities, eff_types = [''], E_end = 0, 
                 binnum = 20, corr ='_smear', corr_sim = '', 
              folder_1d = 'unbinned_Er_pdf_itp', size = 100000, 
                 E_threshold_keV = 1, unit_pdf = (u.tonne*u.yr*u.keV)**(-1), unit_Er = u.keV, print_check = True):
    if E_end == 0:
        E_end,_ = get_all_components_NR(nuclei)
        
    print('E_end = ',E_end)
    bin_ = np.logspace(np.log10(E_threshold_keV), np.log10(E_end), binnum)*unit_Er
    bin_centers = (bin_[:-1]+bin_[1:])/2
    binned_data_pd[nuclei+'_Er_bin ['+str(unit_Er)+']'] = [str(bin_[i].value)+'_'+str(bin_[i+1].value) for i in range(0, len(bin_)-1)]
    binned_data_pd[nuclei+'_Er_bincenters ['+str(unit_Er)+']'] = bin_centers.value

    for metallicity in metallicities:
        for eff_type in eff_types:
            binned_data_pd = binning_1dhist(binned_data_pd, folder_1d, nuclei, NR_pcles, size, bin_, eff_type, 
                                    metallicity, ebinding = True, corr = corr, corr_sim =corr_sim, 
                   unit_pdf = (u.tonne*u.yr*u.keV)**(-1), unit_Er = u.keV, E_threshold_keV = E_threshold_keV, 
                   plot= False, log = True, print_check = print_check)
    
    return binned_data_pd

def get_pcle_offname(pcle):
    if pcle == 'atmNu':
        return 'atm'
    else:
        return pcle

def get_analytic_calDcalT(null_binned_df, det_binned_df, Phi_minus1, Z_alpha):
    sum_keppa_ratio = null_binned_df/null_binned_df
    sum_keppa_ratio = sum_keppa_ratio.fillna(1.)
    #print(sum_keppa_ratio )
    term1_numerator = sum((
        null_binned_df * (np.log(sum_keppa_ratio+det_binned_df))**2
                          ).to_numpy().flatten()
                         )
    term1_denominator = sum((
        (null_binned_df+det_binned_df)*(np.log(sum_keppa_ratio+det_binned_df))**2
                            ).to_numpy().flatten()
                           )
    #print(term1_numerator, term1_denominator)
    term1 = (Phi_minus1 - Z_alpha * np.sqrt(term1_numerator/term1_denominator)
            )**2
    term2 = sum((
        (null_binned_df+det_binned_df)*(np.log(sum_keppa_ratio+det_binned_df))**2
                ).to_numpy().flatten()
               )
    term3 = 1/ (sum((
        det_binned_df * np.log(sum_keppa_ratio+det_binned_df)
                    ).to_numpy().flatten()
                   )
               )**2
    
    #print(term1, term2, term3)
    calDcalT = term1*term2*term3
    return calDcalT


def get_analytic_calDcalT_shape(null_binned_df, det_binned_df, Phi_minus1, Z_alpha, print_check = False):
    sum_keppa_ratio = null_binned_df/null_binned_df
    #print(sum_keppa_ratio )
    term1_numerator_binned = (
        null_binned_df * (np.log(sum_keppa_ratio+det_binned_df))**2
                          ).to_numpy().flatten()
    term1_denominator_binned = (
        (null_binned_df+det_binned_df)*(np.log(sum_keppa_ratio+det_binned_df))**2
                            ).to_numpy().flatten()
    if print_check:
        print(term1_denominator_binned.shape, term1_numerator_binned.shape)
    term1_numerator_binned[np.isnan(term1_numerator_binned)] = 0
    term1_denominator_binned[np.isnan(term1_denominator_binned)] = 0
    
    term1_numerator = sum(term1_numerator_binned)
    term1_denominator = sum(term1_denominator_binned)
    
    term1 = (Phi_minus1 - Z_alpha * np.sqrt(term1_numerator/term1_denominator)
            )**2
    
    term2_binned = (
        (null_binned_df+det_binned_df)*(np.log(sum_keppa_ratio+det_binned_df))**2
                ).to_numpy().flatten()
    
    if print_check:
        print(term2_binned.shape)
    term2_binned[np.isnan(term2_binned)] = 0
    term2 = sum(term2_binned)
    
    term3_binned_denominator = (
        det_binned_df * np.log(sum_keppa_ratio+det_binned_df)
                    ).to_numpy().flatten()
    term3_binned_denominator[np.isnan(term3_binned_denominator)] = 0
    
    
    term3 = 1/ (sum(term3_binned_denominator)**2)
    if print_check:
        print(term1, term2, term3)
    calDcalT = term1*term2*term3
    return calDcalT

def get_line_color(pcle, space, nuclei):
    
    if pcle == 'CNO':
        if nuclei == 'Xenon':
            if 'all' in space :
                c ='red'
                s = pcle + ' all'
            elif 'ideal' in space:
                c= 'lightpink'
                s = pcle + ' '+r'Xe E$_{r}$ ideal'
            else:
                c=  'lightcoral'
                s = pcle + ' '+r'2$\nu\beta\beta$'
        elif nuclei == 'Argon':
            if 'ideal' in space:
                c= 'lightpink'
                s = pcle + ' '+r'Ar E$_{r}$ ideal'
            else:
                c= 'lightpink'
                s = pcle + ' '+r'Ar E$_{r}$ $^{222}$Rn, 10% $\sigma$'
    elif pcle == 'pep':
        if nuclei == 'Xenon':
            if 'all' in space :
                c ='blue'  
                s = pcle + ' all'
            elif 'ideal' in space:
                c= 'cyan'
                s = pcle + ' '+r'Xe E$_{r}$ ideal'
            else:
                c= 'turquoise'
                s = pcle + ' '+r'2$\nu\beta\beta$'
        elif nuclei == 'Argon':
            
            if 'ideal' in space:
                c= 'cyan'
                s = pcle + ' '+r'Ar E$_{r}$ ideal'
            else:
                c='cyan'
                s = pcle + ' '+r'Ar E$_{r}$ 10% $\sigma$+$\epsilon(E_{r}$)'
    
    elif pcle == 'hep':
        if nuclei == 'Xenon':
            if 'S1/S2' in space :
                c ='purple'  
                s = 'Xe S1/S2 all'
            elif 'ideal' in space:
                c= 'plum'
                s = r'Xe E$_{r}$ ideal'
        elif nuclei == 'Argon':
            
            if 'ideal' in space:
                c= 'pink'
                s = r'Ar E$_{r}$ ideal'
            else:
                c='deeppink'
                s = r'Ar E$_{r}$ 10% $\sigma$+$\epsilon(E_{r}$)'
    
    elif pcle == 'dsnb':
        if nuclei == 'Xenon':
            if 'S1/S2' in space :
                c ='midnightblue'  
                s = 'Xe S1/S2 all'
            elif 'ideal' in space:
                c= 'lightsteelblue'
                s = r'Xe E$_{r}$ ideal'
        elif nuclei == 'Argon':
            if 'ideal' in space:
                c= 'paleturquoise'
                s = r'Ar E$_{r}$ ideal'
            else:
                c='darkcyan'
                s = r'Ar E$_{r}$ 10% $\sigma$+$\epsilon(E_{r}$)'
                
    elif 'atmNu' in pcle:
        if nuclei == 'Xenon':
            if 'S1/S2' in space :
                c ='darkolivegreen'  
                s = 'Xe S1/S2 all'
            elif 'ideal' in space:
                c='darkkhaki'  
                s = r'Xe E$_{r}$ ideal'
         
        elif nuclei == 'Argon':
            
            if 'ideal' in space:
                c='gold' 
                s = r'Ar E$_{r}$ ideal'
            else:
                c='darkorange'
                s = r'Ar E$_{r}$ 10% $\sigma$+$\epsilon(E_{r}$)'
    else:
        c = 'black'
        s = 'else'
    return c, s

def get_thinNEST_effnum(file):
    with open(file, "r") as f:
        for line in f:
            if 'effective exposure' in line:
                line_split = line.split()
                eff = float(line_split[-1])
                print(eff)
                break
            
    return eff

'''
def find_beta(mean_null, alpha, mean_alter, simnum = 500000, binnum = 50, plot = False):
    if binnum == 0:
        counts_null, bins_null = np.histogram(np.random.poisson(mean_null, simnum), density = True)
        counts_alter, bins_alter = np.histogram(np.random.poisson(mean_alter, simnum),  density = True)
    else:
        counts_null, bins_null = np.histogram(np.random.poisson(mean_null, simnum), binnum,  density = True)
        counts_alter, bins_alter = np.histogram(np.random.poisson(mean_alter, simnum), binnum,  density = True)
        
    bincenters_null = (bins_null[1:]+bins_null[:-1])/2
    cdf_null = np.cumsum(counts_null*np.diff(bins_null))
    
    N_alpha = linear_intp((1-alpha), cdf_null, bincenters_null)
    
    
    bincenters_alter = (bins_alter[1:]+bins_alter[:-1])/2
    cdf_alter = np.cumsum(counts_alter*np.diff(bins_alter))

    beta = linear_intp(N_alpha, bincenters_alter, cdf_alter)
    if plot:
        fig, ax = plt.subplots()
        ax.bar(bincenters_null, counts_null, np.diff(bins_null), color = 'red', label = 'null')
        ax.plot(bincenters_null, cdf_null, color = 'red')
        
        ax.bar(bincenters_alter, counts_alter, np.diff(bins_alter), color = 'blue', label = 'alter')
        ax.plot(bincenters_alter, cdf_alter, color = 'blue')
        
        ax.axhline(y = 1-alpha)
        ax.axhline(y = beta)
        ax.axvline(x = N_alpha)
        ax.legend()
    return N_alpha, beta

'''
def get_poisrdm_Rate(pcles, bins_groups,N = 500000, 
                     nuclei_2d = 'Xenon', footnote = 'thrd', E_threshold_keV = 1, metallicity = 'high',
                     folder_2d = 'nestpy_unbinned_unbound',
                    plot = True, print_check = True, plot_xlims = [0, 150], plot_ylims = [2, 5]):
    tot_binnums = 0
    for bins_info in bins_groups:
        _, _, _, _, binnumberx, binnumbery = bins_info
        tot_binnums +=binnumberx*binnumbery
    if print_check:
        print('tot_binnums = ', tot_binnums)
    total_rates_region_allpcles = np.zeros(tot_binnums)
    for pcle in pcles:
        file_name = os.path.join(folder_2d, 'S1S2_'+pcle+'.csv')
        print(file_name, os.path.exists(file_name))

        dataf = pd.read_csv(file_name)

        valid_dataf = dataf.loc[(dataf['cS1[phd]'] > 0 ) & (dataf['cS2[phd]']>0) ]

        s1 = valid_dataf['cS1[phd]']
        log10s2 = np.log10(valid_dataf['cS2[phd]'])
        eff_num =len(dataf)


        eventR, _, _, _, _, _ = read_pcle_cdf(pcle, nuclei_2d, E_threshold_keV, '', recoil_type = '',
                      read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, 
                                              metallicity = metallicity, ebind = True, plot_pdf = False, corr = '')
        DT_longtime = eff_num/eventR
        print(pcle, 'E_threshold_keV = ', E_threshold_keV, eventR)
        counts_all,xbins_all, ybins_all= np.histogram2d(s1, log10s2, bins =[100, 100])
        if print_check:
            print('sim number = ', eff_num, 
                  '\nvalid_dataf = ', len(valid_dataf))
        
        if plot:
            fig, ax= plt.subplots(figsize = (6,6))
            ax = setup_cdfpdf_ax(ax,pcle, 'cs1', 'log10(cs2)', True, '', 20, 20, 
                        vlines = [0,0], hlines = [0,0], 
                             xlims = plot_xlims, ylims = plot_ylims, log = [0,0])
            cp =ax.contour((xbins_all[1:]+xbins_all[:-1])/2, 
                           (ybins_all[1:]+ybins_all[:-1])/2, 
                           (counts_all/DT_longtime).T)
            ax.clabel(cp, inline=True, fontsize=12)
            
        total_rates_region = []
        for bins_info in bins_groups:
            binS1_min, binS1_max, binlog10S2_min, binlog10S2_max, binnumberx, binnumbery = bins_info
            counts,xbins, ybins = np.histogram2d(s1, log10s2, bins =[binnumberx, binnumbery]
                                       ,range = [[binS1_min, binS1_max], [binlog10S2_min, binlog10S2_max]]
                                       )
            rates = counts/DT_longtime
            rate_unit = rates.unit
            total_rates_region.extend(rates.flatten())
            
            if print_check:
                print('number within binrange region = ', int(sum(sum(counts))),
                  '\nxbins = ', xbins, 
                  '\nybins = ', ybins,
                     '\nrates = ', rates.flatten(),
                     '\nfraction of rate within binrange = ', sum(rates.flatten())/eventR)
            if plot:
           
                ax.plot([min(xbins), min(xbins)], [min(ybins), max(ybins)], color ='black')
                ax.plot([max(xbins), max(xbins)], [min(ybins), max(ybins)], color ='black')
                ax.plot([min(xbins), max(xbins)], [min(ybins), min(ybins)], color ='black')
                ax.plot([min(xbins), max(xbins)], [max(ybins), max(ybins)], color ='black')

                for i in range(0, len(xbins)-1):
                    for j in range(0, len(ybins)-1):
                        ax.plot([xbins[i], xbins[i+1]], [ybins[j], ybins[j]],  color ='black', ls = '--')

                for j in range(0, len(ybins)-1):
                    for i in range(0, len(xbins)-1):
                        ax.plot([xbins[i], xbins[i]], [ybins[j], ybins[j+1]],  color ='black', ls = '--')
        total_rates_region_allpcles+=np.array(qua2arr(total_rates_region).value)
        
    
        if print_check:
    
            print('fraction of rate within tot bin range = ', sum(total_rates_region)/eventR,
                 '\nrate within tot bin range = ', sum(total_rates_region)
                 )

        
    
    mean_total_rates_region_allpcles = np.zeros(len(total_rates_region_allpcles))
    for r, rate_bin in enumerate(total_rates_region_allpcles):
        mean_total_rates_region_allpcles[r] = np.average(np.random.poisson(rate_bin, N))
    mean_total_rates_region_allpcles*=rate_unit
    return mean_total_rates_region_allpcles, total_rates_region_allpcles*rate_unit

'''

def get_DT_greaterthan_G_method2(exposures, null_sum_ijkappa, alter_sum_ijkappa, alpha, beta, 
                     simnum = 500000, binnum = 50, 
                     plot= False, print_check = False):
    
    Ns_alpha,powers, Gs = [],[],[]
    for exp, exposure in enumerate(exposures):

        E_N_null = null_sum_ijkappa * exposure
        E_N_alter = alter_sum_ijkappa * exposure

        sim_N_null = np.random.poisson(E_N_null, simnum)
        sim_N_alter = np.random.poisson(E_N_alter, simnum)
        
        
        N_alpha_exp = np.percentile(sim_N_null,  (1-alpha)*100)
        power_exp = len(sim_N_alter[sim_N_alter>N_alpha_exp])/len(sim_N_alter)
        
        
        G = (N_alpha_exp - E_N_null) / np.sqrt(E_N_null)
        Ns_alpha.append(N_alpha_exp) 
        powers.append(power_exp)
        Gs.append(G)
        
        if print_check:
            
            
            print('sumnull2d = ', null_sum_ijkappa,
              'sumalter2d = ', alter_sum_ijkappa, '\n'
              'DT = ', exposure, 
              'E_N_null = ', E_N_null,
              'E_N_alter = ', E_N_alter, '\n', 
              'E_N_alter - E_N_null = ', E_N_alter-E_N_null,'\n',
              'N_alpha = ', N_alpha_exp, 
              'fraction of sim_Nnull > N_alpha :', len(sim_N_null[sim_N_null>N_alpha_exp])/len(sim_N_null), 
              'fraction of sim_Nalter > N_alpha :', power_exp, 
              'G = ', G, 
              '\n\n')
            
        if exp>0 and power_exp<0.1 and powers[exp-1] > 0.1 and power_exp>0:
            break
    DT = linear_intp(1-beta, powers, exposures)
    #print(DT)
    return DT, Ns_alpha, betas, Gs



def get_DT_alphabeta_G_method2(exposures, null_sum_ijkappa, alter_sum_ijkappa, alpha, beta, 
                     simnum = 500000, binnum = 50, 
                     plot= False, print_check = False):
    
    Ns_alpha,betas, Gs = np.zeros(len(exposures)), np.zeros(len(exposures)), np.zeros(len(exposures))
    for exp, exposure in enumerate(exposures):

        E_N_null = null_sum_ijkappa * exposure
        E_N_alter = alter_sum_ijkappa * exposure

        N_alpha_exp, beta_exp = find_beta(E_N_null, alpha, E_N_alter, 
                                              simnum = simnum, binnum = binnum, plot = plot)
        G = (N_alpha_exp - E_N_null) / np.sqrt(E_N_null)
        Ns_alpha[exp], betas[exp], Gs[exp] = N_alpha_exp, beta_exp, G
        if print_check:
            print('sumnull2d = ', null_sum_ijkappa,
              'sumalter2d = ', alter_sum_ijkappa, '\n'
              'DT = ', exposure, 
              'E_N_null = ', E_N_null,
              'E_N_alter = ', E_N_alter, '\n', 
              'E_N_alter - E_N_null = ', E_N_alter-E_N_null,'\n',
              'N_alpha = ', N_alpha_exp, ' 1-beta = ', beta_exp, 'G = ', G, 
              '\n\n')

    DT = linear_intp(1-beta, betas, exposures)
    #print(DT)
    return DT, Ns_alpha, betas, Gs


'''


def get_nume_DT_alphabeta_G_method2(exposures, null_sum_ijkappa, alter_sum_ijkappa, alpha, power, simnum = 500000, 
                     plot= False, print_check = False):
    
    Ns_alpha,powers, Gs, test_exposures = [],[],[],[]
    for exp, exposure in enumerate(exposures):

        E_N_null = null_sum_ijkappa * exposure
        E_N_alter = alter_sum_ijkappa * exposure
        sim_N_alter = np.random.poisson(E_N_alter, simnum)
        
        N_alpha_exp = poisson.ppf(round(1-alpha, 3), E_N_null)
        power_exp = len(sim_N_alter[sim_N_alter>N_alpha_exp])/len(sim_N_alter)
        
        G = (N_alpha_exp - E_N_null) / np.sqrt(E_N_null)
        Ns_alpha.append(N_alpha_exp)
        powers.append(power_exp)
        Gs.append(G)
        test_exposures.append(exposure)
        
        if print_check:
            print('sumnull2d = ', null_sum_ijkappa,
              'sumalter2d = ', alter_sum_ijkappa, '\n'
              'DT = ', exposure, 
              'E_N_null = ', E_N_null,
              'E_N_alter = ', E_N_alter, '\n', 
              'E_N_alter - E_N_null = ', E_N_alter-E_N_null,'\n',
              'alpha = ', alpha, ' N_alpha = ', N_alpha_exp, '\n',
               'out of ', len(sim_N_alter), 'E(N_alter) simulation, ', len(sim_N_alter[sim_N_alter>N_alpha_exp]), 
               'is greater than N_alpha\n', 
               ' power = ', power_exp, 'G = ', G, 
              '\n\n')
            
        if exp>0 and power_exp>power and powers[exp-1] < power and power_exp>0:
            break
    Ns_alpha = np.array(Ns_alpha)
    powers = np.array(powers)
    Gs = np.array(Gs)
    test_exposures = np.array(test_exposures)
    
    DT = np.interp(power,  powers ,test_exposures)
    if plot:
        fig, ax = plt.subplots()
        
        ax = setup_cdfpdf_ax(ax, '', '', '', False, '', 20, 20, 
                vlines = [0], hlines = [0], xlims = [0,0], ylims = [0,1], log = [0,0])
        ax.plot(test_exposures, powers) 
        ax.axhline(power)
        ax.axvline(DT)
    return DT, Ns_alpha, powers, Gs, test_exposures


def get_bin_regions(pcles, bins_info, binned_rates, 
                     nuclei_2d = 'Xenon', footnote = 'thrd', E_threshold_keV = 1, metallicity = 'high',
                     folder_2d = 'nestpy_unbinned_unbound', binnumber = [100, 100], 
                    print_check = True):
    
    xbins_input, ybins_input = bins_info
    print(xbins_input.shape, ybins_input.shape)
    for pcle in pcles:
        file_name = os.path.join(folder_2d, 'S1S2_'+pcle+'.csv')
        print(file_name, os.path.exists(file_name))

        dataf = pd.read_csv(file_name)

        valid_dataf = dataf.loc[(dataf['cS1[phd]'] > 0 ) & (dataf['cS2[phd]']>0) ]

        s1 = valid_dataf['cS1[phd]']
        log10s2 = np.log10(valid_dataf['cS2[phd]'])
        eff_num =len(dataf)


        eventR, _, _, _, _, _ = read_pcle_cdf(pcle, nuclei_2d, E_threshold_keV, '', recoil_type = '',
                      read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, 
                                              metallicity = metallicity, ebind = True, plot_pdf = False, corr = '')
        DT_longtime = eff_num/eventR
        print(pcle, eventR)
        if len(xbins_input)==1 and len(ybins_input)==1: 
            counts, xbins, ybins = np.histogram2d(s1, log10s2, bins = binnumber)
            print( binnumber[0], 'x bins within ', min(s1), max(s1), 
                   '\n',binnumber[1], 'y bins within ', min(log10s2), max(log10s2))
        elif len(xbins_input)>1 and len(ybins_input)>1:
            counts, xbins, ybins = np.histogram2d(s1, log10s2, bins = [xbins_input, ybins_input])
            print('number within binrange region = ', int(sum(sum(counts))),
              'fraction of counts within binrange region = ', int(sum(sum(counts)))/len(valid_dataf),
              'samebin? ', all(xbins == xbins_input), all(ybins == ybins_input)
             )
            print(counts.shape, xbins.shape, ybins.shape)
        r = counts/DT_longtime
        print(binned_rates.shape)
        binned_rates+=r
     
    if len(xbins_input)==1 and len(ybins_input)==1: 
        return binned_rates, xbins, ybins
    elif len(xbins_input)>1 and len(ybins_input)>1:
        return binned_rates
    
def get_null_NR(det_pcle, detector):
    if 'atm' in det_pcle:
        pcles_null = ['dsnb', 'hep', '8B', 'pp', 'Be7_384', 'Be7_861', 'CNO','pep',  'nubb', 'Kr85', 'Rn222']
        pcles_det = ['atmNu_'+detector+'_avg']
        xbins_all = np.linspace(1, 401, 101)
        ybins_all = np.linspace(2, 4.7, 101)
        return pcles_null, pcles_det, xbins_all, ybins_all
    elif 'dsnb' in det_pcle:
        pcles_null = ['atmNu_'+detector+'_avg', 'hep', '8B', 'pp', 'Be7_384', 'Be7_861', 'CNO','pep',  'nubb', 'Kr85', 'Rn222']
        pcles_null_woatm = ['hep', '8B', 'pp', 'Be7_384', 'Be7_861', 'CNO','pep',  'nubb', 'Kr85', 'Rn222']
        pcles_det = ['dsnb']
        xbins_all = np.linspace(1, 361, 91)
        ybins_all = np.linspace(1.9, 4.6, 81)
        return pcles_null, pcles_null_woatm, pcles_det, xbins_all, ybins_all
    elif 'hep' in det_pcle:
        pcles_null = ['atmNu_'+detector+'_avg', 'dsnb', '8B', 'pp', 'Be7_384', 'Be7_861', 'CNO','pep',  'nubb', 'Kr85', 'Rn222']
        pcles_null_wo8B = ['atmNu_'+detector+'_avg', 'dsnb', 'pp', 'Be7_384', 'Be7_861', 'CNO','pep',  'nubb', 'Kr85', 'Rn222']
        pcles_det = ['hep' ]
        xbins_all = np.linspace(1, 25, 11)
        ybins_all = np.linspace(1.9, 4, 11)
        return pcles_null, pcles_null_wo8B, pcles_det, xbins_all, ybins_all
    
    
def get_info_signalpcle(signal_pcle, title):
    if signal_pcle == 'atm':
        method = 'lessbgMethod2'
        Argon_xlim, Xenon_xlim = [10, 2000], [3, 200]
        Argon_xticks = [10, 50, 100, 1000]
        Xenon_xticks = [3,5, 10, 20, 30, 50, 100]
        Argon_ylim, Xenon_ylim = [1e-6, 0.8], [1e-6, 0.8]
    elif signal_pcle == 'hep':
        if title == 'ideal':
            method = 'no8BbgMethod2'
        else:
            method = 'allHEPMethod2'
        Argon_xlim, Xenon_xlim = [5, 30], [3, 10]
        Argon_xticks = [5, 8, 10, 15, 20, 30]
        Xenon_xticks = [3,4,5,6,8,10]
        Argon_ylim, Xenon_ylim = [1e-6, 0.8], [1e-6, 0.8]
    elif signal_pcle == 'dsnb':
        method = 'lessatmlesselseMethod2'
        Argon_xlim, Xenon_xlim = [10, 100], [3, 20]
        Argon_xticks = [10, 20, 30, 50, 100]
        Xenon_xticks = [3,4,5,6,8,10, 15, 20]
        Argon_ylim, Xenon_ylim = [1e-6, 9e-3], [1e-6, 9e-3]
    return method, Argon_xlim, Xenon_xlim, Argon_ylim, Xenon_ylim, Argon_xticks, Xenon_xticks


def get_official_pcle(pcle):
   
    if pcle == 'atmNu_CJPL_avg':
        return 'Atm CJPL'
    elif pcle == 'atmNu_Kamioka_avg':
        return 'Atm Kamioka'
    elif pcle == 'atmNu_LNGS_avg':
        return 'Atm LNGS'
    elif pcle == 'atmNu_SURF_avg':
        return 'Atm SURF'
    elif pcle == 'atmNu_SNOlab_avg':
        return 'Atm SNOlab'
    elif pcle == 'atm':
        return 'Atm'
    elif pcle == '8B':
        return r'$^{8}$B'
    elif pcle == 'dsnb':
        return 'DSNB'
    else:
        return pcle
    
    
    
def get_pois_q(null_ijkappa, alter_ijkappa, DT, runtime = 10000, Data_null = True, Data_alter = True):
    i = 0
    if Data_null:
        qs_simnull =  np.zeros(runtime)
    if Data_alter:
        qs_simalter  =  np.zeros(runtime)
        
    null_num_ijkappa = null_ijkappa * DT
    alter_num_ijkappa = alter_ijkappa * DT
    
    while i < runtime:
        
        #when data N is generated under the null hypothesis
        if Data_null:
            poisson_N_null_ijkappa = np.random.poisson(null_num_ijkappa)
            null_N_null_data = binned_likelihood_general(poisson_N_null_ijkappa, null_num_ijkappa)
            alter_N_null_data = binned_likelihood_general(poisson_N_null_ijkappa, alter_num_ijkappa)

            q_simnull = -2 * np.sum(null_N_null_data - alter_N_null_data) 
            qs_simnull[i] = q_simnull
        
        #when data N is generated under the alternative hypothesis
        if Data_alter:
            poisson_N_alter_ijkappa = np.random.poisson(alter_num_ijkappa)
            null_N_alter_data = binned_likelihood_general(poisson_N_alter_ijkappa, null_num_ijkappa)
            alter_N_alter_data = binned_likelihood_general(poisson_N_alter_ijkappa, alter_num_ijkappa)

            q_simalter = -2 * np.sum(null_N_alter_data - alter_N_alter_data) 
            qs_simalter[i] = q_simalter
        
        i+=1 
    
    if Data_null and not Data_alter:
        return qs_simnull
    elif Data_alter and not Data_null:
        return qs_simalter
    elif Data_alter and Data_null:
        return qs_simnull, qs_simalter
    
    
    
def simulate_wilks_thm(mean, expected, num_sample = 500, runtime = 10000):
    #https://stephens999.github.io/fiveMinuteStats/wilks.html
    qs = np.zeros(runtime)
    i=0
    while i < runtime:
        data_generated = np.random.poisson(mean, num_sample)
        null_data = binned_likelihood_general(data_generated.mean(), expected)
        alter_data = binned_likelihood_general(data_generated.mean(), data_generated.mean())
        qs[i] = -2 * num_sample * np.sum(null_data - alter_data) 
        i+=1
    return qs