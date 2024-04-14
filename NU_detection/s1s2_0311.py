import numpy as np
import os
import random
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import interp1d
from recoil_1223 import *
from read import * 
import time
path='./'



def get_data(folder, pcle, nuclei, numevent, footnote):
    files = glob.glob(os.path.join(folder,'S1S2*'+pcle+'*'+nuclei+'*'+str(numevent)+'*'+footnote+'*.txt'))
    
    if len(files) == 1:
        data = read_file_dataline(files[0]).T
        with open(files[0], "r") as f1:
            last_line = f1.readlines()[-1]
            effnum = int(last_line.split( )[-1])
    else:
        print('more than one file selected ')
        print(files)
        return 0,0
    
    return data, effnum


    
def get_data_component(data, data_path):
    if 'thinNEST_unbinned' in  data_path:
        E = data.T[0]
        s1phd = data.T[1]
        s2phd = data.T[2]
        rmm = data.T[3]
        zmm = data.T[4]
        E_true = data.T[-1]
        return E, s1phd, s2phd, rmm, zmm, E_true
    elif 'execNEST_results' in data_path:
        E = data.T[0]
        t_drift = data.T[2]
        z = data.T[5]
        
        s1_phd = data.T[-5]
        s2_phd = data.T[-1]
        
        return E, t_drift, z, s1_phd, s2_phd
    elif 'nestpy' in data_path:
        E = data.T[0]
        s1phd = data.T[1]
        s2phd = data.T[2]
        rmm = data.T[3]
        zmm = data.T[4]
        E_true = data.T[-1]
        
        return E, s1phd, s2phd, rmm, zmm, E_true
    



    
    
def logPoisson(exp, obs):#ln poisson per bin , ln n! cancel
    if exp == 0:
        return 0
    else:
        return -exp + obs * np.log(exp)  



def binned_likelihood_general(datObs, datExp):
 
    binned_log_likelihood = np.array([logPoisson(exp, obs) for exp, obs in zip(datExp.flatten(), datObs.flatten())])
    #print(datExp.shape, datObs.shape, datExp.flatten().shape, datObs.flatten().shape, len(binned_log_likelihood))
    return binned_log_likelihood

def binned_likelihood(datObs, datExp):
    binned_log_likelihood = ((-datExp + datObs * np.log(datExp)).replace(np.nan, 0))**((datExp!=0).astype(int))
    return binned_log_likelihood


def get_qs(exposures , alter_binned, null_binned):
    qs = []
    for exp in exposures:
    
        null = binned_likelihood_general(alter_binned * exp, null_binned * exp)
        alter = binned_likelihood_general(alter_binned * exp, alter_binned * exp)
        q = -2 * np.sum(null - alter) 

        qs.append(q)
    return np.array(qs)

def get_es(exposures , alter_binned, null_binned, N = 10000, alpha = 0.0027):
    taus_0 = []
    Rs_real = []
    sim_exposures = []
    for exp, exposure in enumerate(exposures):
        start_time = time.time()
        alter_model = pd.DataFrame(alter_binned)* exposure
        null_model = pd.DataFrame(null_binned) * exposure

        sim_data_from = null_model
        Rs = likelihood_ratio_test(null_model, alter_model, sim_data_from, N = N )
        
        real_data = alter_model
        real_null = binned_likelihood(real_data, null_model)
        real_alter =binned_likelihood(real_data, alter_model)
        R_real = sum((real_null - real_alter).to_numpy().flatten())

        tau_0, bin_centers, cdf = get_tau0(Rs, bin_num = 8000, alpha = alpha)

        Rs_real.append(R_real)
        taus_0.append(tau_0)
        sim_exposures.append(exposure)
        print('---------', time.time()- start_time,'-------------','\n')
        if ( Rs_real[exp-1]>taus_0[exp-1] ) and ( Rs_real[exp]<taus_0[exp] ):
            print('starts to reject')
            break
    return sim_exposures, taus_0, Rs_real


def likelihood_ratio_test(null_model, alter_model, sim_data_from, N = 10000 ):
    i = 0
    Rs = np.zeros(N)
    while i < N:
        data = pd.DataFrame(np.random.poisson(sim_data_from))
        
        null = binned_likelihood(data, null_model)
        alter =binned_likelihood(data, alter_model) 

        Rs[i] = sum((null - alter).to_numpy().flatten())
        i+=1
    return Rs



def get_tau0(data, bin_num = 8000, alpha = 0.01):
    bins = np.linspace(np.floor(min(data)),np.ceil(max(data)), bin_num)
    bin_centers = (bins[:-1]+bins[1:])/2
    
    count, _ = np.histogram(data, bins) 
    cdf = np.cumsum(count)/len(data)
    
    tau_0 = linear_intp(alpha, cdf,bin_centers)
    return tau_0, bin_centers, cdf


def get_binned_data_Er(pcles, nubbscale, pertonyr_dict):
    pertonyrs = np.zeros(list(pertonyr_dict.values())[0].shape)
    
    for pcle_name in pcles:
        print(pcle_name)
        if pcle_name in ['nubb']:
            print('nubb #/tonyr scaled by ', nubbscale)
            num_pertonyrscale = nubbscale
        else:
            num_pertonyrscale = 1
            
        pertonyr = np.array(pertonyr_dict.get(pcle_name) * num_pertonyrscale)
        print('scaled#/tonyr: ', np.sum(pertonyr),'original #/tonyr:' ,sum(pertonyr_dict.get(pcle_name)) )
        
        pertonyrs+=pertonyr
        print()
    return pertonyrs



def get_binned_data(pcles, nubbscale, binned_datas_dict, pcle_dict, 
                    metallicity = '', print_check = True, unit_pertyr = 1/u.tonne/u.yr):
    pcle_bins = np.zeros(list(binned_datas_dict.values())[0].shape)
    for pcle_name in pcles:
        if print_check:
            print(pcle_name)
        effnum = pcle_dict.get(pcle_name)[0]
        if metallicity == 'high':
            pertonyr = pcle_dict.get(pcle_name)[1]*unit_pertyr
        elif metallicity == 'low':
            pertonyr = pcle_dict.get(pcle_name)[2]*unit_pertyr
        else:
            print('select metallicity')
            pertonyr = 0*unit_pertyr
        #pertonyr = pcle_dict.get(pcle_name)[-1]
        if pcle_name in ['nubb']:
            if print_check:
                print('nubb #/tonyr scaled by ', nubbscale)
            num_pertonyrscale = nubbscale
        else:
            num_pertonyrscale = 1
            
        scaled_pertonyr = pertonyr*num_pertonyrscale
        perpcle_bin = np.array(binned_datas_dict.get(pcle_name)/effnum * scaled_pertonyr)
        if print_check:
            print('number of events: ', np.sum(np.array(binned_datas_dict.get(pcle_name))), 
                  ', eff num: ', effnum)
            print('#/tonyr:' ,scaled_pertonyr, 'sum of per pcle bin' , np.sum(perpcle_bin))

            print(perpcle_bin.shape)
            print()
        pcle_bins+=perpcle_bin
        
    return pcle_bins



def get_evenlybinned_data(pcles, nubbscale, binned_datas_dict, pcle_dict, 
                    metallicity = '', print_check = True, unit_eventR = 1/u.tonne/u.yr):
    binned_pcles_rate = np.zeros(list(binned_datas_dict.values())[0].shape)
    for pcle_name in pcles:
        if print_check:
            print(pcle_name)
            
        effnum = pcle_dict.get(pcle_name)[0]
        if metallicity == 'high':
            eventR = pcle_dict.get(pcle_name)[1]*unit_eventR
        elif metallicity == 'low':
            eventR = pcle_dict.get(pcle_name)[2]*unit_eventR
        else:
            print('select metallicity')
            eventR = 0*unit_eventR
            
        DT_longtime = effnum/eventR
        #pertonyr = pcle_dict.get(pcle_name)[-1]
        if pcle_name in ['nubb']:
            if print_check:
                print('nubb #/tonyr scaled by ', nubbscale)
            num_pertonyrscale = nubbscale
        else:
            num_pertonyrscale = 1
        
  
        binned_perpcle_rate = np.array(binned_datas_dict.get(pcle_name) / DT_longtime *num_pertonyrscale)

        if print_check:
            print('number of events: ', np.sum(np.array(binned_datas_dict.get(pcle_name))), 
                  ', eff num: ', effnum)
            print(metallicity+' eventR :' ,eventR, ', sum binned_perpcle_rate' , np.sum(binned_perpcle_rate))

            print(binned_perpcle_rate.shape)
            print()
        binned_pcles_rate+=binned_perpcle_rate
        
    return binned_pcles_rate


def binning_2dhist(folder, nuclei, pcles,  numevents, footnote, bin_S1, bin_S2, 
                   E_threshold_keV = 1, eff_type = '', corr = '', ebinding = True, plot= True, print_check = True):
    
    binneddatas, eff_nums, pertonyr_pcles_high, pertonyr_pcles_low = [],[],[],[]

    if plot:
        fig, ax = plt.subplots(1 ,len(pcles),  figsize = (10*len(pcles) ,6))
            
    for b, (pcle, numevent) in enumerate(zip(pcles, numevents)):
        data, num = get_data(folder, pcle, nuclei, numevent, footnote)
        if print_check:
            print(pcle, num)
        eff_nums.append(num)
        #get_data(data_path, pcle_name, bind, recoil_type, det, nuclei,numevent, footnote)
        if folder in ['thinNEST_unbinned','nestpy_unbinned', 'nestpy_unbinned_unbound', 
                      'nestpy_unbinned_NRwin_15_300_1e3_1e4', 'nestpy_unbinned_NRwin_15_300_25e2_16e3']:
            E, s1, s2, rmm, zmm, E_true = get_data_component(data, folder)
            
        elif folder in ['execNEST_results']:
            E, t_drift, z, s1, s2 = get_data_component(data, folder)
        
        
        binneddata = np.histogram2d(s1, np.log10(s2), [bin_S1, bin_S2])[0]
        binneddatas.append(binneddata)
        
        if plot: 
            ax[b].set_title(pcle)
            ax[b].hist2d(s1, np.log10(s2), [bin_S1, bin_S2])
        
        
        exposure_high,_,_,_,_,_ = read_pcle_cdf(pcle, nuclei, E_threshold_keV, eff_type, 
                                   recoil_type = '', read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, 
                                   metallicity = 'high', ebind = ebinding, plot_pdf = False, corr = corr)
        

        exposure_low,_,_,_,_,_ = read_pcle_cdf(pcle, nuclei, E_threshold_keV, eff_type, 
                                   recoil_type = '', read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, 
                                   metallicity = 'low', ebind = ebinding, plot_pdf = False, corr = corr)
        pertonyr_pcles_high.append(exposure_high.value)
        pertonyr_pcles_low.append(exposure_low.value)
        
    return np.array(binneddatas), eff_nums, np.array(pertonyr_pcles_high)/u.tonne/u.yr, np.array(pertonyr_pcles_low)/u.tonne/u.yr 


def binning_2dhist_1E7(pcles,bin_S1, bin_S2, nuclei = 'Xenon', folder_2d = 'nestpy_unbinned_unbound', 
                   E_threshold_keV = 1, eff_type = '', corr = '', ebinding = True, print_check = True):
    
    binneddatas, eff_nums, eventR_pcles_high, eventR_pcles_low = [],[],[],[]
            
    for pcle in pcles:
        file_name = os.path.join(folder_2d, 'S1S2_'+pcle+'.csv')
        print(file_name, os.path.exists(file_name))

        dataf = pd.read_csv(file_name)
        valid_dataf = dataf.loc[(dataf['cS1[phd]'] > 0 ) & (dataf['cS2[phd]']>0) ]

        s1 = valid_dataf['cS1[phd]']
        log10s2 = np.log10(valid_dataf['cS2[phd]'])
        eff_num =len(dataf)
        if print_check:
            print(pcle, 'sim number = ', eff_num, 
                  '\nvalid_dataf = ', len(valid_dataf))
        
        binneddata = np.histogram2d(s1, log10s2, [bin_S1, bin_S2])[0]
        eff_nums.append(eff_num)
        binneddatas.append(binneddata)
        
        eventR_high,_,_,_,_,_ = read_pcle_cdf(pcle, nuclei, E_threshold_keV, eff_type, 
                                   recoil_type = '', read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, 
                                   metallicity = 'high', ebind = ebinding, plot_pdf = False, corr = corr)
        
        eventR_low,_,_,_,_,_ = read_pcle_cdf(pcle, nuclei, E_threshold_keV, eff_type, 
                                   recoil_type = '', read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, 
                                   metallicity = 'low', ebind = ebinding, plot_pdf = False, corr = corr)
        eventR_pcles_high.append(eventR_high.value)
        eventR_pcles_low.append(eventR_low.value)
        
    return np.array(binneddatas), eff_nums, np.array(eventR_pcles_high)/u.tonne/u.yr, np.array(eventR_pcles_low)/u.tonne/u.yr 



def binning_1dhist(binned_data_pd, folder, nuclei, pcles, size, bin_, eff_type, 
                   metallicity, recoil_type = '', ebinding = True, corr = '', corr_sim = '', 
                   unit_pdf = (u.tonne*u.yr*u.keV)**(-1), unit_Er = u.keV, E_threshold_keV = 1, 
                   plot= True, log = True, print_check = True):
    
    for p, pcle in enumerate(pcles):
        
        if plot:
            fig, ax = plt.subplots(figsize = (8,5))
        if recoil_type == '':
            recoil_type = get_recoil_type(pcle)
        if print_check:
            print(recoil_type)
        ebind = get_Ebind(pcle,recoil_type, ebinding)

        
        
        eventR,_, valid_Er_keV, valid_pdf, _, _ = read_pcle_cdf(pcle, nuclei, E_threshold_keV, eff_type, recoil_type = recoil_type, 
                  read_pdf = True, Er_keV = 0, pdf = 0, endpt = -1, metallicity = metallicity, 
                                                              ebind = True, plot_pdf = plot, corr = corr)
        if plot:
            ax.plot(valid_Er_keV, valid_pdf/norm)
        
        if '_smear' in corr_sim:
            print('smear energy ')
            corr_sim_ = '_'+metallicity+corr_sim
            
            if get_solar_components(pcle):
                print('solar components')
                
            else:
                print('not solar components')
                corr_sim_ = corr_sim
            
        else:
            print('other')
            corr_sim_ =corr_sim
        
        file_name = 'Er_'+pcle+ebind+recoil_type+'_'+nuclei+'_'+eff_type+'_'+str(size)+'_thrd'+str(E_threshold_keV)+'keV'+corr_sim_+'.txt'
        file= os.path.join(folder,file_name)
        if print_check:
            print(file, corr_sim)
       
        if corr_sim == '_smearsim':
            _, _, Er_keV_sim, _ = read_file_dataline(file)
        else:
            Er_keV_sim, _  = np.array(read_file_data(file))
        eff_num = len(Er_keV_sim)
        if plot:
            counts_all, bin_ , _ = ax.hist(Er_keV_sim*unit_Er, bin_, log = True, density = True, alpha = 0.5)
        else:
            counts_all, bin_ = np.histogram(Er_keV_sim*unit_Er, bin_)
        DT_longtime = eff_num/eventR
        rates = counts_all/DT_longtime
        
        col_name =metallicity+'_'+pcle+ebind+recoil_type+'_'+nuclei+'_'+eff_type+'_'+str(size)+'_thrd'+str(E_threshold_keV)+'keV ' +corr+str(rates.unit)
        binned_data_pd[col_name] = rates.value

        if print_check:
            print(pcle, recoil_type, ebind, 'terminate at ', str(max(valid_Er_keV)), 
                  '\nwhole spectrum eventR = ', eventR, 
              '\nfraction of counts within Er ', min(bin_), ' to ', max(bin_), '= ', sum(counts_all)/eff_num, 
              '\neventR within Er ', min(bin_), ' to ', max(bin_), '= ', sum(rates), '\n\n'
             )
        if plot:
            ax = setup_cdfpdf_ax(ax, pcle + ' '+eff_type+' '+str(norm), 
                    'Er_keV', str((valid_pdf/norm).unit), True, '', 20, 20, 
                    vlines = [E_threshold_keV], hlines = [0], xlims = [0,0], 
                                 ylims = [max(valid_pdf/norm).value/1000,
                                          1.1*max(valid_pdf*unit_pdf/norm).value], log = [1,1])
       
       
    return binned_data_pd