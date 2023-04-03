#%%
''' Import Necessary Modules '''
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
import mne
import matplotlib.pyplot as plt
import os

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof import FOOOF
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

#%% '''Load Data '''

#first the fMRI matrix from Keith
path="../../Data/tbiattn_Xmeasure_20230118.mat"

mat_contents = sio.loadmat(path)

print(mat_contents.keys())
#%% then the EEG data

data_path='../../Data/TBI_fALFF-EEG_Spectra/'
condition = 'EO'

#%% CREATING A DICT FROM THE MAIN FOLDER THAT CONTAINING THE EEG DATA 
data_dict = {}

avail_sub = [x[-3:] for x in os.listdir('../../Data/TBI_fALFF-EEG_Spectra')]

for subject in avail_sub:
    tmp = os.listdir('{}TBI{}/'.format(data_path, subject))
    data_file = [x for x in tmp if x.split('.')[0][-2:]=='EO']
    if len(data_file) != 0:
        tmp = sio.loadmat('{}TBI{}/{}'.format(data_path, 
                                          subject, 
                                          data_file[0]))
        data_key = [i for i in tmp.keys() if i[-2:]==condition][0]
        if (isinstance(tmp[data_key], np.ndarray)) and (data_tmp.shape[0] == 129):
            data_dict[subject] = tmp[data_key]*1e-6

#%% CREATE A SFREQ_DICT THAT CONTAINS SFREQ FOR EACH SUBJECT
# TBI001 - TBI018 : 1000Hz  
# TBI021 - TBI052 : 250Hz  
# TBI055 - TBI061 : 1000Hz

thousand_1 = ['0{}'.format(x) if len(str(x))==2 else '00{}'.format(x) for x in range(1, 19)]
thousand_2 = ['0{}'.format(x) if len(str(x))==2 else '00{}'.format(x) for x in range(55, 62)]
thousandHz = thousand_1 + thousand_2
twofiftyHz = ['0{}'.format(x) for x in range(21, 53)]

sfreq_dict = { k:1000 for k in thousandHz}
sfreq_dict.update({ k:250 for k in twofiftyHz})

#%% DEFINE HELPER FUNCTIONS 
def create_custom_raw (eeg_data, n_ch, sfreq):
    ''' Create a raw with given propeties'''
    info = mne.create_info(n_ch, 
                           sfreq=sfreq,
                           ch_types=['eeg']*n_ch)
    return mne.io.RawArray(eeg_data, info)

def compute_spectra (raw, fmin, fmax, n_fft=None):
    power_spectrum = raw.compute_psd(fmin=fmin, fmax=fmax, method='welch', n_fft=n_fft)
    return power_spectrum

#%% CREATE POWER_SPECTRUM_DICT

n_ch = 129
fmin = 0
fmax = 40

power_spectrum_dict = {}

for sub in data_dict.keys():
    
    raw = create_custom_raw(data_dict[sub], n_ch, sfreq_dict[sub])
    print(sub)
    power_spectrum_dict[sub] = compute_spectra(raw, fmin, fmax, n_fft=sfreq_dict[sub])

# %% ''' Get all the necessary data from fMRI file'''

falff=pd.DataFrame(mat_contents["fALFF"])
tbi_subj=pd.DataFrame(mat_contents["isTBI"])
tbi_subj.columns=["isTBI"]
hc_subj=pd.DataFrame(mat_contents["isHC"])
hc_subj.columns=["isHC"]
session=pd.DataFrame(mat_contents["Session"])
session.columns=["Session"]
age=pd.DataFrame(mat_contents["Age"])
age.columns=["Age"]
sex=pd.DataFrame(mat_contents["Sex"])
sex.columns=["Sex"]
Subject_ID=pd.DataFrame(mat_contents["Subject"])
Subject_ID.columns=["Sub_ID"]
Subject_ID["Sub_ID"] = [x[0] for x in Subject_ID["Sub_ID"]]


#%% ''' Make large fMRI data dataframe'''

data=pd.concat([falff,tbi_subj,hc_subj,session,age,sex,Subject_ID], axis=1)


#%% ''' Filter fMRI to contain Only TBI and HC Session 1'''

tbi_sess1=data[(data["isTBI"]==1) & (data["Session"]==1)]
hc_sess1=data[(data["isHC"]==1) & (data["Session"]==1)]

tbi_sess2=data[(data["isTBI"]==1) & (data["Session"]==2)]

# %% '''Removing NaNs in fMRI data '''


#subset the columns that have the original 86 region parcellation
c = tbi_sess1.columns[0:86] 

#use this to remove nan's
falff_tbi_sess1=tbi_sess1.dropna(axis=0, how="any", subset=c)
falff_hc_sess1=hc_sess1.dropna(axis=0, how="any",subset=c)

#%% ''' Convert these to arrays for packages'''

falff_tbi_sess1_arr=falff_tbi_sess1.iloc[:,0:86].to_numpy(dtype="float32")
falff_hc_sess1_arr=falff_hc_sess1.iloc[:,0:86].to_numpy(dtype="float32")
# %% ''' T stats between tbi and hc '''

stat, TBI_HC_Sess1_p_vals=stats.ttest_ind(falff_tbi_sess1_arr[:86], falff_hc_sess1_arr[:86])

reject, TBI_HC_Sess1_p_val_correct=multitest.fdrcorrection(TBI_HC_Sess1_p_vals)

#%% ''' T stats between TBI sess1 and sess2'''


# drop row/subjects that have NAs
falff_tbi_sess2=tbi_sess2.dropna(axis=0, how="any",subset=c)

#first get two subject lists
falff_sess1_list=[]
for item in falff_tbi_sess1["Sub_ID"]:
    falff_sess1_list.append(item.split("_")[1])
    
falff_sess2_list=[]
for item in falff_tbi_sess2["Sub_ID"]:
    falff_sess2_list.append((item.split("_")[1]))


# Find the subject # that have both a S1 and S2
common_sessions=set(falff_sess1_list).intersection(set(falff_sess2_list))

# Re-create the full ID for the subject that have both a S1 and S2
falff_common_sess1 = ['TBIATTN_{}_S1'.format(x) for x in common_sessions]
falff_common_sess2 = ['TBIATTN_{}_S2'.format(x) for x in common_sessions]


falff_tbi_common_sess1 = falff_tbi_sess1.set_index(falff_tbi_sess1["Sub_ID"])
falff_tbi_common_sess1=falff_tbi_common_sess1.loc[falff_common_sess1]

falff_tbi_sess2 = falff_tbi_sess2.set_index(falff_tbi_sess2["Sub_ID"])
falff_tbi_sess2=falff_tbi_sess2.loc[falff_common_sess2]

falff_tbi_common_sess1_arr=falff_tbi_common_sess1.iloc[:,0:86].to_numpy(dtype="float32")
falff_tbi_sess2_arr=falff_tbi_sess2.iloc[:,0:86].to_numpy(dtype="float32")


#stats and FDR correction
tbi_stats, TBI_Sess1_2_p_vals=stats.ttest_rel(falff_tbi_common_sess1_arr[:86], falff_tbi_sess2_arr[:86])

tbi_reject, TBI_Sess1_2_p_val_correct=multitest.fdrcorrection(TBI_Sess1_2_p_vals)




# %%'''Find subjects that intersect in EEG and fMRI'''

#get falff subs
falff_sub_list=[]
for item in falff_tbi_sess1["Sub_ID"]:
    falff_sub_list.append(item.split("_")[1])
    

# %%
working_dict={}
for i in ['__header__', '__version__', '__globals__','freq']:
    working_dict[i] = spectra_data[i]
    spectra_data.pop(i)
# %%
eeg_sub_list=[x for x in data_dict.keys()]
# for key in spectra_data.keys():
#     eeg_sub_list.append(key.split("_")[0][-3:])

    
# %%
common_subjects=set(eeg_sub_list).intersection(set(falff_sub_list))

for person in falff_sub_list:
    if bool(person not in eeg_sub_list):
        print('Has fMRI but no EEG: {}'.format(person))
# %%
# for the EEG      TBIATTN001_EC
# for the fmri      TBIATTN_003_S1

select_falff_common_subjects = ['TBIATTN_{}_S1'.format(x) for x in common_subjects]
#select_eeg_common_subjects=['TBIATTN{}_{}'.format(id, condi) for id in common_subjects for condi in ['EO', 'EC']]
select_eeg_common_subjects=[x for x in common_subjects]
#%% '''Only get data from common subjects'''
#first fMRI
falff_tbi_sess1 = falff_tbi_sess1.set_index(falff_tbi_sess1["Sub_ID"])
falff_tbi_sess1=falff_tbi_sess1.loc[select_falff_common_subjects]
#reorder datasets for plotting
falff_tbi_sess1 = falff_tbi_sess1.sort_index( ascending=True)
#create avg flaff
falff_tbi_sess1['Avg_FALFF']=falff_tbi_sess1.iloc[:,0:85].mean(axis=1)

#%%
#frontal avg FALFF
f = pd.read_csv('../../Data/fs86_yeo7_lobe.txt')
frontal_falff = f[f['Lobe']=='frontal']
frontal_index=frontal_falff.IDX.tolist()
falff_tbi_sess1['Frontal_FALFF']=falff_tbi_sess1.iloc[:,frontal_index].mean(axis=1)

#then EEG
#eeg_tbi = {}

#selected_short = ['TBIATTN{}'.format(id) for id in common_subjects]

# for selected in selected_short:
#     eeg_tbi_sess1[selected] = np.zeros((2, 129, 126), dtype=float)

# selected_long = select_eeg_common_subjects

# for selected in selected_long:
#     if selected[-2:] == 'EC':
#         eeg_tbi_sess1[selected[:-3]][0] = spectra_data[selected]
#     else:
#         eeg_tbi_sess1[selected[:-3]][1] = spectra_data[selected]
# # %%
# avg_eeg_spectra_tbi_sess1={}
# for id in eeg_tbi_sess1.keys():
#     avg_eeg_spectra_tbi_sess1[id]=eeg_tbi_sess1[id].mean(axis=0)

# %%

# def check_nans(data, nan_policy='zero'):
#     """Check an array for nan values, and replace, based on policy."""

#     # Find where there are nan values in the data
#     nan_inds = np.where(np.isnan(data))

#     # Apply desired nan policy to data
#     if nan_policy == 'zero':
#         data[nan_inds] = 0
#     elif nan_policy == 'mean':
#         data[nan_inds] = np.nanmean(data)
#     else:
#         raise ValueError('Nan policy not understood.')

#     return data

# # %%
# freqs=[]
# for item in working_dict["freq"]:
#     freqs.append(item[0])

# freqs=np.array(freqs)
# #%%
# spectrum=np.power(10,data1)
# fm = FOOOFGroup(peak_width_limits=[0.5, 8], max_n_peaks=6, min_peak_height=0.2, aperiodic_mode='knee')
# fm.report(freqs, spectrum, [5, 40])

#for sub in avg_eeg_spectra_tbi_sess1.keys():

# %%
''' Load EEG data, compute power spectrum, do FOOOF'''
#%%
'''GROUP'''

# dic_concat_sub_data={"subject_004" : spectrum_data,
#                      "subject_005" : spectrum_data_005,
#                      "subject_007" : spectrum_data_007}

def get_aper_exp (sub_spect, condi, freqs , type_data):
    '''
        Computes the FOOOF of a spectra and
        returns its averaged aperiodic exponent.
    '''
    if type_data=="Group":
        fg = FOOOFGroup(peak_width_limits=[2, 10], 
                        max_n_peaks=4, 
                        min_peak_height=0.1, 
                        aperiodic_mode=condi)
        fg.fit(freqs, sub_spect, [5, 40])
        results=fg.get_params('aperiodic_params', 'exponent').mean(axis=0)
    if type_data=="Single":
        fm=FOOOF(peak_width_limits=[2, 10], 
                 max_n_peaks=4, 
                 min_peak_height=0.1, 
                 aperiodic_mode=condi)
        fm.fit(freqs, sub_spect, [5, 40])
        results=fm.get_params('aperiodic_params', 'exponent')
        
    
    return results

#%% For all subject, run the fooof fitting with "knee" and "fixed", then store in
#   the new df "group_exp_results"

group_exp_results=pd.DataFrame(columns=["Subject_ID", "fixed_aper_exp", "knee_aper_exp"])

for sub in power_spectrum_dict.keys():
    power_spectra,freq= power_spectrum_dict[sub].get_data(return_freqs=True)
    print(np.ndim(freq), np.ndim(power_spectra))
    fixed_aper_exp = get_aper_exp(sub_spect=power_spectra,condi= "fixed", freqs=freq,
                                  type_data='Group')
    knee_aper_exp  = get_aper_exp(power_spectra, "knee", freqs=freq, type_data="Group")
    
    group_exp_results.loc[len(group_exp_results.index)] = [sub, fixed_aper_exp, knee_aper_exp]

#%%
#filter EEG subjects
group_exp_results_filtered= group_exp_results[group_exp_results['Subject_ID'].isin(select_eeg_common_subjects)]
group_exp_results_filtered=group_exp_results_filtered.sort_values(by='Subject_ID',ascending=True)

#%%
frontal_picks= ['1','6','5','11','13']
frontal_group_exp=pd.DataFrame(columns=["Subject_ID", "fixed_aper_exp", "knee_aper_exp"])

for sub in power_spectrum_dict.keys():
    power_spectra,freq= power_spectrum_dict[sub].get_data(return_freqs=True, 
                                                          picks=frontal_picks)
    print(np.ndim(freq), np.ndim(power_spectra))
    fixed_aper_exp = get_aper_exp(sub_spect=power_spectra.mean(axis=0),condi= "fixed", freqs=freq,
                                  type_data='Single')
    knee_aper_exp  = get_aper_exp(power_spectra.mean(axis=0), condi="knee", freqs=freq, 
                                  type_data="Single")
    
    frontal_group_exp.loc[len(frontal_group_exp.index)] = [sub, fixed_aper_exp, knee_aper_exp]

frontal_group_exp_filter=frontal_group_exp[frontal_group_exp['Subject_ID'].isin(select_eeg_common_subjects)]
# ch_locations=pd.read_csv('../../Data/EGI_129_ch_location.csv')
# ch_dict=dict(zip(ch_locations.labels, ch_locations.Number))
# [ch_dict.pop(key) for key in ['E130','E131','E132']]

#%%




# %%
'''ONE FOOOF MEAN CHANNELS'''
test_fm=FOOOF(peak_width_limits=[2, 10], 
              max_n_peaks=6,
              min_peak_height=0.1,
              aperiodic_mode="knee")
freq_range = [5, 40]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
test_fm.report(freqs, power_spectrum_dict.mean(axis=0), freq_range)
plt.show()
#%%
results_test=test_fm.get_results()
# %%
''' investigate errors'''

from fooof.analysis.error import compute_pointwise_error_fm, compute_pointwise_error_fg
# %%
compute_pointwise_error_fm(test_fm, plot_errors=True)

#%%

plt.scatter(frontal_group_exp_filter["knee_aper_exp"],falff_tbi_sess1['Frontal_FALFF'])

# Set the x and y axis labels
plt.xlabel('Frontal EEG Exponent Value')
plt.ylabel('Frontal FALFF Value')
plt.title('Knee Aperiodic Exp')


plt.show()




#%%

tt = ch_locations.copy()
tt = tt[tt['Number'] <= 129]

dict_dig={}

for row in range(tt.shape[0]):
    
    lab = str(row)
    x_pt = float(tt.loc[row]['X'])
    y_pt = float(tt.loc[row]['Y'])
    z_pt = float(tt.loc[row]['Z'])
    
    dict_dig[lab] = np.array([x_pt, y_pt, z_pt])
    

my_montage = mne.channels.make_dig_montage(
    ch_pos=dict_dig, 
    # nasion=None, 
    # lpa=None, 
    # rpa=None, 
    # hsp=None, 
    # hpi=None, 
    coord_frame='head')




# %%
