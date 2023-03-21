#%%
''' Import Necessary Modules '''
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest
import mne
import matplotlib.pyplot as plt

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum

#%%
'''Load Data '''

#first the fMRI matrix from Keith
path="../../Data/tbiattn_Xmeasure_20230118.mat"

mat_contents = sio.loadmat(path)

print(mat_contents.keys())

#then the EEG data

#spectra_data=sio.loadmat("../Data/TBI_restingEEG_Spectra.mat")
subject_004=sio.loadmat("../../Data/EO.mat")
# %%
''' Get all the necessary data from fMRI file'''

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


#%%
''' Make large dataframe'''
data=pd.concat([falff,tbi_subj,hc_subj,session,age,sex,Subject_ID], axis=1)


#%%
''' Only TBI and HC Session 1'''
tbi_sess1=data[(data["isTBI"]==1) & (data["Session"]==1)]
hc_sess1=data[(data["isHC"]==1) & (data["Session"]==1)]

tbi_sess2=data[(data["isTBI"]==1) & (data["Session"]==2)]

# %%
'''Removing NaN '''

#subset the columns that have the original 86 region parcellation
c = tbi_sess1.columns[0:86] 

#use this to remove nan's
falff_tbi_sess1=tbi_sess1.dropna(axis=0, how="any", subset=c)
falff_hc_sess1=hc_sess1.dropna(axis=0, how="any",subset=c)

#%%
''' Make these arrays'''
falff_tbi_sess1_arr=falff_tbi_sess1.iloc[:,0:86].to_numpy(dtype="float32")
falff_hc_sess1_arr=falff_hc_sess1.iloc[:,0:86].to_numpy(dtype="float32")
# %%
#between tbi and hc
stat, TBI_HC_Sess1_p_vals=stats.ttest_ind(falff_tbi_sess1_arr[:86], falff_hc_sess1_arr[:86])

reject, TBI_HC_Sess1_p_val_correct=multitest.fdrcorrection(TBI_HC_Sess1_p_vals)

#%%
#between TBI sess1 and sess2

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

#%%
#stats
tbi_stats, TBI_Sess1_2_p_vals=stats.ttest_rel(falff_tbi_common_sess1_arr[:86], falff_tbi_sess2_arr[:86])

tbi_reject, TBI_Sess1_2_p_val_correct=multitest.fdrcorrection(TBI_Sess1_2_p_vals)


# %%
''' EEG DATA '''
data1=spectra_data["TBIATTN003_EO"]
data2=spectra_data["TBIATTN005_EO"]
data3=spectra_data["TBIATTN004_EC"]
# %%
# %%
'''Find subjects that intersect'''

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
eeg_sub_list=[]
for key in spectra_data.keys():
    eeg_sub_list.append(key.split("_")[0][-3:])

    
# %%
common_subjects=set(eeg_sub_list).intersection(set(falff_sub_list))

for person in falff_sub_list:
    if bool(person not in eeg_sub_list):
        print('Has fMRI but no EEG: {}'.format(person))
# %%
# for the EEG      TBIATTN001_EC
# for the fmri      TBIATTN_003_S1

select_falff_common_subjects = ['TBIATTN_{}_S1'.format(x) for x in common_subjects]
select_eeg_common_subjects=['TBIATTN{}_{}'.format(id, condi) for id in common_subjects for condi in ['EO', 'EC']]

#%%
'''Only get data from common subjects'''
#first fMRI
falff_tbi_sess1 = falff_tbi_sess1.set_index(falff_tbi_sess1["Sub_ID"])
falff_tbi_sess1=falff_tbi_sess1.loc[select_falff_common_subjects]

#%%
#then EEG
eeg_tbi_sess1 = {}

selected_short = ['TBIATTN{}'.format(id) for id in common_subjects]


for selected in selected_short:
    eeg_tbi_sess1[selected] = np.zeros((2, 129, 126), dtype=float)

selected_long = select_eeg_common_subjects

for selected in selected_long:
    if selected[-2:] == 'EC':
        eeg_tbi_sess1[selected[:-3]][0] = spectra_data[selected]
    else:
        eeg_tbi_sess1[selected[:-3]][1] = spectra_data[selected]
# %%
avg_eeg_spectra_tbi_sess1={}
for id in eeg_tbi_sess1.keys():
    avg_eeg_spectra_tbi_sess1[id]=eeg_tbi_sess1[id].mean(axis=0)

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
# %%
subject_005=sio.loadmat("../../Data/TBIATTN005_EO.mat")
subject_005=subject_005["TBIATTN005_EO"]
subject_007=sio.loadmat("../../Data/TBIATTN007_EO.mat")
subject_007=subject_007["TBIATTN007_EO"]

#%%
# Create some dummy metadata
n_channels = 129
sampling_freq = 1000  # in Hertz
info = mne.create_info(n_channels, sfreq=sampling_freq,ch_types=['eeg']*129 )
print(info)

#%% Convert mV to V (so MNE is happy)
# subject_004=subject_004["EO"]
# subject_004 = subject_004*1e-6
subject_005 = subject_005*1e-6
subject_007 = subject_007*1e-6
#sub_004_EC = sub_004_EC*1e-6
#%%
preproc = mne.io.RawArray(subject_004, info)
#preproc_EC = mne.io.RawArray(sub_004_EC, info)
preproc_005 = mne.io.RawArray(subject_005, info)
preproc_007 = mne.io.RawArray(subject_007, info)
# %%
spectrum_004=preproc.compute_psd(fmin= 0, fmax=40, method='welch', n_fft=1000)
spectrum_005=preproc_005.compute_psd(fmin= 0, fmax=40, method='welch', n_fft=1000)
spectrum_007=preproc_007.compute_psd(fmin= 0, fmax=40, method='welch', n_fft=1000)

#spectrum_EC=preproc_EC.compute_psd(fmin= 2, fmax=40)
# %%
spectrum_data, freqs=spectrum.get_data(return_freqs=True)
spectrum_data_005, freqs=spectrum_005.get_data(return_freqs=True)
spectrum_data_007, freqs=spectrum_007.get_data(return_freqs=True)

#%%
fm = FOOOFGroup(peak_width_limits=[2, 10], max_n_peaks=4, min_peak_height=0.1, aperiodic_mode='fixed')
fm.fit(freqs, spectrum_data_005, [5, 40])
fm.print_results()

# %%
#test one fooof
test_fm=FOOOF(peak_width_limits=[2, 10], 
              max_n_peaks=6,
              min_peak_height=0.1,
              aperiodic_mode="fixed")
freq_range = [5, 40]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
test_fm.report(freqs, spectrum_data_005.mean(axis=0), freq_range)
# %%
