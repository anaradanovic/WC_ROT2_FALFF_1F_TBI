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
import re

from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof import FOOOF
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectrum
from fooof.analysis.error import compute_pointwise_error_fm, compute_pointwise_error_fg

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
 #DEFINE FOOOF FUNCTIONS
# ''' Load EEG data, compute power spectrum, do FOOOF'''

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

#%% '''Load Data '''

#first the fMRI matrix from Keith
path="../../Data/tbiattn_Xmeasure_20230118.mat"
mat_contents = sio.loadmat(path)

# for key in mat_contents.keys():
#     print(key)

#then the TBI EEG data from NaYOUNG
#%% Loading TBI EEG data
data_path='../../Data/TBI_fALFF-EEG_Spectra/'
condition = 'EO'

# CREATING A DICT FROM THE MAIN FOLDER THAT CONTAINS THE EEG DATA 
TBI_data_dict = {}

avail_sub = [x[-3:] for x in os.listdir('../../Data/TBI_fALFF-EEG_Spectra')]

for subject in avail_sub:
    if subject != "ore":
        tmp = os.listdir('{}TBI{}/'.format(data_path, subject))
        data_file = [x for x in tmp if x.split('.')[0][-2:]=='EO']
        if len(data_file) != 0:
            tmp = sio.loadmat('{}TBI{}/{}'.format(data_path, 
                                                  subject, 
                                                  data_file[0]))
            data_key = [i for i in tmp.keys() if i[-2:]==condition][0]
            if (isinstance(tmp[data_key], np.ndarray)) and (tmp[data_key].shape[0] == 129):
                TBI_data_dict[subject] = tmp[data_key]*1e-6


# CREATE A SFREQ_DICT THAT CONTAINS SFREQ FOR EACH SUBJECT
# TBI001 - TBI018 : 1000Hz  
# TBI021 - TBI052 : 250Hz  
# TBI055 - TBI061 : 1000Hz

thousand_1 = ['0{}'.format(x) if len(str(x))==2 else '00{}'.format(x) for x in range(1, 19)]
thousand_2 = ['0{}'.format(x) if len(str(x))==2 else '00{}'.format(x) for x in range(55, 62)]
thousandHz = thousand_1 + thousand_2
twofiftyHz = ['0{}'.format(x) for x in range(21, 53)]

sfreq_dict = { k:1000 for k in thousandHz}
sfreq_dict.update({ k:250 for k in twofiftyHz})


# Create TBI eeg power spectrum dict
n_ch = 129
fmin = 0
fmax = 40

TBI_power_spectrum_dict = {}

for sub in TBI_data_dict.keys():
    
    raw = create_custom_raw(TBI_data_dict[sub], n_ch, sfreq_dict[sub])
    # raw.set_montage(hydra_mon, on_missing='ignore')
    print(sub)
    TBI_power_spectrum_dict[sub] = compute_spectra(raw, fmin, fmax, n_fft=sfreq_dict[sub])

dem_data_HC=pd.DataFrame(columns=["Sub_ID", "Age", "Sex"])

#%% Read in the HC and get them into a form that can be used for FOOOF

# Your path should like this: ~/ROOT/proj-PROJECT/DATA_FOLDER/
#                            SUBJECT/SESSION
ROOT = ".."
# PROJECT = "./"
DATA_FOLDER = "derivatives"
TASK = "eyesclose"  # Can only take 2 values: "ant" or "eyesclose"
DATATYPE = "eeg"
DESCRIPTION = "processed"
# EXTENSION = ".fif"  # The extension of your file (here mne file so .fif)
EXTENSION_CSV = ".csv"  # The extension of your file (here mne file so .fif)
SUFFIX = "eeg"
SESSION = "01"

# Define the root path
root_path_structure = [
    ROOT,
    # f"proj-{PROJECT}",
    DATA_FOLDER,
]

pattern = re.compile(r"sub-HC\d+")
# List all the subjects
folder_contains = os.listdir(os.path.join(*root_path_structure))
subject_list = sorted([subj for subj in folder_contains if pattern.search(subj)])

HC_power_dict={}

for subject in subject_list:
    # Get the subject number
    subject_number = int(re.findall(r"\d+", subject)[0])

    # Get the subject_id

    subject_id = subject[4:]
    print(subject_id)

    if TASK == "ant":
        status = "expe"
    elif TASK == "eyesclose":
        status = "resting"

    # Generate the path
    file_path_structure = root_path_structure + [
        subject,
        f"ses-{SESSION}",
        DATATYPE,
        "raw",
        status,
        "processed",
    ]
    file_path = os.path.join(*file_path_structure)

    # Generate the filename for the "csv" file
    spectra_filename_structure = [
        f"sub-{subject_id}",
        f"ses-{SESSION}",
        f"task-{TASK}",
        f"spectra{EXTENSION_CSV}"
    ]
    
    spectra_filename = os.path.join(file_path, "_".join(spectra_filename_structure))
    if (
        "bad" not in subject
    ):
        spectra=pd.read_csv(spectra_filename)
        HC_power_dict[subject_id]=spectra
        
        beh_filename_structure= [
        f"sub-{subject_id}",
        f"ses-{SESSION}",
        f"task-{TASK}",
        f"spectra{EXTENSION_CSV}"
    ]
    
    #Generate the filename for the metadata file
        metadata_filename_structure = root_path_structure + [
        f"sub-{subject_id}",
        f"ses-{SESSION}",
        "beh",
        "sub-{}_ses-{}_desc-metadata_beh.csv".format(subject_id,SESSION)]
        
        metadata_filename=os.path.join(*metadata_filename_structure)
        
        temp_dem_data=pd.read_csv(metadata_filename)
        age=temp_dem_data.loc[0,"age"]
        sex=temp_dem_data.loc[0,"sex"]
        
        
        dem_data_HC.loc[len(dem_data_HC.index)] = [subject_id, age, sex]
        
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
hc=hc_sess1.columns[0:86]

#use this to remove nan's
falff_tbi_sess1=tbi_sess1.dropna(axis=0, how="any", subset=c)
falff_hc_sess1=hc_sess1.dropna(axis=0, how="any",subset=hc)

#%% ''' Convert these to arrays for packages'''

falff_tbi_sess1_arr=falff_tbi_sess1.iloc[:,0:86].to_numpy(dtype="float32")
falff_hc_sess1_arr=falff_hc_sess1.iloc[:,0:86].to_numpy(dtype="float32")

# drop row/subjects that have NAs
falff_tbi_sess2=tbi_sess2.dropna(axis=0, how="any",subset=c)

# %% ''' T stats between tbi and hc '''

stat, TBI_HC_Sess1_p_vals=stats.ttest_ind(falff_tbi_sess1_arr[:86], falff_hc_sess1_arr[:86])

reject, TBI_HC_Sess1_p_val_correct=multitest.fdrcorrection(TBI_HC_Sess1_p_vals)

#%% ''' T stats between TBI sess1 and sess2'''
#%% ''' Z SCORING '''

# Define the dataset
data_norm = frontal_HC_exponent_results["fixed_aper_exp"].to_list()

# Calculate the mean and standard deviation
mean = sum(data_norm) / len(data_norm)
std_dev = stats.tstd(data_norm)

# Calculate the z-scores of the dataset
z_scores = [(x - mean) / std_dev for x in (TBI_frontal_group_exp_filter["fixed_aper_exp"].to_list()) ]

# Create a histogram of the z-scores
plt.hist(z_scores, bins=10)
plt.xlabel("Z-score")
plt.ylabel("Frequency")
plt.title("Distribution of Z-scores of TBI compared to HC")
plt.show()

#%%
#check age differences between TBI and HC
TBI_Age=falff_tbi_sess1["Age"]
#get HC data
dem_data_HC_filtered=dem_data_HC[dem_data_HC['Sub_ID'].isin(frontal_HC_exponent_results['Subject_ID'])]
HC_Age=dem_data_HC_filtered["Age"]
#%%get between session common subjects

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


# #stats and FDR correction
# tbi_stats, TBI_Sess1_2_p_vals=stats.ttest_rel(falff_tbi_common_sess1_arr[:86], falff_tbi_sess2_arr[:86])

# tbi_reject, TBI_Sess1_2_p_val_correct=multitest.fdrcorrection(TBI_Sess1_2_p_vals)


# %%'''Find subjects that intersect in EEG and fMRI'''

#get falff TBI subs
TBI_falff_sub_list=[]
for item in falff_tbi_sess1["Sub_ID"]:
    TBI_falff_sub_list.append(item.split("_")[1])
    


hc_falff_sub_list=[]
for item in falff_hc_sess1["Sub_ID"]:
    hc_falff_sub_list.append(item.split("_")[1])


# working_dict={}
# for i in ['__header__', '__version__', '__globals__','freq']:
#     working_dict[i] = spectra_data[i]
#     spectra_data.pop(i)
eeg_TBI_list=[x for x in TBI_power_spectrum_dict.keys()]
eeg_HC_list= [x[-3:] for x in HC_power_dict.keys()]
# for key in spectra_data.keys():
#     eeg_sub_list.append(key.split("_")[0][-3:])

common_TBI_subjects=set(eeg_TBI_list).intersection(set(TBI_falff_sub_list))

common_HC_subjects= set(eeg_HC_list).intersection(set(hc_falff_sub_list))


# for person in falff_sub_list:
#     if bool(person not in eeg_sub_list):
#         print('Has fMRI but no EEG: {}'.format(person))

# for the fmri (TBIATTN_003_S1)
select_falff_common_subjects = ['TBIATTN_{}_S1'.format(x) for x in common_TBI_subjects]
select_falff_common_HC_subjects = ['HCATTN_{}_S1'.format(x) for x in common_HC_subjects]

#EEG analysis
select_eeg_common_subjects=[x for x in common_TBI_subjects]
select_HC_eeg_common_subjects = ['HC{}'.format(x) for x in common_HC_subjects]

#%% FALFF COMMON SUBJECTS WITH EEG FILTER

falff_tbi_sess1 = falff_tbi_sess1.set_index(falff_tbi_sess1["Sub_ID"])
falff_tbi_sess1=falff_tbi_sess1.loc[select_falff_common_subjects]

falff_hc_sess1=falff_hc_sess1.set_index(["Sub_ID"])
falff_hc_sess1=falff_hc_sess1.loc[select_falff_common_HC_subjects]
#reorder datasets for plotting
falff_tbi_sess1 = falff_tbi_sess1.sort_index( ascending=True)
falff_hc_sess1=falff_hc_sess1.sort_index( ascending=True)

#%% Calculate AVG FALFF for different areas TBI and HC
#avg flaff
#falff_tbi_sess1['Avg_FALFF']=falff_tbi_sess1.iloc[:,0:85].mean(axis=1)

# frontal avg FALFF TBI AND HC
f = pd.read_csv('../../Data/fs86_yeo7_lobe.txt')
frontal_falff = f[f['Lobe']=='frontal']
frontal_index=frontal_falff.IDX.tolist()
falff_tbi_sess1['Frontal_FALFF']=falff_tbi_sess1.iloc[:,frontal_index].mean(axis=1)

falff_hc_sess1['Frontal_FALFF']=falff_hc_sess1.iloc[:,frontal_index].mean(axis=1)

#%%''' HEALTHY CONTROLS FOOF all elecrodes (NOT COMPLETE!!!! STILL NEEDS WORK)'''

for sub_HC in HC_power_dict.keys():
    power_spectra= HC_power_dict['HC001']
    power_spectra=power_spectra.loc[:, ~power_spectra.columns.str.contains('freq')].to_numpy()
    freq=HC_power_dict[sub_HC].loc[:,'freq'].to_numpy()
    print(np.ndim(freq), np.ndim(power_spectra))
    fixed_aper_exp = get_aper_exp(sub_spect=power_spectra,condi= "fixed", freqs=freq,
                                  type_data='Group')
    knee_aper_exp  = get_aper_exp(power_spectra, "knee", freqs=freq, type_data="Group")
    
    group_exp_results.loc[len(group_exp_results.index)] = [sub, fixed_aper_exp, knee_aper_exp]

#filter EEG subjects
#####
#%% GET FRONTAL ELECTRODES FOR BOTH HC AND TBI
elec_names=pd.read_csv('../../Data/ChanlocLines128.csv')
elec_names=elec_names.to_dict(orient='list')

#remove E's from the names for Nayoung's data style
elec_dict = {}
for key, value in elec_names.items():
    if key.startswith("E"):
        new_key = key[1:]  # slice the key to remove the "E"
    else:
        new_key = key
    elec_dict[new_key] = value

#make it so that the values are not in a list
elec_name_dict = {key: value[0] for key, value in elec_dict.items()}

#now create the picks based off the dict
frontal_picks= []
for key, value in elec_name_dict.items():
    #if value.startswith("F") & (value not in (['FP2','FPz', 'FP1'])): #trying to remove problematic electrodes
    if value=="FCZ" or value=="Fz":
        frontal_picks.append(key)

#create same as above for HC data
column_name_dict = {key: value[0] for key, value in elec_names.items()}

#now create the picks based off the dict
frontal_picks_columns= []
for key, value in column_name_dict.items():
    #if value.startswith("F") & (value not in (['FP2','FPz', 'FP1'])):
    if value=="FCZ" or value=="Fz" :
        frontal_picks_columns.append(key)



#%% Run the fooof fitting with "knee" and "fixed" on all channels

group_exp_results=pd.DataFrame(columns=["Subject_ID", "fixed_aper_exp", "knee_aper_exp"])

for sub in TBI_power_spectrum_dict.keys():
    power_spectra,freq= TBI_power_spectrum_dict[sub].get_data(return_freqs=True)
    print(np.ndim(freq), np.ndim(power_spectra))
    fixed_aper_exp = get_aper_exp(sub_spect=power_spectra,condi= "fixed", freqs=freq,
                                  type_data='Group')
    #knee_aper_exp  = get_aper_exp(power_spectra, "knee", freqs=freq, type_data="Group")
    
    group_exp_results.loc[len(group_exp_results.index)] = [sub, fixed_aper_exp, knee_aper_exp]
group_exp_results_filtered= group_exp_results[group_exp_results['Subject_ID'].isin(select_eeg_common_subjects)]
group_exp_results_filtered=group_exp_results_filtered.sort_values(by='Subject_ID',ascending=True)

#%% FRONTAL TBI EXPONENTS CALCULATION
TBI_frontal_group_exp = pd.DataFrame(columns=["Subject_ID", "fixed_aper_exp"])

for sub in TBI_power_spectrum_dict.keys():
    power_spectra,freq= TBI_power_spectrum_dict[sub].get_data(return_freqs=True, 
                                                          picks=frontal_picks)
    print(np.ndim(freq), np.ndim(power_spectra))
    fixed_aper_exp = get_aper_exp(sub_spect=power_spectra.mean(axis=0),condi= "fixed", freqs=freq,
                                  type_data='Single')
    # knee_aper_exp = get_aper_exp(power_spectra.mean(axis=0), condi="knee", freqs=freq, 
    #                               type_data="Single")
    
    TBI_frontal_group_exp.loc[len(TBI_frontal_group_exp.index)] = [sub, fixed_aper_exp]

TBI_frontal_group_exp_filter = TBI_frontal_group_exp[TBI_frontal_group_exp['Subject_ID'].isin(select_eeg_common_subjects)]

#%% # FRONTAL HCs EXPONENTS CALCULATION
frontal_HC_exponent_results = pd.DataFrame(columns=["Sub_ID", "fixed_aper_exp"])

for sub_HC in HC_power_dict.keys():
    power_spectra= HC_power_dict[sub_HC]
    power_spectra=power_spectra.loc[:, frontal_picks_columns].to_numpy()
    frontal_power = power_spectra.mean(axis=1)
    
    freq=HC_power_dict[sub_HC].loc[:,'freq'].to_numpy()
    
    print(np.ndim(freq), np.ndim(power_spectra))
    
    fixed_aper_exp = get_aper_exp(sub_spect=frontal_power,
                                  condi= "fixed", 
                                  freqs=freq,
                                  type_data='Single')
    # knee_aper_exp  = get_aper_exp(frontal_power, 
    #                               condi = "knee", 
    #                               freqs=freq, 
    #                               type_data="Single")
    
    frontal_HC_exponent_results.loc[len(frontal_HC_exponent_results.index)] = [sub_HC, fixed_aper_exp]
    

#frontal_HC_exp_filter = frontal_HC_exponent_results[frontal_HC_exponent_results['Subject_ID'].isin(select_HC_eeg_common_subjects)]


#%% PLOT FLAFF VS EXPONENTS


plt.scatter(TBI_frontal_group_exp_filter["fixed_aper_exp"],
            falff_tbi_sess1['Frontal_FALFF'],
            color='blue',
            label='TBI')

plt.scatter(frontal_HC_exp_filter["fixed_aper_exp"],
            falff_hc_sess1['Frontal_FALFF'],
            color='red',
            label='Healthy Controls')


# Set the x and y axis labels
plt.xlabel('Frontal EEG Exponent Value')
plt.ylabel('Frontal FALFF Value')
plt.title('TBI and HC Fixed Frontal Aperiodic Exp x Avg Frontal FALFF')
plt.legend()

plt.show()

#%% PLOT HEALTHY CONTROLS
#merged=pd.merge(frontal_HC_exponent_results, dem_data_HC, on = "Sub_ID")

plt.scatter(merged[merged["Age"] < 50]["Age"], merged[merged["Age"] < 50]["fixed_aper_exp"])

# Set the x and y axis labels
plt.xlabel('Age')
plt.ylabel('Frontal EEG Exponent Value')
plt.title('HC Fixed Frontal Aperiodic Exp x Age <50')


plt.show()

    #%%
plt.scatter(TBI_age_set, TBI_frontal_group_exp_filter["fixed_aper_exp"])

# Set the x and y axis labels
plt.xlabel('Age')
plt.ylabel('Frontal EEG Exponent Value')
plt.title('TBI Fixed Frontal Aperiodic Exp x Age')
plt.show()

#%% STATS AGE VS EXPONENT
#HC
stat, pval = stats.pearsonr(merged["Age"], 
                            merged["fixed_aper_exp"],
                            alternative='two-sided')
#TBI
stat, pval =stats.pearsonr(TBI_age_set, TBI_frontal_group_exp_filter["fixed_aper_exp"])


# %%''' investigate errors'''
compute_pointwise_error_fm(test_fm, plot_errors=True)




#%% To plot the CH locations from the EGI guide


# ch_locations=pd.read_csv('../../Data/EGI_129_ch_location.csv')
# ch_dict=dict(zip(ch_locations.labels, ch_locations.Number))
# [ch_dict.pop(key) for key in ['E130','E131','E132']]

# tt = ch_locations.copy()
# tt = tt[tt['Number'] <= 129]

# dict_dig={}

# for row in range(tt.shape[0]):
    
#     lab = str(row)
#     x_pt = float(tt.loc[row]['X'])
#     y_pt = float(tt.loc[row]['Y'])
#     z_pt = float(tt.loc[row]['Z'])
    
#     dict_dig[lab] = np.array([x_pt, y_pt, z_pt])
    
# %% NOT NEEDED CODE
'''ONE FOOOF MEAN CHANNELS'''
test_fm=FOOOF(peak_width_limits=[2, 10], 
              max_n_peaks=6,
              min_peak_height=0.1,
              aperiodic_mode="knee")
freq_range = [5, 40]

# Report: fit the model, print the resulting parameters, and plot the reconstruction
test_fm.report(freqs, power_spectrum_dict.mean(axis=0), freq_range)
plt.show()
results_test=test_fm.get_results()
#%% #montage fun- NOT NEEDED
# my_montage = mne.channels.make_dig_montage(
#     ch_pos=dict_dig, 
#     # nasion=None, 
#     # lpa=None, 
#     # rpa=None, 
#     # hsp=None, 
#     # hpi=None, 
#     coord_frame='head')

#hydra_mon=mne.channels.make_standard_montage('GSN-HydroCel-129')
#%% ''' NOT NEEDED CODE'''
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

# freqs=[]
# for item in working_dict["freq"]:
#     freqs.append(item[0])

# freqs=np.array(freqs)
# #%%
# spectrum=np.power(10,data1)
# fm = FOOOFGroup(peak_width_limits=[0.5, 8], max_n_peaks=6, min_peak_height=0.2, aperiodic_mode='knee')
# fm.report(freqs, spectrum, [5, 40])

#for sub in avg_eeg_spectra_tbi_sess1.keys():
