from analysis_helpers import *

# Run to set up the environment for running the analysis.
'''
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
'''

# 10 Hz Network Scan, Run 30
# REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/data.raw.h5'
# SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/sorted/well000/sorter_output'
# ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/analyzer/'
# RUN = 30

# 10 Hz NU4, Run 32
REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000032/data.raw.h5'
SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000032/sorted/well000/sorter_output'
ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000032/analyzer/' 
RUN = 32
ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_32'
STIM_RATE = 10  # Hz
POCS = [162, 411]  # Points of change in seconds
    
# make output_dir if it does not exist
if not os.path.exists(ANALYSIS_OUTPUT_DIR):
    os.makedirs(ANALYSIS_OUTPUT_DIR)

#assert rec_path data exists
if not os.path.exists(REC_PATH):
    raise FileNotFoundError(f"Recording file does not exist: {REC_PATH}")

# assert sort_path data exists
if not os.path.exists(SORT_PATH):
    raise FileNotFoundError(f"Sorting output directory does not exist: {SORT_PATH}")

# assert analyzer_path data exists
if not os.path.exists(ANALYZER_PATH):
    raise FileNotFoundError(f"Analyzer output directory does not exist: {ANALYZER_PATH}")

# veify HAVE_NUMBA is True - needed for quality metrics apparently...
from spikeinterface.curation.curation_tools import HAVE_NUMBA
if not HAVE_NUMBA:
    raise ImportError("Numba is not installed. Please install it to run this analysis.")

# approximate points of change caused by stimulation onset in seconds
pocs = POCS
# plot example stim rate.
fig, ax = plt.subplots(figsize=(24, 6))
stim_dur = pocs[1] - pocs[0]  # duration of stimulation in seconds
stim_onsets = np.arange(pocs[0], pocs[1], 1/STIM_RATE)  # 10 minutes = 600 seconds
#x_range = (0, 600)  # 10 minutes = 600 seconds
x_range = (pocs[0], pocs[1])  # 2 minutes 30 seconds
y_values = np.zeros_like(stim_onsets)
y_values[::int(1/STIM_RATE * 100)] = 1  # Set stimulation on periods to 1
ax.plot(stim_onsets, y_values, 'g-', linewidth=2, label='Stimulation On/Off')
#ax.plot(np.arange(0, 600, 1/STIM_RATE), np.ones(600) * 0.5, 'r-', linewidth=2)
#ax.set_title(f"Stimulation Rate: {STIM_RATE} Hz")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Stimulation On/Off")
ax.set_ylim(-0.1, 1.1)
#ax.set_xlim(0, 600)
ax.set_xlim(x_range)
ax.grid(True)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
#ax.axvline(162, color='blue', linewidth=0.5, linestyle='--', label='POC 1')
#ax.axvline(411, color='green', linewidth=0.5, linestyle='--', label='POC 2')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_OUTPUT_DIR, f"RUN{RUN}_stimulation_rate_demo.png"))
plt.close(fig)