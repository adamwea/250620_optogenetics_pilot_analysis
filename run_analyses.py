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

#
plot_raster_units(
    #h5_file_path=REC_PATH,
    analyzer_output_dir=ANALYZER_PATH,
    well_id="well000",
    #start_time=0,
    #duration=duration,
    #max_channels=200,
    markers=pocs,
    output_file=f"RUN{RUN}_well000_raster.png",
    output_dir=ANALYSIS_OUTPUT_DIR,
    job_kwargs=job_kwargs,
    verbose=True,
)
print("Raster plot saved successfully.")

#
# plot traces with spikes identified
#print(f"Plotting traces with spikes for well000 starting at {start_time} seconds for {duration} seconds...")
print("Plotting traces with spikes for well000...")
selected_units, selected_channels = plot_traces_and_spikes(
    h5_file_path=REC_PATH,
    analyzer_output_dir=ANALYZER_PATH,
    well_id="well000",
    #start_time=start_time,
    #duration=duration,
    start_time=0,  # Start at the beginning of the recording
    duration=600,  # 10 minutes = 600 seconds
    max_channels=4,
    #selected_units=selected_units,  # Use previously selected units if available
    pocs=pocs,  # Points of change for stimulation onset
    output_dir=ANALYSIS_OUTPUT_DIR,
    #output_file=f"well000_traces_with_spikes_{start_time}.png"
    output_file=f"RUN{RUN}_well000_traces_with_spikes.png",
)

    #
# plot traces with spikes identified
#print(f"Plotting traces with spikes for well000 starting at {start_time} seconds for {duration} seconds...")
print("Plotting traces and sorting by distance for reviewing distance dependent effect of stimulation on firing rates...")
selected_units, selected_channels = plot_traces_and_spikes(
    h5_file_path=REC_PATH,
    analyzer_output_dir=ANALYZER_PATH,
    well_id="well000",
    #start_time=start_time,
    #duration=duration,
    start_time=0,  # Start at the beginning of the recording
    duration=600,  # 10 minutes = 600 seconds
    max_channels=25,
    #selected_units=selected_units,  # Use previously selected units if available
    pocs=pocs,  # Points of change for stimulation onset
    sort_dist=True,  # Sort by distance
    fig_size=(24, 20),  # Set figure size to 24x20 inches
    output_dir=ANALYSIS_OUTPUT_DIR,
    #output_file=f"well000_traces_with_spikes_{start_time}.png"
    output_file=f"RUN{RUN}_well000_traces_with_spikes_sort_dist.png",
)    



job_kwargs = {
'n_jobs': 24,
#'n_jobs': 256, # Adjust based on your system's capabilities
'verbose': True,
'progress_bar': True,
#'peak_sign':"pos"
#'recompute': True,  # Recompute all extensions
}

start_times = [0, 162, 411]
end_times = [162, 411, 600] # durations between points of change
durations = [end - start for start, end in zip(start_times, end_times)]
selected_units, selected_channels = None, None
for start_time, duration in zip(start_times, durations):
    
    
    print(f"Plotting traces for well000 starting at {start_time} seconds for {duration} seconds...")
    #
    plot_channel_spike_heatmaps(
        #h5_file_path=REC_PATH,
        analyzer_output_dir=ANALYZER_PATH,
        well_id="well000",
        #start_time=0,
        #duration=300,
        #duration=600,         # 10 min = 600 seconds
        #max_channels=200,
        start_time=start_time,
        duration=duration,
        output_file=f"RUN{RUN}_well000_heatmap_{start_time}.png",
        #recompute=True,
        output_dir=ANALYSIS_OUTPUT_DIR,
        contrast=True,  # Set to True to use high contrast colormap
        job_kwargs=job_kwargs,
        verbose=True,
    )
    
