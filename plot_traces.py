import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
from MEA_Analysis.MEAProcessingLibrary import mea_processing_library as mea
import os


import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
from MEA_Analysis.MEAProcessingLibrary import mea_processing_library as mea
import os


def plot_traces_with_spikes_per_well(
    h5_file_path: str,
    analyzer_output_dir: str,
    well_id: str,
    start_time: float = 0.0,
    duration: float = 1.0,
    max_channels: int = 64,
    output_file: str = None,
):
    """
    Plot one subplot per channel with raw traces and overlaid spike times for a given well.
    
    Args:
        h5_file_path (str): Path to the .h5 recording file.
        analyzer_output_dir (str): Path to the analyzer output base directory.
        well_id (str): The well ID (e.g., 'A1') to plot.
        start_time (float): Start time (in seconds) for the trace.
        duration (float): Duration (in seconds) of the trace window.
        max_channels (int): Max number of channels to display.
        output_file (str): Path to save the figure (optional).
    """
    # Load recording
    _, recordings, _, _ = mea.load_recordings(h5_file_path)
    if well_id not in recordings:
        print(f"⚠️ Well {well_id} not found in recording.")
        return
    recording = recordings[well_id]
    segment = recording[0]  # assuming one segment

    # Load sorting analyzer
    sa_path = os.path.join(analyzer_output_dir, well_id)
    if not os.path.exists(sa_path):
        print(f"⚠️ Analyzer directory for well {well_id} not found at: {sa_path}")
        return
    sa = si.load_sorting_analyzer(sa_path)
    sorting = sa.sorting
    sampling_rate = segment.get_sampling_frequency()

    # Time window
    start_frame = int(start_time * sampling_rate)
    end_frame = int((start_time + duration) * sampling_rate)
    times = np.arange(start_frame, end_frame) / sampling_rate

    # Traces
    traces = segment.get_traces(start_frame=start_frame, end_frame=end_frame)
    num_channels = min(traces.shape[1], max_channels)
    traces = traces[:, :num_channels]
    
    # Global y-axis limits
    channel_mins = np.min(traces, axis=0)
    channel_maxs = np.max(traces, axis=0)
    global_ymin = np.min(channel_mins)
    global_ymax = np.max(channel_maxs)
    y_margin = 0.05 * (global_ymax - global_ymin)
    global_ymin -= y_margin
    global_ymax += y_margin
    
    # All spike times from all units
    spike_times_all_units = []
    for unit_id in sorting.get_unit_ids():
        spike_frames = sorting.get_unit_spike_train(unit_id, start_frame=start_frame, end_frame=end_frame)
        spike_times_all_units.extend(spike_frames / sampling_rate)
    spike_times_all_units = np.array(sorted(spike_times_all_units))

    # Plot one subplot per channel
    fig, axs = plt.subplots(num_channels, 1, figsize=(12, 2 * num_channels), sharex=True)
    if num_channels == 1:
        axs = [axs]

    for ch in range(num_channels):
        ax = axs[ch]
        trace = traces[:, ch]
        ax.plot(times, trace, color='black', linewidth=0.6)

        # Interpolate spike amplitude from trace at spike times
        if len(spike_times_all_units) > 0:
            spike_idxs = ((spike_times_all_units - start_time) * sampling_rate).astype(int)
            valid_mask = (spike_idxs >= 0) & (spike_idxs < trace.shape[0])
            spike_idxs = spike_idxs[valid_mask]
            spike_times = spike_times_all_units[valid_mask]
            spike_amps = trace[spike_idxs]
            #ax.scatter(spike_times, spike_amps, color='red', s=10)

        ax.set_ylabel(f"Ch {ch}")
        ax.set_ylim(global_ymin, global_ymax)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        print(f"Channel {ch} - Min: {channel_mins[ch]:.2f}, Max: {channel_maxs[ch]:.2f}")

    axs[-1].set_xlabel("Time (s)")
    plt.suptitle(f"Traces with Spikes — Well {well_id}", y=1.02)
    plt.tight_layout()

    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"✅ Plot saved to {output_file}")
    else:
        plt.show()

    
REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/data.raw.h5'
SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/sorted/well000/sorter_output'
ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/analyzer/'

#start_time = 15  # seconds
#for start_time in [0, 5, 10, 15, 20, 25, 30]:
for start_time in [180, 185, 190, 195, 200, 205, 210]:
    print(f"Plotting traces for well000 starting at {start_time} seconds...")
    
    # Call the function to plot traces with spikes
    plot_traces_with_spikes_per_well(
        # h5_file_path="/data/run123.h5",
        # sorted_output_dir="/data/sorted_output",
        # analyzer_output_dir="/data/analyzer_output",
        h5_file_path=REC_PATH,
        #sorted_output_dir=SORT_PATH,
        analyzer_output_dir=ANALYZER_PATH,
        well_id="well000",
        #start_time=5.0,
        start_time=start_time,
        duration=5,
        max_channels=200,
        output_file=f"well000_traces_with_spikes_{start_time}.png"
)