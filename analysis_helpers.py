import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.full as si
from MEA_Analysis.MEAProcessingLibrary import mea_processing_library as mea
import os
from scipy.signal import correlate, find_peaks
from matplotlib.colors import LogNorm
from MEA_Analysis.NetworkAnalysis_aw.plot_network_activity import plot_raster
import numpy as np
from scipy.special import gammaln
from matplotlib.ticker import AutoLocator

import numpy as np
import cvxpy as cp

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def plot_unit_locations(
    analyzer_output_dir: str,
    well_id: str,
    title: str = None,
    output_file: str = None,
    output_dir: str = None,
    verbose: bool = False
    ) -> None:
    """
    Plot unit locations on the chip (4200×2100 nm), overlaying channel locations subtly.
    Always draw the nominal chip boundary; mark out-of-bounds units beyond it.
    Maintains chip aspect ratio and overlays unit count.

    Parameters
    ----------
    analyzer_output_dir : str
        Directory where sorting analyzer folders are stored.
    well_id : str
        Identifier of the recording well to process.
    title : str, optional
        Title for the plot.
    output_file : str, optional
        Filename for saving the figure. If None, shows interactively.
    output_dir : str, optional
        Directory to save the figure.
    verbose : bool, optional
        If True, prints processing messages.
    """
    chip_w, chip_h = 4200, 2100
    def log(msg: str):
        if verbose:
            print(msg)

    # Load analyzer
    analyzer_path = os.path.join(analyzer_output_dir, well_id)
    log(f"Loading sorting analyzer from {analyzer_path}...")
    sorting_analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = sorting_analyzer.sorting

    # Ensure unit_locations
    if 'unit_locations' not in sorting_analyzer.extensions:
        log("Computing unit_locations extension...")
        sorting_analyzer.compute('unit_locations')
    unit_loc_ext = sorting_analyzer.extensions['unit_locations']
    unit_data = unit_loc_ext.data.get('unit_locations', unit_loc_ext.data)

    # Parse coordinates
    if isinstance(unit_data, dict):
        unit_ids = list(unit_data.keys())
        coords = [unit_data[uid] for uid in unit_ids]
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
    else:
        xs, ys = unit_data[:, 0], unit_data[:, 1]
        unit_ids = sorting.get_unit_ids()
    n_units = len(xs)
    log(f"Plotting {n_units} units")

    # Channel locations
    chan_locs = sorting_analyzer.get_channel_locations()
    chan_xs, chan_ys = chan_locs[:, 0], chan_locs[:, 1]

    # Setup figure with chip aspect ratio
    fig_w = 8
    fig_h = fig_w * (chip_h / chip_w)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Scatter channels and units
    ax.scatter(chan_xs, chan_ys, s=5, color='gray', alpha=0.5, label='Channels')
    ax.scatter(xs, ys, s=10, color='red', edgecolor='k', alpha=0.8, label='Units')

    # Draw chip boundary rectangle
    rect = plt.Rectangle((0, 0), chip_w, chip_h,
                         fill=False, edgecolor='red', linestyle='--', linewidth=2)
    ax.add_patch(rect)

    # Determine plot limits to include any outliers
    x_min = min(0, np.min(chan_xs), np.min(xs))
    x_max = max(chip_w, np.max(chan_xs), np.max(xs))
    y_min = min(0, np.min(chan_ys), np.min(ys))
    y_max = max(chip_h, np.max(chan_ys), np.max(ys))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Equal scaling so chip appears with correct aspect
    ax.set_aspect('equal')

    # Overlay unit count in corner
    ax.text(0.95, 0.95, f"Units: {n_units}",
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    if title:
        ax.set_title(title)
    #ax.legend(loc='best')
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()

    # Save or show
    if output_file:
        out_dir = output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        outpath = os.path.join(out_dir, output_file)
        plt.savefig(outpath, dpi=300)
        log(f"Saved figure to {outpath}")
    else:
        plt.show()

def _plot_unit_locations(
    analyzer_output_dir: str,
    well_id: str,
    title: str = None,
    output_file: str = None,
    output_dir: str = None,
    verbose: bool = False
    ) -> None:
    """
    Plot unit locations on the chip, overlaying channel locations subtly,
    and display the number of units plotted.

    Parameters
    ----------
    analyzer_output_dir : str
        Directory where sorting analyzer folders are stored.
    well_id : str
        Identifier of the recording well to process.
    title : str, optional
        Title for the plot.
    output_file : str, optional
        Filename for saving the figure. If None, shows interactively.
    output_dir : str, optional
        Directory to save the figure.
    verbose : bool, optional
        If True, prints processing messages.
    """
    def log(msg: str):
        if verbose:
            print(msg)

    # Load analyzer
    analyzer_path = os.path.join(analyzer_output_dir, well_id)
    log(f"Loading sorting analyzer from {analyzer_path}...")
    sorting_analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = sorting_analyzer.sorting

    # Ensure unit_locations extension is available
    if 'unit_locations' not in sorting_analyzer.extensions:
        log("Computing unit_locations extension...")
        sorting_analyzer.compute('unit_locations')
    unit_loc_ext = sorting_analyzer.extensions['unit_locations']
    unit_data = unit_loc_ext.data.get('unit_locations', unit_loc_ext.data)

    # Parse unit coordinates
    if isinstance(unit_data, dict):
        unit_ids = list(unit_data.keys())
        coords = [unit_data[uid] for uid in unit_ids]
        xs = np.array([c[0] for c in coords])
        ys = np.array([c[1] for c in coords])
    else:
        xs = unit_data[:, 0]
        ys = unit_data[:, 1]
        unit_ids = sorting.get_unit_ids()

    n_units = len(xs)
    log(f"Plotting {n_units} units")

    # Get channel locations
    chan_locs = sorting_analyzer.get_channel_locations()
    chan_xs = chan_locs[:, 0]
    chan_ys = chan_locs[:, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(chan_xs, chan_ys, s=10, color='gray', alpha=0.5, label='Channels')
    ax.scatter(xs, ys, s=100, color='red', edgecolor='k', alpha=0.8, label='Units')

    # Overlay unit count text
    ax.text(
        0.95, 0.95,
        f"Units: {n_units}",
        transform=ax.transAxes,
        ha='right', va='top', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    if title:
        ax.set_title(title)
    ax.legend(loc='best')
    plt.tight_layout()

    if output_file:
        out_dir = output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        outpath = os.path.join(out_dir, output_file)
        plt.savefig(outpath, dpi=300)
        log(f"Saved figure to {outpath}")
    else:
        plt.show()

def plot_epoch_spike_counts(
    analyzer_output_dir: str,
    well_id: str,
    start_times: list,
    durations: list,
    labels: list = None,
    title: str = None,
    recompute: bool = False,
    job_kwargs: dict = None,
    output_file: str = None,
    output_dir: str = None,
    verbose: bool = False
    ) -> None:
    """
    Generate a bar-plot comparing spike counts during specified epochs.

    - Removes spacing between bars and assigns unique colors.
    - If exactly 3 epochs, only plots epoch 1 and 3.
    - Performs a t-test on spike times between epoch 1 and 3, annotates significance.

    Parameters
    ----------
    analyzer_output_dir : str
        Directory where sorting analyzer folders are stored.
    well_id : str
        Identifier of the recording well to process.
    start_times : list of float
        List of start times (in seconds) for each epoch.
    durations : list of float
        Corresponding list of durations (in seconds) for each epoch.
    labels : list of str, optional
        Labels for each epoch. If None, epochs are numbered.
    title : str, optional
        Title for the plot.
    recompute : bool, optional
        If True, force recomputation of spike_times extension.
    job_kwargs : dict, optional
        Additional kwargs passed to sorting_analyzer.compute().
    output_file : str, optional
        Filename for saving the figure. If None, shows interactively.
    output_dir : str, optional
        Directory to save the figure (and CSV if provided).
    verbose : bool, optional
        If True, print detailed processing messages.
    """
    assert len(start_times) == len(durations), "start_times and durations must be same length"
    n_epochs = len(start_times)

    # Determine which epochs to plot
    if n_epochs == 3:
        plot_indices = [0, 2]
    else:
        plot_indices = list(range(n_epochs))
    plot_count = len(plot_indices)

    # Prepare labels
    if labels is None:
        labels = [f"Epoch {i+1}" for i in range(n_epochs)]
    if len(labels) != n_epochs:
        raise ValueError("labels must have same length as start_times and durations")

    def log(msg: str):
        if verbose:
            print(msg)

    # Load sorting analyzer
    analyzer_path = os.path.join(analyzer_output_dir, well_id)
    log(f"Loading sorting analyzer from {analyzer_path}...")
    sorting_analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = sorting_analyzer.sorting
    sr = sorting_analyzer.sampling_frequency

    # Get raw spike times (in seconds)
    all_trains = sorting.get_all_spike_trains()[0][0] / sr  # convert to seconds
    times = all_trains

    # Compute counts and collect spike times per epoch
    counts = []
    epoch_spikes = []
    for idx in plot_indices:
        t0 = start_times[idx]
        dur = durations[idx]
        t1 = t0 + dur
        mask = (times >= t0) & (times < t1)
        epoch_times = times[mask]
        count = len(epoch_times)
        counts.append(count)
        epoch_spikes.append(epoch_times)
        log(f"Epoch {labels[idx]}: {count} spikes ({t0:.2f}-{t1:.2f}s)")

    # Plot bar chart with no spacing and unique colors
    fig, ax = plt.subplots(figsize=(3, 5))
    x = np.arange(plot_count)
    bar_width = 1.0  # full width, no padding
    # generate distinct colors
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(plot_count)]
    bars = ax.bar(x, counts, width=bar_width, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[i] for i in plot_indices], rotation=45, ha='right')
    ax.set_ylabel("Spike count")
    if title:
        ax.set_title(title)

    # Significance test between first and last epoch (if exactly two plotted)
    if plot_count == 2:
        # perform two-sample t-test on spike times
        stat, pval = ttest_ind(epoch_spikes[0], epoch_spikes[1], equal_var=False)
        # determine annotation
        if pval < 0.001:
            star = '***'
        elif pval < 0.01:
            star = '**'
        elif pval < 0.05:
            star = '*'
        else:
            star = None
        if star:
            # draw line
            y_max = max(counts)
            y_line = y_max * 1.05
            x1, x2 = x[0], x[1]
            # ax.plot([x1, x2], [y_line, y_line], lw=1.5, color='black')
            # ax.text((x1 + x2) / 2, y_line * 1.02, star, ha='center', va='bottom')
            
            # adjust y limit to fit the annotation
            ax.set_ylim(0, y_line * 1.1)

    plt.tight_layout()
    
    #HACK: hardcode y lim for a second.
    print("HACK: Hardcoding y limit")
    ax.set_ylim(0, 75000)  # Adjust as needed for your data

    # Save or show
    if output_file:
        out_dir = output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        outpath = os.path.join(out_dir, output_file)
        plt.savefig(outpath, dpi=300)
        log(f"Figure saved to {outpath}")
    else:
        plt.show()

    # Optionally save counts to CSV
    if output_dir:
        import pandas as pd
        df = pd.DataFrame({
            'label': [labels[i] for i in plot_indices],
            'start_time_s': [start_times[i] for i in plot_indices],
            'duration_s': [durations[i] for i in plot_indices],
            'spike_count': counts
        })
        csv_path = os.path.join(output_dir, f"spike_counts_{well_id}.csv")
        df.to_csv(csv_path, index=False)
        log(f"Counts saved to {csv_path}")

def _plot_epoch_spike_counts(
    analyzer_output_dir: str,
    well_id: str,
    start_times: list,
    durations: list,
    labels: list = None,
    title: str = None,
    recompute: bool = False,
    job_kwargs: dict = None,
    output_file: str = None,
    output_dir: str = None,
    verbose: bool = False
    ) -> None:
    """
    Generate a bar-plot comparing spike counts during specified epochs.

    Parameters
    ----------
    analyzer_output_dir : str
        Directory where sorting analyzer folders are stored.
    well_id : str
        Identifier of the recording well to process.
    start_times : list of float
        List of start times (in seconds) for each epoch.
    durations : list of float
        Corresponding list of durations (in seconds) for each epoch.
    labels : list of str, optional
        Labels for each epoch. If None, epochs are numbered.
    recompute : bool, optional
        If True, force recomputation of sorting extensions.
    job_kwargs : dict, optional
        Additional kwargs passed to sorting_analyzer.compute().
    output_file : str, optional
        Filename for saving the figure. If None, shows interactively.
    output_dir : str, optional
        Directory to save the figure (and CSV if provided).
    verbose : bool, optional
        If True, print detailed processing messages.
    """
    assert len(start_times) == len(durations), "start_times and durations must be same length"
    n_epochs = len(start_times)
    if labels is None:
        labels = [f"Epoch {i+1}" for i in range(n_epochs)]
    elif len(labels) != n_epochs:
        raise ValueError("labels must have same length as start_times and durations")

    def log(msg: str):
        if verbose:
            print(msg)

    # Load sorting analyzer
    analyzer_path = os.path.join(analyzer_output_dir, well_id)
    log(f"Loading sorting analyzer from {analyzer_path}...")
    sorting_analyzer = si.load_sorting_analyzer(analyzer_path)
    sorting = sorting_analyzer.sorting
    sr = sorting_analyzer.sampling_frequency

    # Get spike times (in seconds)
    all_trains = sorting.get_all_spike_trains()[0][0] / sr  # convert to seconds

    # Count spikes per epoch
    counts = []
    for t0, dur in zip(start_times, durations):
        t1 = t0 + dur
        #count = np.sum((times >= t0) & (times < t1))
        count = len(all_trains[(all_trains >= t0) & (all_trains < t1)])
        counts.append(count)
        log(f"Epoch {t0:.2f}-{t1:.2f} s: {count} spikes")

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(n_epochs), counts)
    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Spike count")
    if title:
        ax.set_title(title)
    plt.tight_layout()

    # Save or show
    if output_file:
        save_dir = output_dir if output_dir else os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, output_file)
        plt.savefig(outpath, dpi=300)
        log(f"Figure saved to {outpath}")
    else:
        plt.show()

    # Optionally save counts to CSV
    if output_dir:
        import pandas as pd
        df = pd.DataFrame({
            'label': labels,
            'start_time_s': start_times,
            'duration_s': durations,
            'spike_count': counts
        })
        csv_path = os.path.join(output_dir, f"spike_counts_{well_id}.csv")
        df.to_csv(csv_path, index=False)
        log(f"Counts saved to {csv_path}")

def plot_traces_and_spikes(
    h5_file_path: str,
    analyzer_output_dir: str,
    well_id: str,
    start_time: float = 0.0,
    duration: float = np.inf,
    max_channels: int = 5,
    pocs: list = None,
    sort_dist: bool = False,
    selected_units: list = None,
    fig_size: tuple = (24, 4),
    output_file: str = None,
    output_dir: str = None,
    verbose: bool = True,
    ):
    """
    Plot raster of spike trains for a given recording well.
    
    
    """
    
    def log(msg: str):
        if verbose:
            print(msg)
            
    # Step 1: Load sorting analyzer
    log(f"Loading sorting analyzer from {analyzer_output_dir} for well {well_id}...")
    sorting_analyzer = si.load_sorting_analyzer(os.path.join(analyzer_output_dir, well_id))
    if np.isinf(duration):
        duration = sorting_analyzer.get_total_duration()
    sorting = sorting_analyzer.sorting
    log(f"Total duration of sorting: {sorting_analyzer.get_total_duration():.2f} seconds")
    log(f"Sampling frequency: {sorting_analyzer.sampling_frequency} Hz")
    log(f"Number of units: {len(sorting.get_unit_ids())}")
    log(f"Number of channels: {sorting_analyzer.get_num_channels()}")
    log(f"Start time: {start_time:.2f} seconds, Duration: {duration:.2f} seconds")
    
    # Step 2: Build spiking data for each unit
    log(f"Building spiking data for {well_id}...")
    spiking_data_by_unit = {}
    for unit_id in sorting.get_unit_ids():
        unit_data = {}
        unit_data['spike_times'] = sorting.get_unit_spike_train(
            unit_id,
            start_frame=int(start_time * sorting.get_sampling_frequency()),
            end_frame=int((start_time + duration) * sorting.get_sampling_frequency())
        )
        unit_data['spike_times'] = unit_data['spike_times'] / sorting.get_sampling_frequency()  # convert to seconds
        spiking_data_by_unit[unit_id] = unit_data
    log(f"Found {len(spiking_data_by_unit)} units with spikes in the specified time window.")
    
    
    # Step 2.1: Get Firing Rates
    firing_rates = {}
    for unit in spiking_data_by_unit:
        #duration = sorting_analyzer.get_total_duration()
        dur = duration
        spike_times = spiking_data_by_unit[unit]['spike_times']
        # if pocs, cut all time between the two pocs. This should remove activity during stimulation from consideration for computing firing rates.
        if pocs:
            # # filter spike times to only include those within the specified points of change
            # spike_times = spike_times[(spike_times >= pocs[0]) & (spike_times <= pocs[-1])]
            # filter to exclude spikes within the points of change
            spike_times = spike_times[(spike_times < pocs[0]) | (spike_times > pocs[-1])]
            #duration = duration - (pocs[-1] - pocs[0])  # adjust duration to exclude stimulation period
            dur = dur - (pocs[-1] - pocs[0])  # adjust duration to exclude stimulation period
        if len(spike_times) > 0:
            firing_rate = len(spike_times) / dur
            firing_rates[unit] = firing_rate
    log(f"Firing rates calculated for {len(firing_rates)} units.")
        
    # Step 3: Get example traces.
    # - given max_channels, select a representative distribution of channels spanning firing rates.
    example_traces = {}
    if len(firing_rates) > 0:
        # Select top units by firing rate
        
        # simple method
        # top_units = sorted(firing_rates, key=firing_rates.get, reverse=True)[:max_channels]
        
        # more complex method: get mean and std deviation. Compose a list of units with similar mean and std deviation.
        firing_rate_values = np.array(list(firing_rates.values()))
        mean_fr = np.mean(firing_rate_values)
        std_fr = np.std(firing_rate_values)
        log(f"Mean firing rate: {mean_fr:.2f}, Std deviation: {std_fr:.2f}")
        # Select units within 1 std deviation of the mean
        if selected_units is None:
            top_units = [unit for unit, fr in firing_rates.items() if abs(fr - mean_fr) <= std_fr]
            if len(top_units) > max_channels:
                # sort by firing rate
                top_units = sorted(top_units, key=lambda u: firing_rates[u], reverse=True)
                # splice list into max_channels parts, with equal elements in each part
                #chunks = np.linspace(0, len(top_units), max_channels + 1, dtype=int)
                chunks = []
                chunk_length = len(top_units) // max_channels
                for i in range(max_channels):
                    start = i * chunk_length
                    end = (i + 1) * chunk_length if i < max_channels - 1 else len(top_units)
                    chunks.append(top_units[start:end])            
                
                # Select random sample from each chunk
                np.random.seed(42)
                top_units = [np.random.choice(chunk) for chunk in chunks if len(chunk) > 0]
        else:
            print(f"Using pre-selected units: {selected_units}")
            top_units = selected_units
            
        log(f"Selected {len(top_units)} top units for example traces.")
        
        # Compute amplitudes if needed
        # if 'spike_amplitudes' not in sorting_analyzer.extensions:
        #     log("Computing spike amplitudes...")
        #     sorting_analyzer.compute('spike_amplitudes', peak_sign='pos')
            
        # # 
            
        #spike_amplitudes = sorting_analyzer.extensions['spike_amplitudes'].data['amplitudes']
        # load h5 file to get raw traces from extreemum channels
        log("Loading raw traces for example units...")
        log(f"Loading recordings from {h5_file_path}...")
        _, recs, _, _ = mea.load_recordings(h5_file_path)
        if well_id not in recs:
            raise ValueError(f"Well '{well_id}' not found in recordings.")
        segment = recs[well_id][0]
        sr = segment.get_sampling_frequency()
        total_dur = segment.get_num_frames() / sr
        #log(f"Recording duration: {total_dur:.2f} s")
        
        # get extreemum channels for each unit
        # get corresponding raw trace for that channel
        sorting = sorting_analyzer.sorting
        #templates = sorting_analyzer.extensions['templates']
        from spikeinterface.core import get_template_extremum_channel
        extremum_channel_ids = get_template_extremum_channel(
                sorting_analyzer,
                #unit_id,
                peak_sign='pos',  # or 'neg' depending on your data
                #mode='extremum',
                outputs='id'
            )
        
        #top_extremum_channels = {unit: extremum_channels[unit] for unit in top_units if unit in extremum_channels}
        top_extremum_channels = {}
        for unit in top_units:
            if unit in extremum_channel_ids:
                top_extremum_channels[unit] = extremum_channel_ids[unit]
            else:
                log(f"Warning: Unit {unit} does not have an extremum channel.")
        log(f"Retrieved extremum channels for {len(top_extremum_channels)} top units.")
        
        # get channel location
        #channel_locations = sorting_analyzer.get_channel_locations()
        channel_locations = segment.get_channel_locations(channel_ids=list(top_extremum_channels.values()))
        extremum_channel_locations = {channel_id: channel_locations[i] for i, channel_id in enumerate(top_extremum_channels.values())}
        log(f"Retrieved channel locations for {len(extremum_channel_locations)} channels.")
        
        # get center of chip for later
        all_channel_locations = segment.get_channel_locations()
        max_x = max(all_channel_locations[:, 0])
        max_y = max(all_channel_locations[:, 1])
        center_x = max_x / 2
        center_y = max_y / 2
        center_location = (center_x, center_y)
        log(f"Center of chip: ({center_x:.2f}, {center_y:.2f})")
        
        
        #HACK shift center down to investigate something.
        # Unclear where the optrode is shining.
        center_y = max_y/5 # shift center down to 20% of max_y
        center_location = (center_x, center_y)
        log(f"HACK: Adjusted center of chip: ({center_x:.2f}, {center_y:.2f})")
        
        
        # get raw traces for these channels
        # spikeinterface.core.get_template_extremum_channel(
            # templates_or_sorting_analyzer, peak_sign: neg' | 'pos' | 'both = 'neg', mode: extremum' | 'at_index' | 
            # 'peak_to_peak = 'extremum', outputs: id' | 'index = 'id')
        raw_traces = segment.get_traces(
            channel_ids=list(top_extremum_channels.values()),
            start_frame=int(start_time * sr),
            end_frame=int((start_time + duration) * sr),
            return_scaled=True,
        )
        raw_traces = raw_traces.T  # transpose to have channels as rows
        trace_dict = {}
        for i, channel_id in enumerate(top_extremum_channels.values()):
            unit_id = top_units[i]
            trace_dict[(unit_id, channel_id)] = raw_traces[i] / sr  # convert to seconds
        example_traces = trace_dict
        log(f"Retrieved raw traces for {len(trace_dict)} channels.")
        
        # sort by distance if sort_dist is True
        if sort_dist:
            # sort by distance from center
            distances = {
                (channel_id): np.sqrt(
                    (location[0] - center_x) ** 2 + (location[1] - center_y) ** 2
                )
                for (channel_id), location in extremum_channel_locations.items()
            }
            sorted_distances = np.argsort(list(distances.values()))
            # apply sorted indices to example_traces
            example_traces = {(unit_id, channel_id): example_traces[(unit_id, channel_id)]
                              for i, (unit_id, channel_id) in enumerate(sorted(example_traces.keys(), key=lambda x: distances[x[1]]))}
            log("Sorted example traces by distance from center of chip.")

        # get y axis limits for the plot
        if len(example_traces) > 0:
            y_min = min(np.min(trace) for trace in example_traces.values())* 0.9  # 10% below minimum
            y_max = max(np.max(trace) for trace in example_traces.values())* 1.1  # 10% above maximum
        else:
            y_min = 0
            y_max = 1
        #log(f"Retrieved example traces for {len(example_traces)} units.")
        log(f"Y-axis limits for traces: {y_min:.2f} to {y_max:.2f}")
        
        # x-lim based on start_time and duration
        #duration = sorting_analyzer.get_total_duration()
        x_lim = (start_time, start_time + duration)
        
        
        # plot example traces
        num_traces = len(example_traces)
        #fig, axs = plt.subplots(num_traces, 1, figsize=(24, 4), sharex=True)
        fig, axs = plt.subplots(
            nrows=num_traces, ncols=1,
            figsize=fig_size,
            sharex=True,
            constrained_layout=True
        )
        # ensure axs is always iterable
        if num_traces == 1:
            axs = [axs]

        for idx, ((unit_id, channel_id), trace) in enumerate(example_traces.items()):
            ax = axs[idx]
            
            # channel location
            channel_location = extremum_channel_locations.get(channel_id)
            
            # compute distance from center of chip
            distance_from_center = np.sqrt(
                (channel_location[0] - center_x) ** 2 +
                (channel_location[1] - center_y) ** 2
            )
            
            # time axis in seconds
            fs = sorting.get_sampling_frequency()
            x_trace = np.arange(len(trace)) / fs + start_time
            ax.plot(x_trace, trace,
                    color='blue',
                    label=f"Chan {channel_id}, Unit {unit_id}\n"
                          f"Channel loc: {channel_location}\n"
                          f"Distance from center: {distance_from_center:.2f}")

            # overlay relevant spikes
            spike_times = spiking_data_by_unit[unit_id]['spike_times'] # convert to absolute time
            for t in spike_times:
                #t += start_time  # adjust to absolute time
                if start_time <= t < start_time + duration:
                    ax.axvline(x=t,
                            color='red', linestyle='--', linewidth=0.5)
                    
            
            
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel("µV")
            ax.legend(loc='upper right', fontsize='small')
            
            # set x-axis limits
            ax.set_xlim(x_lim)
            
            # remove top & right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if idx != num_traces - 1:
                # interior plots: hide all x‐axis elements
                ax.label_outer()           
                ax.spines['bottom'].set_visible(False)
                
                # remove x-ticks and labels
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.tick_params(axis='x', which='both',                            bottom=False, top=False,
                            labelbottom=False)
            else:
                # bottom plot: show only bottom spine/ticks/labels
                ax.spines['bottom'].set_visible(True)
                ax.xaxis.set_major_locator(AutoLocator())
                ax.tick_params(axis='x', which='both',
                            bottom=True, top=False,
                            labelbottom=True)
                ax.set_xlabel("Time (s)")

        # suptitle + layout
        # fig.suptitle(
        #     f"Example traces for {well_id} from {start_time:.2f} to {start_time + duration:.2f} s",
        #     y=1.02
        # )
        plt.tight_layout()

        # save or show
        if output_file:
            if output_dir:
                output_file = os.path.join(output_dir, output_file)
            plt.savefig(output_file, dpi=300)
            print(f"Example traces plot saved to {output_file}")
        else:
            plt.show()
            
        selected_units, selected_channels = zip(*example_traces.keys())
        log(f"Selected {len(selected_units)} units and {len(selected_channels)} channels for plotting.")
        return selected_units, selected_channels

    log(f"Retrieved example traces for {len(example_traces)} units.")

'''# These functions are a forming idea. No currently in use. 2025-07-30 21:50:16 ===================================='''
def l1_trend_filter(
    f: np.ndarray,
    lam: float,
    solver: str = 'ECOS',
    tol: float = 1e-4,
    verbose: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    1D Total Variation Denoising / L1 Trend Filtering.

    Solves:
        minimize_r 0.5 * ||f - r||_2^2 + lam * ||D r||_1
    where D is the first-difference operator.

    Args:
        f (np.ndarray): 1D data array of length n (e.g., binned firing rates).
        lam (float): Regularization parameter (higher => fewer jumps).
        solver (str): CVXPY solver to use (e.g., 'ECOS', 'OSQP').
        tol (float): Threshold for detecting non-zero jumps in r.
        verbose (bool): If True, CVXPY prints solver output.

    Returns:
        r_est (np.ndarray): Denoised piecewise-constant signal of length n.
        cp_indices (np.ndarray): 1D array of change-point indices where |r[t] - r[t-1]| > tol.
    """
    n = f.size
    # First-difference operator D: shape (n-1, n)
    D = np.diff(np.eye(n), axis=0)

    # CVXPY variable
    r = cp.Variable(n)
    # Objective: data fidelity + TV penalty
    objective = cp.Minimize(
        0.5 * cp.sum_squares(f - r)
        + lam * cp.norm1(D @ r)
    )
    prob = cp.Problem(objective)
    prob.solve(solver=solver, verbose=verbose)

    if r.value is None:
        raise RuntimeError("Solver failed to find a solution")

    r_est = r.value
    # Find jumps larger than tol
    diffs = np.diff(r_est)
    cp_indices = np.where(np.abs(diffs) > tol)[0] + 1

    return r_est, cp_indices

def get_tv_change_points(
    spike_times: np.ndarray,
    bin_size: float = 5.0,
    max_time: float = None,
    lam: float = 10.0,
    tol: float = 1e-4,
    solver: str = 'ECOS',
    ) -> dict:
    """
    Bin spike_times, then run ℓ1 trend filtering to find piecewise-constant segments.

    Args:
        spike_times (np.ndarray): 1D array of spike times (seconds).
        bin_size (float): Width of time bins (seconds).
        max_time (float): If set, total duration to bin (seconds).
        lam (float): TV regularization weight (higher → fewer jumps).
        tol (float): Jump detection threshold on |r[t] - r[t-1]|.
        solver (str): cvxpy solver name.

    Returns:
        dict:
          'rates'          : np.ndarray of length T, the denoised rate r_est;
          'cp_bins'        : np.ndarray of int, bin indices where jumps occur;
          'bin_edges'      : np.ndarray of length T+1, the edges used for histogram.
    """
    # 1) Determine bin edges
    if max_time is None:
        max_time = spike_times.max()
    edges = np.arange(0., max_time + bin_size, bin_size)

    # 2) Histogram to get counts per bin
    counts, _ = np.histogram(spike_times, bins=edges)

    # 3) (Optionally) convert counts → rate per second
    rates = counts / bin_size

    # 4) TV-denoise
    r_est, cp_bins = l1_trend_filter(
        f=rates,
        lam=lam,
        solver=solver,
        tol=tol
    )

    return {
        'rates':    r_est,
        'cp_bins':  cp_bins,
        'bin_edges': edges
    }

def bocd_poisson(
    counts: np.ndarray,
    hazard: float = 1/60, # expecting roughly 2 changes in 10 minutes. with 5 second bins, this is 2/120 = 1/60
    alpha0: float = 20,
    beta0: float = 10,
    threshold: float = 0.5
    ) -> dict:
    """
    Bayesian Online Change-Point Detection for Poisson rate data.

    Args:
        counts (np.ndarray): 1D array of non-negative integer spike counts per bin.
        hazard (float): Constant hazard rate H = P(cp at any bin).
        alpha0 (float): Prior Gamma shape parameter.
        beta0 (float): Prior Gamma rate parameter.
        threshold (float): Probability threshold for flagging a change-point.

    Returns:
        dict: {
            'run_length_probs': np.ndarray of shape (T, T+1),
                run_length_probs[t, r] = P(r_t = r | x_{1:t}),
            'cp_probs': np.ndarray of shape (T,),
                cp_probs[t] = P(change-point at t | x_{1:t}),
            'change_points': list of int,
                indices t where cp_probs[t] >= threshold
        }
    """
    T = len(counts)
    # Allocate run-length probability matrix: rows=time, cols=run-length
    R = np.zeros((T, T+1))
    R[0, 0] = 1.0  # at t=0, run-length 0 with prob 1

    # Posterior parameters for each possible run-length
    alphas = [alpha0]
    betas = [beta0]

    cp_probs = np.zeros(T)

    # Helper: predictive probability P(x | alpha, beta) under Poisson-Gamma
    def pred_prob(x, alpha, beta):
        # p(x) = Gamma(alpha + x) / (Gamma(alpha) * x!) * (beta**alpha) / ((beta+1)**(alpha + x))
        num = np.exp(gammaln(alpha + x) - gammaln(alpha) - gammaln(x + 1))
        return num * (beta ** alpha) / ((beta + 1) ** (alpha + x))

    for t in range(1, T):
        x = counts[t]
        pred_probs = np.array([pred_prob(x, a, b) for a, b in zip(alphas, betas)])

        # Compute growth probabilities (no change): r_t = r_{t-1} + 1
        growth = R[t-1, :t] * pred_probs * (1 - hazard)
        # Compute change probabilities: r_t = 0
        cp = (R[t-1, :t] * pred_probs * hazard).sum()

        # Update run-length distribution at time t
        R[t, 1 : t+1] = growth
        R[t, 0] = cp
        # Normalize
        R[t] /= R[t].sum()

        # Record cp probability
        cp_probs[t] = R[t, 0]

        # Update posterior params for next step
        new_alphas = [alpha0 + x]
        new_betas = [beta0 + 1]
        for r in range(1, t+1):
            new_alphas.append(alphas[r-1] + x)
            new_betas.append(betas[r-1] + 1)
        alphas, betas = new_alphas, new_betas

    #change_points = list(np.where(cp_probs >= threshold)[0])
    mode_run = R.argmax(axis=1)       # for each t, which r maximizes P(r_t=r)
    change_points = np.where(mode_run == 0)[0]
    return {
        'run_length_probs': R,
        'cp_probs': cp_probs,
        'change_points': change_points
    }

def get_bocd_change_points(
    spike_times: np.ndarray,
    bin_size: float = 5,
    max_time: float = None,
    **bocd_kwargs
    ) -> dict:
    """
    Given spike times, bin into counts and detect change-points via BOCPD.

    Args:
        spike_times (np.ndarray): 1D array of spike times (seconds).
        bin_size (float): Width of time bins (seconds), default 1s.
        max_time (float): If set, total duration to bin (seconds).
        bocd_kwargs: Passed to bocd_poisson (hazard, alpha0, beta0, threshold).

    Returns:
        dict: output from bocd_poisson on binned spike counts.
    """
    if max_time is None:
        max_time = spike_times.max()
    # Bin edges from 0 to max_time with step bin_size
    edges = np.arange(0, max_time + bin_size, bin_size)
    counts, _ = np.histogram(spike_times, bins=edges)
    return bocd_poisson(counts, **bocd_kwargs)

'''# ================================'''

def plot_raster_units(
    #h5_file_path: str,
    analyzer_output_dir: str,
    well_id: str,
    # start_time: float = 0.0,
    # duration: float = 1.0,
    start_time: float = 0.0,
    duration: float = np.inf,
    #max_channels: int = 200,
    markers: list = None,
    title: str = None,
    output_file: str = None,
    output_dir: str = None,
    #job_kwargs: dict = None,
    verbose: bool = False,
    ):
    """Plot raster of spike trains for a given recording well.
    Parameters
    ----------
    h5_file_path : str
        Path to the HDF5 file containing raw recordings.
    analyzer_output_dir : str
        Directory where sorting analyzer folders are stored.
    well_id : str
        Identifier of the recording well to process.
    start_time : float, optional
        Start time in seconds to slice the recording (default: 0.0).
    duration : float, optional
        Duration in seconds of the time window (default: 1.0).
    max_channels : int, optional
        Maximum number of channels to display in the raster plot (default: 200).
    output_file : str, optional
        Path to save the resulting figure. If None, shows interactively.
    output_dir : str, optional
        Directory to save the output file. If None, uses the current directory.
    job_kwargs : dict, optional
        Additional keyword arguments for sorting_analyzer.compute().
    verbose : bool, optional
        If True, print detailed processing messages.
    """
    
    # Step 2: Prep spiking_data_by_channel
    print(f"Building spiking data for {well_id}...")
    sorting_analyzer = si.load_sorting_analyzer(os.path.join(analyzer_output_dir, well_id))
    if np.isinf(duration):
        duration = sorting_analyzer.get_total_duration()
    sorting = sorting_analyzer.sorting
    spiking_data_by_unit = {}
    for unit_id in sorting.get_unit_ids():
        unit_data = {}
        unit_data['spike_times'] = sorting.get_unit_spike_train(unit_id, start_frame=int(start_time * sorting.get_sampling_frequency()), end_frame=int((start_time + duration) * sorting.get_sampling_frequency()))
        unit_data['spike_times'] = unit_data['spike_times'] / sorting.get_sampling_frequency()  # convert to seconds
        spiking_data_by_unit[unit_id] = unit_data
    
    # Get spike trains and plot raster
    fig, ax = plt.subplots(figsize=(12, 3))
    ax = plot_raster(
        ax,
        spiking_data_by_unit,
    )
    #ax.set_title(f"Raster plot for {well_id} from {start_time:.2f} to {start_time + duration:.2f} s")
    if title:
        #title = f"{title} - {start_time:.2f} to {start_time + duration:.2f} s"
        ax.set_title(title)
    ax.set_xlim(start_time, start_time + duration)
    
    # plot vertical dotted lines, red, at markers
    if markers:
        for marker in markers:
            ax.axvline(x=marker, color='red', linestyle='--')

    # save or show the figure
    if output_file:
        if output_dir:
            output_file = os.path.join(output_dir, output_file)
        plt.savefig(output_file, dpi=300)
        print(f"Raster plot saved to {output_file}")
    else:
        plt.show()
    
def plot_channel_spike_heatmaps(
    #h5_file_path: str,
    analyzer_output_dir: str,
    well_id: str,
    start_time: float = 0.0,
    duration: float = 1.0,
    tpl_ms_before: float = 1.0,
    tpl_ms_after: float = 2.0,
    corr_threshold_factor: float = 0.5,
    title: str = None,
    output_file: str = None,
    recompute: bool = False,
    output_dir: str = None,
    contrast: bool = False,
    job_kwargs: dict = None,
    verbose: bool = False,
    ) -> None:
    """
    Plot per-bin spike count heatmap for a given recording window.

    Parameters
    ----------
    h5_file_path : str
        Path to the HDF5 file containing raw recordings.
    analyzer_output_dir : str
        Directory where sorting analyzer folders are stored.
    well_id : str
        Identifier of the recording well to process.
    start_time : float, optional
        Start time in seconds to slice the recording (default: 0.0).
    duration : float, optional
        Duration in seconds of the time window (default: 1.0).
    tpl_ms_before : float, optional
        Milliseconds before a spike to include in waveform (default: 1.0).
    tpl_ms_after : float, optional
        Milliseconds after a spike to include in waveform (default: 2.0).
    corr_threshold_factor : float, optional
        Fraction of maximum correlation to use as detection threshold.
    output_file : str, optional
        Path to save the resulting figure. If None, shows interactively.
    recompute : bool, optional
        If True, force recomputation of sorting extensions (default: False).
    output_dir : str, optional
        Directory to save computed CSV metrics. If None, metrics are not saved.
    job_kwargs : dict, optional
        Additional keyword arguments for sorting_analyzer.compute().
    verbose : bool, optional
        If True, print detailed processing messages.
    """
    def log(msg: str):
        if verbose:
            print(msg)

    # Step 1: Load recording and slice
    # log(f"Loading recordings from {h5_file_path}...")
    # _, recs, _, _ = mea.load_recordings(h5_file_path)
    # if well_id not in recs:
    #     raise ValueError(f"Well '{well_id}' not found in recordings.")
    # segment = recs[well_id][0]
    # sr = segment.get_sampling_frequency()
    # total_dur = segment.get_num_frames() / sr
    # log(f"Recording duration: {total_dur:.2f} s")

    # # Slice segment if needed
    # t0, t1 = int(start_time * sr), int((start_time + duration) * sr)
    # if start_time != 0 or duration < total_dur:
    #     segment = segment.time_slice(start_time, start_time + duration)
    #     log(f"Segment sliced: {start_time:.2f}–{start_time + duration:.2f} s")

    # Step 2: Load sorting analyzer and compute extensions
    analyzer_path = os.path.join(analyzer_output_dir, well_id)
    log(f"Loading sorting analyzer from {analyzer_path}...")
    sorting_analyzer = si.load_sorting_analyzer(analyzer_path)
    #sorting = sorting_analyzer.sorting
    #sorting.register_recording(segment)

    # Determine which extensions to compute
    extensions = [
        'random_spikes',
        'noise_levels',  
        'waveforms',
        'templates', 
        'spike_amplitudes', 
        'spike_locations',
        'quality_metrics', 
        'unit_locations'
    ]
    kwargs = job_kwargs or {}
    # list_to_remove = []
    # for key, data in sorting_analyzer.extensions.items():
    #     if data is None:
    #         #sorting_analyzer.extensions.pop(key, None)
    #         list_to_remove.append(key)
    # for key in list_to_remove:
    #     sorting_analyzer.extensions.pop(key, None)
    # if any of the extensions are None, something went wrong, clear them and recompute everything
    # if any(sorting_analyzer.extensions.get(ext) is None for ext in extensions):
    #     log("Some extensions are None, clearing and recomputing all.")
    #     #sorting_analyzer.delete_extension()
    #     #sorting_analyzer.extensions.clear()
    #     recompute = True
    for ext in extensions:
        if ext not in sorting_analyzer.extensions or recompute:
            log(f"Computing extension: {ext}")
            try:
                data = sorting_analyzer.extensions.get(ext)
                if data is None: sorting_analyzer.delete_extension(ext)
            except Exception as e:
                log(f"Error deleting extension {ext}: {e}")
                raise e
            if ext == 'waveforms':
                sorting_analyzer.compute(ext, ms_before=tpl_ms_before, ms_after=tpl_ms_after, **kwargs)
            elif ext == 'spike_amplitudes':
                sorting_analyzer.compute(ext, peak_sign='pos', **kwargs)
            else:
                sorting_analyzer.compute(ext, **kwargs)

    # Save quality metrics if requested
    if output_dir:
        metrics = sorting_analyzer.extensions['quality_metrics'].data['metrics']
        fname = f"quality_metrics_{well_id}_{start_time:.2f}.csv"
        outpath = os.path.join(output_dir, fname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        metrics.to_csv(outpath, index=False)
        log(f"Quality metrics saved to {outpath}")

    # Step 3: Retrieve spike locations and amplitudes
    spike_locs = sorting_analyzer.extensions['spike_locations']
    if spike_locs is None:
        log("No spike location data available, aborting.")
        return
    #spike_amps = sorting_analyzer.extensions['spike_amplitudes']
    # if spike_locs is None or spike_amps is None:
    #     log("No spike location or amplitude data available, aborting.")
    #     return

    loc_data = spike_locs.data['spike_locations']
    #amp_data = spike_amps.data['amplitudes']
    
    # Filter spikes within time window
    sorting = sorting_analyzer.sorting
    sr = sorting_analyzer.sampling_frequency
    spk_times = sorting.get_all_spike_trains()[0][0] / sr #seconds
    xs = [i[0] for i in loc_data]  # x coordinates
    ys = [i[1] for i in loc_data]  # y coordinates
    
    # Step 3.1: Filter spikes by time window
    mask = (spk_times >= start_time) & (spk_times < start_time + duration)
    xs = np.array(xs)[mask]
    ys = np.array(ys)[mask]
    #amps = np.array(amp_data)[mask]

    # Step 4: Bin spikes into heatmap grid
    pitch = 17.5  # nm per bin
    # HACK
    pitch = pitch*2.5 # larger bins, better visibility
    #chan_locs = segment.get_channel_locations()
    chan_locs = sorting_analyzer.get_channel_locations()
    all_x = np.concatenate([xs, chan_locs[:, 0]])
    all_y = np.concatenate([ys, chan_locs[:, 1]])
    # x_min, x_max = all_x.min(), all_x.max()
    # y_min, y_max = all_y.min(), all_y.max()
    x_min, x_max = np.nanmin(all_x), np.nanmax(all_x)
    y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
    n_x = int(np.ceil((x_max - x_min) / pitch))
    n_y = int(np.ceil((y_max - y_min) / pitch))
    log(f"Grid: {n_x}×{n_y} bins, pitch={pitch} nm")

    # Compute bin indices
    x_idx = np.floor((xs - x_min) / pitch).astype(int)
    y_idx = np.floor((ys - y_min) / pitch).astype(int)
    valid = (x_idx >= 0) & (x_idx < n_x) & (y_idx >= 0) & (y_idx < n_y)
    #x_idx, y_idx, amps = x_idx[valid], y_idx[valid], amps[valid]
    x_idx, y_idx = x_idx[valid], y_idx[valid]

    # Count spikes per bin
    counts = np.zeros(n_x * n_y, dtype=int)
    for xi, yi in zip(x_idx, y_idx):
        counts[yi * n_x + xi] += 1
    grid = counts.reshape((n_y, n_x))

    # Step 5: Plot
    if not contrast:
        fig, ax = plt.subplots(figsize=(12, 6))
        masked = np.ma.masked_equal(grid, 0)
        norm = LogNorm(vmin=masked.min(), vmax=masked.max()) if masked.count() else None
        im = ax.imshow(masked, origin='lower', extent=[x_min, x_max, y_min, y_max], aspect='auto', norm=norm)
        #ax.set_title(f"Spike count heatmap: {well_id} ({start_time:.2f}–{start_time+duration:.2f} s)")
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        cbar = fig.colorbar(im, ax=ax, label="Spike count/bin")

        # Overlay channel positions
        cx = chan_locs[:, 0]
        cy = chan_locs[:, 1]
        ax.scatter(cx, cy, s=5, edgecolor='k', facecolor='none', alpha=0.5, label='Channels')
        ax.legend(loc='upper right')
        plt.tight_layout()
    else:        
        # Step 5: Plot with dark background & high-contrast map
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        # Use inferno colormap for high contrast
        masked = np.ma.masked_equal(grid, 0)
        norm = LogNorm(vmin=masked.min(), vmax=masked.max()) if masked.count() else None
        im = ax.imshow(
            masked,
            origin='lower',
            extent=[x_min, x_max, y_min, y_max],
            aspect='auto',
            cmap='inferno',
            norm=norm
        )
        # White text and axes for visibility
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        ax.tick_params(colors='white')

        #ax.set_title(f"Spike count heatmap: {well_id} ({start_time:.2f}–{start_time+duration:.2f} s)")
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        cbar = fig.colorbar(im, ax=ax, label="Spike count/bin")
        cbar.outline.set_edgecolor('white'); cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

        # Overlay channels in white
        cx, cy = chan_locs[:,0], chan_locs[:,1]
        ax.scatter(cx, cy, s=10, edgecolor='white', facecolor='none', alpha=0.25, label='Channels')
        legend = ax.legend(loc='upper right', facecolor='black', edgecolor='white')
        for text in legend.get_texts(): text.set_color('white')
        
    # Set title and layout 
    if title:
        title = f"{title} ({start_time:.2f}–{start_time+duration:.2f} s)"
        ax.set_title(title, color='white' if contrast else 'black')

    plt.tight_layout()

    if output_file:
        output_file = os.path.join(output_dir, output_file) if output_dir else output_file
        plt.savefig(output_file, dpi=300)
        log(f"Figure saved to {output_file}")
    else:
        plt.show()

def __plot_channel_spike_heatmaps(
    h5_file_path: str,
    analyzer_output_dir: str,
    well_id: str,
    start_time: float = 0.0,
    duration: float = 1.0,
    tpl_ms_before: float = 1.0,
    tpl_ms_after: float = 2.0,
    corr_threshold_factor: float = 0.5,
    output_file: str = None,
    recompute: bool = False,
    output_dir: str = None,
    job_kwargs: dict = None,
    ):
    """
    1) Load raw traces for the given time window.
    2) (Optional) Extract per‑unit templates.
    3) For each unit & each channel, cross‑correlate raw trace with that unit’s template,
        detect peaks, and record time & amplitude.
    4) Build per‑channel: total spike count & max amplitude.
    5) Plot two heatmaps: spike counts & max amplitudes.
    """

    # --- 1) load recording + raw traces ---
    print(f"Loading recording from {h5_file_path} for well {well_id}...")
    _, recs, _, _ = mea.load_recordings(h5_file_path)
    if well_id not in recs:
        raise ValueError(f"Well {well_id} not found")  # mea.load_recordings → dict of RecordingExtractors 
    segment = recs[well_id][0]
    sr = segment.get_sampling_frequency()
    #probe = segment.get_probe().copy()  # copy the probe to avoid modifying the original
    
    # get total duration of the recording and print it
    total_duration = segment.get_num_frames() / sr
    print(f"Total duration of recording: {total_duration:.2f} seconds")
    
    # # time slice segment
    # chunk = segment.time_slice(start_time, start_time + duration)
    # print(f"Chunk successfully sliced from {start_time}s to {start_time + duration}s")
    # total_duration_chunk = chunk.get_num_frames() / sr
    # print(f"Total duration of chunk: {total_duration_chunk:.2f} seconds")
    # chunk_start_time = chunk.get_start_time()
    # print(f"Chunk start time: {chunk_start_time:.2f} seconds")
    # chunk_end_time = chunk.get_end_time()
    # print(f"Chunk end time: {chunk_end_time:.2f} seconds")
    # if chunk is None:
    #     print(f"⚠️ No data found for well {well_id} from {start_time}s to {start_time + duration}s")
    #     return
    
    chunk = segment
    
    # reassign segment to chunk going forward
    #chunk.set_probe(probe)  # set the probe to the chunk
    channel_locations_pre = segment.get_channel_locations()  # (n_ch, 2)
    segment = chunk

    print(f"Getting traces for well {well_id} from {start_time}s to {start_time + duration}s...")
    t0 = int(start_time * sr)
    t1 = int((start_time + duration) * sr)
    #traces = segment.get_traces(start_frame=t0, end_frame=t1)  # → (n_samples, n_channels)
    channel_locations = segment.get_channel_locations()  # (n_ch, 2) 
    # assert channel locations are the same as before
    if not np.array_equal(channel_locations, channel_locations_pre):
        print("⚠️ Channel locations have changed after slicing the segment.")
        print("Original channel locations:", channel_locations_pre)
        print("New channel locations:", channel_locations)
    #n_samples, n_ch = traces.shape

    # --- 2) optionally extract per‑unit templates ---
    print(f"Loading sorting analyzer for well {well_id}...")
    sorting_analyzer = si.load_sorting_analyzer(os.path.join(analyzer_output_dir, well_id))
    original_duration = sorting_analyzer.get_total_duration()
    print(f"Original sorting analyzer duration: {original_duration:.2f} seconds")
    sorting = sorting_analyzer.sorting  # SortingExtractor
    sorting.register_recording(segment)  # set the recording to the sorting analyzer
    #spk_times = sorting.get_times()  # spike times in seconds
    spk_times = sorting.get_all_spike_trains()  # dict of unit_id → spike times in seconds
    spk_times = spk_times[0][0] / sr
    print(f"Min time: {np.min(spk_times):.2f} seconds, Max time: {np.max(spk_times):.2f} seconds")
    #spk_times = sorting_analyzer.sorting.get_times()
    
    # print(f"Slicing sorting and creating new sorting analyzer for {well_id} from {start_time}s to {start_time + duration}s...")
    # sorting = sorting_analyzer.sorting  # SortingExtractor
    # probe = sorting_analyzer.get_probe()  # ProbeExtractor
    # sorting_slice = sorting_analyzer.sorting.time_slice(start_time, start_time + duration)
    # segment.set_probe(probe)  # set the probe to the segment
    # sliced_analyzer = si.create_sorting_analyzer(
    #     sorting=sorting_slice,
    #     recording=segment,
    #     )
    # #sliced_analyzer.set_probe(probe)  # set the probe to the sliced analyzer
    # sorting_analyzer = sliced_analyzer  # reassign to the sliced analyzer
    
    
    # #sorting_analyzer.time_slice(start_time, start_time + duration)  # slice the sorting analyzer to match the segment time slice
    # post_slice_duration = sliced_analyzer.get_total_duration()
    # print(f"Original sorting analyzer duration: {original_duration:.2f} seconds")
    # print(f"Post-slice sorting analyzer duration: {post_slice_duration:.2f} seconds")
    
    
    # --- 2.1) get computable extensions ---
    # https://spikeinterface.readthedocs.io/en/stable/modules/core.html#sortinganalyzer
    # >>> ['random_spikes', 'waveforms', 'templates', 'noise_levels', 
    # 'amplitude_scalings', 'correlograms', 'isi_histograms', 'principal_components', 
    # 'spike_amplitudes', 'spike_locations', 'template_metrics', 'template_similarity', 'unit_locations', 'quality_metrics']
    print("Available computable extensions:")
    kwargs = job_kwargs if job_kwargs else {}
    for ext in sorting_analyzer.get_computable_extensions():
        print(f"  - {ext}")
    all_computable_extensions = sorting_analyzer.get_computable_extensions()
    print("Computing some extensions...")
    #print(f"\t - 'templates' and 'spike_amplitudes'")
    # if 'templates' not in sorting_analyzer.extensions or recompute:
    #     sorting_analyzer.compute(['templates', 'spike_amplitudes'], **kwargs)
    
    # if recompute: # delete all keys in sorting_analyzer.extensions
    #     print("Recompute is set to True, clearing all extensions...")
    #     sorting_analyzer.extensions.clear()
    #     print("Cleared all extensions in sorting analyzer.")
    print(f"\t- noise_levels")
    if 'noise_levels' not in sorting_analyzer.extensions or recompute:
        #print(f"\t- 'noise_levels'")
        sorting_analyzer.compute('noise_levels', **kwargs)
    print(f"\t- 'random_spikes'")
    if 'random_spikes' not in sorting_analyzer.extensions or recompute:
        #print(f"\t- 'random_spikes'")
        sorting_analyzer.compute('random_spikes', **kwargs)
    print(f"\t- 'waveforms'")
    if 'waveforms' not in sorting_analyzer.extensions or recompute:
        sorting_analyzer.compute('waveforms', ms_before=tpl_ms_before, ms_after=tpl_ms_after, **kwargs)
    print(f"\t- 'templates'")
    if 'templates' not in sorting_analyzer.extensions or recompute:
        sorting_analyzer.compute('templates', **kwargs)
    print(f"\t- 'spike_amplitudes'")
    if 'spike_amplitudes' not in sorting_analyzer.extensions or recompute:
        sorting_analyzer.compute('spike_amplitudes', peak_sign="pos", **kwargs)
    print(f"\t- 'spike_locations'")
    if 'spike_locations' not in sorting_analyzer.extensions or recompute:
        sorting_analyzer.compute('spike_locations', **kwargs)
    if 'quality_metrics' not in sorting_analyzer.extensions or recompute:
        print(f"\t- 'quality_metrics'")
        sorting_analyzer.compute('quality_metrics', **kwargs)
    if 'unit_locations' not in sorting_analyzer.extensions or recompute:
        print(f"\t- 'unit_locations'")
        sorting_analyzer.compute('unit_locations', **kwargs)
    metrics = sorting_analyzer.extensions['quality_metrics'].data['metrics']
    # save metrics as a CSV file
    #metrics_file = "quality_metrics.csv"
    #metrics_file = f"quality_metrics_{start_time:.2f}.csv"
    if output_dir:
        metrics_file = os.path.join(output_dir, f"quality_metrics_{well_id}_{start_time:.2f}.csv")
        metrics.to_csv(metrics_file, index=False)
        print(f"Quality metrics saved to {metrics_file}")
    
    # --- set up color map for amplitudes ---
    # cmap = plt.get_cmap('viridis')
    amps = sorting_analyzer.extensions['spike_amplitudes']
    # if amps is None:
    #     print("⚠️ No spike amplitudes found in sorting analyzer.")
    #     return
    # max_amp = np.max(amps.data['amplitudes']) if amps else None
    # min_amp = np.min(amps.data['amplitudes']) if amps else None    
    # if max_amp is None or min_amp is None:
    #     print("⚠️ No spike amplitudes found in sorting analyzer.")
    #     return
    # norm = plt.Normalize(vmin=min_amp, vmax=max_amp)
    
    # plot spike locations, scatter plot on 2D map representing chip where channels are located
    # color code by spike amplitude
    spike_locs = sorting_analyzer.extensions['spike_locations']
    if spike_locs is None:
        print("⚠️ No spike locations found in sorting analyzer.")
        return
    
    
    # plot spike locations
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"Spike Locations for {well_id} - {start_time}s to {start_time + duration}s")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    
    # --- vectorized scatter of all spike locations ---
    spike_locs_data = spike_locs.data['spike_locations']  # (n_spikes, 2)
    spike_amps      = amps.data['amplitudes']             # (n_spikes,)
    spk_times = spk_times
    
    # assert these are teh same length
    if len(spike_locs_data) != len(spike_amps) or len(spike_locs_data) != len(spk_times):
        print(f"⚠️ Spike locations and amplitudes have different lengths: {len(spike_locs_data)} vs {len(spike_amps)} vs {len(spk_times)}")
        return
    
    # apply start time and end duration to spike times, cut spike locs and spike amps by the same
    spike_locs_data = spike_locs_data[(spk_times >= start_time) & (spk_times < start_time + duration)]
    spike_amps = spike_amps[(spk_times >= start_time) & (spk_times < start_time + duration)]
    spk_times = spk_times[(spk_times >= start_time) & (spk_times < start_time + duration)]
    
    xs = [i[0] for i in spike_locs_data]  # x coordinates
    ys = [i[1] for i in spike_locs_data]  # y coordinates
    xs = np.array(xs)
    ys = np.array(ys)
    
    # warn if negative values in xs or ys
    if np.any(xs < 0) or np.any(ys < 0):
        print("⚠️ Negative coordinates found in spike locations.")
        # print("⚠️ Negative coordinates found in spike locations, adjusting to positive values.")
        # xs = np.clip(xs, 0, None)
        # ys = np.clip(ys, 0, None)
        
    # warn if any NaN values in xs, ys, or spike_amps
    if np.any(np.isnan(xs)):
        print("⚠️ NaN values found in spike x coordinates.")
        print(f"{len(xs[np.isnan(xs)])} NaN values in xs.")
    if np.any(np.isnan(ys)):
        print("⚠️ NaN values found in spike y coordinates.")
        print(f"{len(ys[np.isnan(ys)])} NaN values in ys.")
    if np.any(np.isnan(spike_amps)):
        print("⚠️ NaN values found in spike amplitudes.")
        print(f"{len(spike_amps[np.isnan(spike_amps)])} NaN values in spike amplitudes.")

    pitch = 17.5       # nm
    channel_locations = segment.get_channel_locations()  # (n_ch, 2)
    channel_xs = [i[0] for i in channel_locations]  # x coordinates
    channel_ys = [i[1] for i in channel_locations]  # y coordinates
    channel_xs = np.array(channel_xs)
    channel_ys = np.array(channel_ys)
    max_x = max(4200.0, np.max(xs), np.max(channel_xs))  # nm
    max_y = max(2100.0, np.max(ys), np.max(channel_ys))  # nm
    min_x = min(0.0, np.min(xs), np.min(channel_xs))  # nm
    min_y = min(0.0, np.min(ys), np.min(channel_ys))  # nm
    n_x = int(np.ceil((max_x - min_x) / pitch))  # number of bins in x
    n_y = int(np.ceil((max_y - min_y) / pitch))  # number of bins in y
    print(f"Grid size: {n_x} x {n_y} bins, pitch: {pitch} nm")
    print(f"Max X: {max_x} nm, Max Y: {max_y} nm")
    print(f"Min X: {min_x} nm, Min Y: {min_y} nm")


    # 0) drop any NaN locations
    good = (~np.isnan(xs)) & (~np.isnan(ys))
    xs  = xs[good]
    ys  = ys[good]
    spike_amps = spike_amps[good]

    # 1) compute bin indices via floor division
    x_idx = np.floor(xs / pitch).astype(int)
    y_idx = np.floor(ys / pitch).astype(int)
    
    # warn if any negative indices
    if np.any(x_idx < 0) or np.any(y_idx < 0):
        print("⚠️ Negative bin indices found in spike locations.")
        # print("⚠️ Negative bin indices found in spike locations, adjusting to positive values.")
        # x_idx = np.clip(x_idx, 0, n_x-1)
        # y_idx = np.clip(y_idx, 0, n_y-1)

    # 2) mask out‐of‐bounds spikes
    # valid = (
    #     (x_idx >= 0) & #(x_idx <= n_x) &
    #     (y_idx >= 0) #& (y_idx <= n_y)
    # )
    valid = (
        (x_idx >= 0) & (x_idx < n_x) &
        (y_idx >= 0) & (y_idx < n_y)
    )
    # print(f"Valid spikes: {np.sum(valid)} out of {len(xs)}")
    # print(f"Invalid spikes: {np.sum(~valid)} out of {len(xs)}")
    invalid = ~valid
    invalid_x_idx = x_idx[invalid]
    invalid_y_idx = y_idx[invalid]
    if np.any(invalid):
        print(f"⚠️ {np.sum(invalid)} spikes out of bounds, ignoring them.")
    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    amps  = spike_amps[valid]
    
    # count spikes at each bin
    n_bins = n_x * n_y
    spike_counts = np.zeros(n_bins, dtype=int)
    for x, y in zip(x_idx, y_idx):
        bin_idx = y * n_x + x
        #print(bin_idx, x, y)
        spike_counts[bin_idx] += 1
    min_spike_count = np.min(spike_counts)
    max_spike_count = np.max(spike_counts)
    print(f"Spike counts: min={min_spike_count}, max={max_spike_count}, total={np.sum(spike_counts)}")
    #print(f"Total spikes in bounds: {np.sum(spike_counts)}")

    
    # 1) reshape into (n_y, n_x)
    spike_counts_grid = spike_counts.reshape((n_y, n_x))
    counts = spike_counts_grid.copy()
    
    # replace all zero counts with 1 for log scale
    counts_masked = np.ma.masked_equal(counts, 0)
    #counts = np.where(counts == 0, 1, counts)  # avoid log(0)
    norm = LogNorm(vmin=np.nanmin(counts_masked), vmax=np.nanmax(counts_masked))  # log scale normalization

    # vmin=np.nanmin(counts)
    # vmax=np.nanmax(counts)

    # 2) plot heatmap
    #fig, ax = plt.subplots(figsize=(8, 6))
    extent = [0, max_x, 0, max_y]  # x from 0→4000 nm, y from 0→2000 nm


    #from matplotlib.colors import LogNorm

    # # mask out zeros so LogNorm only sees >0 values
    # counts_masked = np.ma.masked_equal(spike_counts_grid, 0)

    # if counts_masked.count() == 0:
    #     # no nonzero bins, fall back to a flat linear map
    #     norm = None
    #     print("⚠️ All bins are zero—using linear scale.")
    # else:
    # vmin = counts_masked.min()
    # vmax = counts_masked.max()
    # norm = LogNorm(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        counts_masked,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap='viridis',
        norm=norm
    )

    ax.set_title(f"Spike count heatmap\n{well_id} {start_time:.2f}–{start_time+duration:.2f} s")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    
    cbar = fig.colorbar(im, ax=ax, label="Spike count per bin")
    #plt.tight_layout()
    
    # 3) map each bin index → center coordinate
    # binned_x = x_idx * pitch + pitch / 2.0
    # binned_y = y_idx * pitch + pitch / 2.0

    # # --- now plot them all at once ---
    # sc = ax.scatter(
    #     binned_x, binned_y,
    #     c=amps,
    #     s=5,
    #     cmap=cmap,
    #     norm=norm,
    #     rasterized=True,
    # )
    # cbar = fig.colorbar(sc, ax=ax, label='Spike amplitude (µV)')
    
    # actual channel locations
    channel_locations = segment.get_channel_locations()  # (n_ch, 2) 
    channel_xs = [i[0]+pitch/2 for i in channel_locations]  # x coordinates
    channel_ys = [i[1]+pitch/2 for i in channel_locations]  #    y coordinates
    channel_xs = np.array(channel_xs)
    channel_ys = np.array(channel_ys)

    ax.scatter(channel_xs, channel_ys, color='black', s=2, alpha=0.5, label='Channels')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"✅ Spike locations plot saved to {output_file}")
        print()
    else:
        plt.show()
    

    
    
    # old code for reference
    
    # # --- 2.2) get channel locations ---
    # # (n_ch, 2) array of (x, y) coordinates for each

    
    # # print("Computing each extension...")
    # # for ext in all_computable_extensions:
    # #     print(f"Computing {ext}...")
    # #     sorting_analyzer.compute(ext)
    
    
    # #sorting = sorting_analyzer.sorting  # SortingExtractor 

    # # back‑compat waveform extraction → average template per unit
    # # you may already have this cached in your analyzer folder
    # we = si.extract_waveforms(
    #     recording=segment,
    #     sorting=sorting,
    #     folder=os.path.join(analyzer_output_dir, well_id, "waveforms"),
    #     ms_before=tpl_ms_before,
    #     ms_after=tpl_ms_after,
    #     max_spikes_per_unit=500
    # )  # uses WaveformExtractor under the hood 

    # # build templates dict: unit_id → array (n_samples_tpl, n_channels)
    # templates = {
    #     u: we.get_unit_template(u, mode="average")
    #     for u in sorting.get_unit_ids()
    # }

    # # --- 3) cross‑correlate & detect spikes on every channel ---
    # # prepare per‑channel storage
    # spike_counts   = np.zeros(n_ch, dtype=int)
    # max_amplitudes = np.zeros(n_ch, dtype=float)

    # for unit_id, tpl in templates.items():
    #     # pick the channel with strongest deflection in this unit’s template
    #     # (faster than correlating on every channel)
    #     best_ch = np.argmax(np.max(np.abs(tpl), axis=0))
    #     tpl_wave = tpl[:, best_ch]

    #     # compute raw‑trace ↔ template correlation for each channel
    #     for ch in range(n_ch):
    #         corr = correlate(traces[:, ch], tpl_wave, mode='same')
    #         # dynamic threshold: fraction of max correlation
    #         thresh = corr_threshold_factor * corr.max()
    #         peaks, _ = find_peaks(corr, height=thresh)

    #         # record per‑channel stats
    #         spike_counts[ch] = spike_counts[ch] + len(peaks)
    #         # map each peak back to trace amplitude at that channel
    #         amps = traces[peaks, ch]
    #         if amps.size:
    #             max_amplitudes[ch] = max(max_amplitudes[ch], np.max(np.abs(amps)))

    # # --- 4) build 2D maps from channel_locations ---
    # locs = np.array(segment.get_channel_locations())  # (n_ch, 2) 
    # xs = np.unique(locs[:, 0]); ys = np.unique(locs[:, 1])[::-1]
    # nx, ny = len(xs), len(ys)
    # count_map   = np.zeros((ny, nx))
    # amp_map     = np.zeros((ny, nx))

    # for ch_idx, (x, y) in enumerate(locs):
    #     xi = np.where(xs == x)[0][0]
    #     yi = np.where(ys == y)[0][0]
    #     count_map[yi, xi] = spike_counts[ch_idx]
    #     amp_map[  yi, xi] = max_amplitudes[ch_idx]

    # # --- 5) plot side‑by‑side heatmaps ---
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
    # im0 = ax0.imshow(count_map, origin='upper', aspect='equal')
    # ax0.set_title("Spike count per channel")
    # fig.colorbar(im0, ax=ax0, shrink=0.7)

    # im1 = ax1.imshow(amp_map, origin='upper', aspect='equal')
    # ax1.set_title("Max spike amplitude per channel")
    # fig.colorbar(im1, ax=ax1, shrink=0.7)

    # for ax in (ax0, ax1):
    #     ax.set_xticks([]); ax.set_yticks([])

    # plt.tight_layout()
    # if output_file:
    #     plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.show()

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


