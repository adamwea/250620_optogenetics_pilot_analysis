'''
salloc -A m2043_g -q interactive -C gpu -t 04:00:00 --nodes=1 --gpus=1 --image=adammwea/axonkilo_docker:v7
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
'''
import os
from MEA_Analysis.NetworkAnalysis.awNetworkAnalysis.run_sorter import run_sorter
import glob

# prepare paths =============================================================================

# general location for inputs
input_dir = '/global/homes/a/adammwea/pscratch/z_raw_data/' #dir where all raw data are stored in pscratch (data should be copied here from long term storage before running for optimal I/O)

# raw_data input
raw_data_paths = [
    # 2025-08-01 12:59:38 - if commented out, these have already been sorted
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000028/data.raw.h5', # Baseline
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000029/data.raw.h5', # Baseline
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/data.raw.h5', # 10Hz Network
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000032/data.raw.h5', # 10Hz NU4
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000033/data.raw.h5', # 20Hz Network Default
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000034/data.raw.h5', # 20Hz NU4
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000035/data.raw.h5', # 5Hz Network Default
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000036/data.raw.h5', # 5Hz NU4
    #'irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/25196/Network/000042/data.raw.h5', # Blank Chip Network Default
]

# analyzed_data_path
output_dir = '/global/homes/a/adammwea/pscratch/z_analyzed_data/' #dir where all analyzed data are stored in pscratch (use data transfer bash script to copy to long term storage as needed)

def main():

    # automated paths =============================================================================
    input_path = os.path.join(input_dir, raw_data_path) # absolute path to raw data
    sorted_output_dir = os.path.join(output_dir, os.path.dirname(raw_data_path), 'sorted') # path to sorted data within outputs_dir
    waveform_output_dir = os.path.join(output_dir, os.path.dirname(raw_data_path), 'waveforms') # path to waveform data within outputs_dir
    analyzer_output_dir = os.path.join(output_dir, os.path.dirname(raw_data_path), 'analyzer') # path to analyzer data within outputs_dir

    # print paths
    print()
    print(f"Targeted paths:")
    print(f"input_path: {input_path}")
    print(f"sorted_output_dir: {sorted_output_dir}")
    print(f"analyzer_output_dir: {analyzer_output_dir}")

    # create output directories if they don't exist
    os.makedirs(sorted_output_dir, exist_ok=True)
    os.makedirs(waveform_output_dir, exist_ok=True)

    # =============================================================================
    run_sorter(
        input_path,
        sorted_output_dir,
        analyzer_output_dir,
        use_docker=False,   # NOTE: Default is True. Comment out this line to use docker.
                            #       If running on NERSC, you'll need to run without docker and with shifter.
                            #       see below for shifter command to run on NERSC
        #try_load = False,   # NOTE: Default is True. Comment out this line to try loading the sorted data.
        )
    
    print(f"Sorting completed for {raw_data_path}.")
    
if __name__ == "__main__":
    for raw_data_path in raw_data_paths:
        main()
        print(f"Completed processing for {raw_data_path}.")
    
    print("All analyses completed successfully.")