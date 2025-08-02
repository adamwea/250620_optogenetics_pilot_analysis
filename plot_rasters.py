from analysis_helpers import *

# Run to set up the environment for running the analysis.
'''
shifter --image=adammwea/axonkilo_docker:v7 /bin/bash
'''
    
def main():
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
    
    # POCS
    pocs = POCS
    if pocs is not None:
        assert len(pocs) == 2, "POCS should contain exactly two points of change."

    # Plot raster units
    plot_raster_units(
        #h5_file_path=REC_PATH,
        analyzer_output_dir=ANALYZER_PATH,
        well_id="well000",
        #start_time=0,
        #duration=duration,
        #max_channels=200,
        markers=pocs,
        title=TITLE,
        output_file=f"RUN{RUN}_well000_raster.png",
        output_dir=ANALYSIS_OUTPUT_DIR,
        #job_kwargs=job_kwargs,
        verbose=True,
    )
    print("Raster plot saved successfully.")
    
if __name__ == "__main__":
    
    # # Baseline, Network Scan, Run 28
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000028/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000028/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000028/analyzer/'
    # RUN = 28
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_28'
    # STIM_RATE = None
    # POCS = None  # No stimulation in baseline
    # TITLE = f"RUN{RUN} - Baseline Raster Plot, Network Assay"
    # main()
    
    # # Baseline, NU4, Run 29
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000029/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000029/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000029/analyzer/'
    # RUN = 29
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_29'
    # STIM_RATE = None
    # POCS = None  # No stimulation in baseline
    # TITLE = f"RUN{RUN} - Baseline Raster Plot, NU4 Assay"
    # main()   
    
    # # 10 Hz Network Scan, Run 30
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000030/analyzer/'
    # RUN = 30
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_30'
    # STIM_RATE = 10  # Hz
    # POCS = [167, 415]  # Points of change in seconds
    # TITLE = f"RUN{RUN} - 10Hz Stim Raster Plot, Network Assay"
    # main()

    # # 10 Hz NU4, Run 32
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000032/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000032/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000032/analyzer/' 
    # RUN = 32
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_32'
    # STIM_RATE = 10  # Hz
    # POCS = [162, 411]  # Points of change in seconds
    # TITLE = f"RUN{RUN} - 10Hz Stim Raster Plot, NU4 Assay"
    # main()
    
    # # 20 Hz Network Default, Run 33
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000033/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000033/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000033/analyzer/'
    # RUN = 33
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_33'
    # STIM_RATE = 20  # Hz
    # POCS = [162, 407]  # Points of change in seconds
    # TITLE = f"RUN{RUN} - 20Hz Stim Raster Plot, Network Assay"
    # main()
    
    # # # 20 Hz NU4, Run 34
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000034/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000034/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000034/analyzer/' 
    # RUN = 34
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_34'
    # STIM_RATE = 20  # Hz
    # POCS = [162, 423]  # Points of change in seconds
    # TITLE = f"RUN{RUN} - 20Hz Stim Raster Plot, NU4 Assay"
    # main()
    
    # # 5 Hz Network Default, Run 35
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000035/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000035/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000035/analyzer/'
    # RUN = 35
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_35'
    # STIM_RATE = 5  # Hz
    # POCS = [163, 409]  # Points of change in seconds
    # TITLE = f"RUN{RUN} - 5Hz Stim Raster Plot, Network Assay"
    # main()
    
    # # # 5 Hz NU4, Run 36
    # REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000036/data.raw.h5'
    # SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000036/sorted/well000/sorter_output'
    # ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/P002779/Network/000036/analyzer/' 
    # RUN = 36
    # ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_36'
    # STIM_RATE = 5  # Hz
    # POCS = [164, 432]  # Points of change in seconds
    # TITLE = f"RUN{RUN} - 5Hz Stim Raster Plot, NU4 Assay"
    # main()
    
    # Empty Chip Testing, Run 42
    REC_PATH = '/global/homes/a/adammwea/pscratch/z_raw_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/25196/Network/000042/data.raw.h5'
    SORT_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/25196/Network/000042/sorted/well000/sorter_output'
    ANALYZER_PATH = '/global/homes/a/adammwea/pscratch/z_analyzed_data/irc_maxone_desktop/media/harddrive8tb/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/CDKL5-R59X_MaxOnePlus_T1_05202025_PS/250620/25196/Network/000042/analyzer/'
    RUN = 42
    ANALYSIS_OUTPUT_DIR = '/global/homes/a/adammwea/pscratch/z_output/analysis_42'
    STIM_RATE = None  # Hz
    POCS = None  # No stimulation in empty chip testing
    TITLE = f"RUN{RUN} - Empty Chip Testing Raster Plot, Network Assay"
    main()

    print("All analyses completed successfully.")