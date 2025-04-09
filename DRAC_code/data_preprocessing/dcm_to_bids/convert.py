import os
import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np
from preprocessing import convert_dicom, copy_nii_to_project, process_tsv_files

# Get participant from command line argument
participant = sys.argv[1]
print(f"Processing participant: {participant}")

script_dir = os.path.dirname(os.path.abspath(__file__))  
root_dir = Path(os.path.abspath(os.path.join(script_dir, '../..')) )
shared_data_folder = Path(os.path.abspath(os.path.join(script_dir, '../../../../../TC2See')) )
output_folder = shared_data_folder / "bids_data"
project = "TC2See"

tmp_folder = root_dir / f"data/__tmp3__{ participant }"

try:
    sub = participant

    base_folder = shared_data_folder / f"scanner_data/{project}_{sub}"
    recordings = base_folder / "study"
    regex_runs = r".*Run(\d)_(.*)_\d*"
    regex_anat = r"t1_mprage_sag_p2_iso1.0(?:_\d*)?"

    # regex_tsv = r".*/sub-(?:.*\D)?(\d+)(?:.*\D)?_task-(.*)_run-(\d)_events.[ct]sv"
    # regex_tsv = rf".*/(?:.*\D)?(\d+)(?:.*\D)?_()(\d)_result_store.csv"
    regex_tsv = rf".*{re.escape(os.sep)}(?:.*\D)?(\d+)(?:.*\D)?_()(\d)_result_store.csv"

    TR = 2
    throw_away_trs = 5

    # convert the dicom to niifty
    convert_dicom(base_folder, tmp_folder, regex_runs, regex_anat, script_dir, throw_away_trs=throw_away_trs)
    # copy and rename the files
    copy_nii_to_project(tmp_folder, project, sub, output_folder)
    # copy and strip throw-away trs from the tsv files
    process_tsv_files(base_folder, project, regex_tsv, output_folder, TR, throw_away_trs=throw_away_trs)

    # remove the temporary folder
    if Path(tmp_folder).exists():
        os.system(f"rm -r {tmp_folder}")

except Exception as e:
    print(f"Error converting participant: {participant}")
    print(e)
    # remove the temporary folder
    if Path(tmp_folder).exists():
        os.system(f"rm -r {tmp_folder}")
