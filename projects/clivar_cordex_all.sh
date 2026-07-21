#!/bin/bash
set -e

EXPTS=( PAS_LW1.5_ PAS_LW2.0_ PAS_MENS_ PAS_LENS )

# Loop through all experiments and ensemble members
for EXPT in "${EXPTS[@]}"; do
    if [ "$EXPT" = "PAS_LW1.5_" ]; then
	NUM_ENS=5
    else
	NUM_ENS=10
    fi
    for ENS in $(seq -f "%03g" 1 $NUM_ENS); do
	# Extract experiment from archive
	tar xzvf ../archive/scenarios/${EXPT}"$ENS"_O.tar.gz
	# Call the python script
	if [ "$EXPT" = "PAS_LENS" ]; then
	    # Call an extra time for historical
	    python -c 'from mitgcm_python.projects.clivar_cordex import *; process_expt("'${EXPT}"$ENS"_0/'", historical=True)'
	fi
	python -c 'from mitgcm_python.projects.clivar_cordex import *; process_expt("'${EXPT}"$ENS"_0/'")'
	# Remove the extracted directory
	rm -rf ${EXPT}"$ENS"_O
    done
done
