#!/bin/bash

sbatch job_evaluate_beam_search_2+8.sh
sbatch job_evaluate_beam_search_2.sh
sbatch job_evaluate_beam_search_5.sh
sbatch job_evaluate_beam_search_10.sh
sbatch job_evaluate_beam_search_transpose.sh
sbatch job_evaluate_beam_search_narrow.sh
sbatch job_evaluate_beam_search_shallow.sh
#sbatch job_evaluate_ggcnn_cornell.sh
#sbatch job_evaluate_ggcnn_rss.sh