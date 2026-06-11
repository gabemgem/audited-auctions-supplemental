#!/bin/bash -l

qsub cluster_run.sh data/samples/0_full_tweets.csv 1
qsub cluster_run.sh data/samples/0_full_tweets.csv 2
qsub cluster_run.sh data/samples/0_full_tweets.csv 3

qsub cluster_run.sh data/samples/a_single_modal_normal_distribution.csv 1
qsub cluster_run.sh data/samples/a_single_modal_normal_distribution.csv 2
qsub cluster_run.sh data/samples/a_single_modal_normal_distribution.csv 3

qsub cluster_run.sh data/samples/b_two_modal_normal_distribution_\(side_by_side\).csv 1
qsub cluster_run.sh data/samples/b_two_modal_normal_distribution_\(side_by_side\).csv 2
qsub cluster_run.sh data/samples/b_two_modal_normal_distribution_\(side_by_side\).csv 3

qsub cluster_run.sh data/samples/c_two_modal_normal_distribution_\(diagonal_.\'\).csv 1
qsub cluster_run.sh data/samples/c_two_modal_normal_distribution_\(diagonal_.\'\).csv 2
qsub cluster_run.sh data/samples/c_two_modal_normal_distribution_\(diagonal_.\'\).csv 3

qsub cluster_run.sh data/samples/d_two_modal_normal_distribution_\(diagonal_\'.\).csv 1
qsub cluster_run.sh data/samples/d_two_modal_normal_distribution_\(diagonal_\'.\).csv 2
qsub cluster_run.sh data/samples/d_two_modal_normal_distribution_\(diagonal_\'.\).csv 3

qsub cluster_run.sh data/samples/e_two_modal_normal_distribution_\(stacked\).csv 1
qsub cluster_run.sh data/samples/e_two_modal_normal_distribution_\(stacked\).csv 2
qsub cluster_run.sh data/samples/e_two_modal_normal_distribution_\(stacked\).csv 3

qsub cluster_run.sh data/samples/f_four_modal_normal_distribution.csv 1
qsub cluster_run.sh data/samples/f_four_modal_normal_distribution.csv 2
qsub cluster_run.sh data/samples/f_four_modal_normal_distribution.csv 3

