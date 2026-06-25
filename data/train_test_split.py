"""Federated train/test split generation.

This script previously generated splits for the segmentation datasets, which are
not part of the paper release. The two datasets used by the paper, FedISIC and
FedEMBED, do not require this script: their train/test splits are shipped under
``data/data_split/`` (FedISIC) and as ``train.csv`` / ``test.csv`` (FedEMBED).

No split generation is performed here for FedISIC/FedEMBED.
"""
