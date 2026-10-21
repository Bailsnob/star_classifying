# Project Goal: Classifying Stars with Gaia DR3

## Overview
Use the **Gaia DR3** catalog (queried directly from Python via `astroquery`, no bulk download) to study how well a star's observable properties reveal its physical nature. The project is split into two phases that are later compared: a **supervised** phase that learns to predict a star's spectral type, and an **unsupervised** phase that discovers stellar groups on the Hertzsprung-Russell (HR) diagram.

All work draws from a single shared sample so the two phases can be directly compared. Data comes from `gaiadr3.gaia_source` joined to `gaiadr3.astrophysical_parameters` by `source_id`.

## Phase 1 - Supervised: Predicting Spectral Type
**Goal:** Train a classifier to predict a star's spectral type (`spectraltype_esphs`, one of `O, B, A, F, G, K, M, CSTAR`) from its **photometry and colors** - e.g. `bp_rp` color, apparent magnitude `phot_g_mean_mag`, and absolute magnitude `M_G`.

- Features deliberately **exclude** temperature, so the model must genuinely learn the color-to-type relationship rather than trivially copying a temperature bucket.
- Because the classes are highly imbalanced (O and CSTAR are rare), success is measured with **macro-F1 and a confusion matrix**, not raw accuracy.
- Deliverable: a trained, evaluated classifier plus an analysis of which photometric features matter most.

## Phase 2 - Unsupervised: Discovering Structure on the HR Diagram
**Goal:** Cluster the same stars on the HR diagram using **temperature (`teff_gspphot`) and luminosity (`lum_flame`)** to see what natural groupings emerge (e.g. the main sequence, giants, and other populations) without using any labels.

- Apply clustering (KMeans / Gaussian Mixture), choose the number of clusters via elbow/silhouette, and visualize the clusters on the HR diagram.
- This phase captures the **luminosity axis** (giant vs. dwarf / evolutionary state), information that spectral type alone does not encode.
- Deliverable: an interpreted clustering of the HR diagram.

## Comparing the Two Phases
Finally, compare the supervised labels against the unsupervised clusters (contingency table + Adjusted Rand Index). The expectation is that they **diverge**: spectral type tracks the temperature axis, while HR-diagram clusters also capture luminosity/evolutionary structure. Explaining *why* they differ is a core outcome of the project.
