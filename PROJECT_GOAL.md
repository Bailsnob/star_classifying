# Project Goal: Classifying Stars with Gaia DR3

## Overview
Use the **Gaia DR3** catalog (queried directly from Python via `astroquery`, no bulk download) to study how well a star's observable properties reveal its physical nature. The project is split into two phases that are later compared: a **supervised** phase that learns to predict a star's spectral type, and an **unsupervised** phase that discovers stellar groups on the Hertzsprung-Russell (HR) diagram.

All work draws from a single shared sample so the two phases can be directly compared. Data comes from `gaiadr3.gaia_source` joined to `gaiadr3.astrophysical_parameters` by `source_id`.

## Phase 1 - Supervised: Predicting Spectral Type
**Goal:** Train a classifier to predict a star's spectral type (`spectraltype_esphs`, one of `O, B, A, F, G, K, M, CSTAR`) from its **photometry and colors** - e.g. `bp_rp` color, apparent magnitude `phot_g_mean_mag`, and absolute magnitude `M_G`.

- Features deliberately **exclude** temperature and luminosity. This tests recovery of catalog types from photometric features; it does not provide independent physical validation because Gaia products share observational inputs.
- Because the classes are highly imbalanced (O and CSTAR are rare), success is measured with **macro-F1 and a confusion matrix**, not raw accuracy.
- Deliverable: a trained, evaluated classifier plus an analysis of which photometric features matter most.

## Phase 2 - Unsupervised: Discovering Structure on the HR Diagram
**Goal:** Cluster the same stars on the HR diagram using **temperature (`teff_gspphot`) and luminosity (`lum_flame`)** to see what natural groupings emerge (e.g. the main sequence, giants, and other populations) without using any labels.

- Apply KMeans (silhouette selection; inertia plotted) and a Gaussian mixture (BIC selection), and visualize the groups on the HR diagram. Report boundary selections and compare prespecified alternative k values without choosing a result solely to match expected populations.
- Test whether adding luminosity yields repeatable structure beyond temperature alone. Giant/dwarf separation is a hypothesis, not a guaranteed output, and physical interpretations require additional evidence.
- Deliverable: a clustering analysis with explicit selection rules, temperature-only controls, stability checks, and qualified interpretation.

## Comparing the Two Phases
Compare catalog spectral labels with unsupervised groups using contingency tables and Adjusted Rand Index (ARI). The current implementation uses catalog labels, not predictions from the supervised classifier. If predicted labels are also compared later, identify that comparison separately. The working hypothesis is that adding luminosity can reveal structure not summarized by spectral types. Lower ARI alone does not demonstrate evolutionary structure; examine luminosity within narrow temperature ranges and repeat fits across seeds and sample choices.

The shared sample requires catalog labels and includes extra rare-class rows. Labels therefore influence sampling even though they are not clustering features. Interpret results as applying to this selected sample, not to a representative census of Galactic stars.
