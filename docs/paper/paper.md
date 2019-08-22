---
title: '``coronagraph``: Telescope Noise Modeling for Exoplanets in Python'
tags:
  - Python
  - Astronomy
  - Exoplanets
  - Telescopes
authors:
  - name: Jacob Lustig-Yaeger
    orcid: 0000-0002-0746-1980
    affiliation: 1
  - name: Tyler Robinson
    orcid: 0000-0002-3196-414X
    affiliation: 2
  - name: Giada Arney
    orcid: 0000-0001-6285-267X
    affiliation: 3
affiliations:
  - name: University of Washington
    index: 1
  - name: Northern Arizona State University
    index: 2
  - name: NASA Goddard Space Flight Center
    index: 3
date:
bibliography: paper.bib
---

# Summary

``coronagraph`` is an open-source Python package for generalized telescope noise
modeling for extrasolar planet (exoplanet) science. This package is based on
Interactive Data Language (IDL) code originally developed by T. Robinson [@coronagraph_noise_github], and described in detail with science applications in [@Robinson2016].

Briefly, ``coronagraph`` uses analytic relations to calculate the photon count
rates for a variety of astrophysical, telescope, and
instrumental noise sources. These include photons from
coronagraph speckles, zodiacal and exozodiacal dust, telescope thermal emission,
dark current, read noise, and clock-induced charge.
The model also includes Earth's atmospheric
transmission and emission spectrum from the UV through the near IR for
ground-based telescope modeling [@Meadows2018]. Photons from a user-provided
exoplanet source spectrum are compared against sources of noise to provide
wavelength-dependent signal-to-noise (S/N) ratios and synthetic observations.
The model also features "reverse" S/N calculations, which provide the requisite
exposure times per wavelength interval such that a user-specified S/N is
achieved on the exoplanet spectrum.

The ``coronagraph`` model was designed to assess exoplanet science capabilities
for near- and far-term future telescopes. Since the computations are all
relatively simple and analytic, ``coronagraph`` is ideal for fast sweeps across
a range of possible telescope, instrumental, astrophysical, and planetary
parameters to investigate primary noise sources and the feasibility of different
science case. However, ``coronagraph`` may be impractical for existing
telescopes or telescope concepts that have mature designs due to the
"first order" nature of the model.
Additionally, systematic errors (e.g. from speckle subtraction), stray light
from companion stars and/or background objects, and time-dependent (red) noise
terms are not considered. More specific or detailed use cases than those
outlined above may be appropriate for the Exoplanet Open-Source Imaging Mission
Simulator (EXOSIMS) [@exosims_code].

``coronagraph`` has already been used in numerous practical science
applications. This includes peer-reviewed work on the potential for
direct-imaging Proxima Centauri b with ground- and space-based telescopes
[@Meadows2018], and the detectability of exoplanet aurorae on Proxima Centauri b
[@Luger2017]. The ``coronagraph`` model is used within NASA Goddard's
interactive tool **Coronagraphic Spectra of Exoplanets** [@cron_model_nasa],
and is actively being used to motivate science cases for next-generation,
space-based, direct-imaging mission concepts [@Mennesson2016; @Bolcar2016].

The ``coronagraph`` package may also be used to simulate signal-to-noise ratios
and synthetic spectra for transiting exoplanets in transmission and emission.
The transiting exoplanet modules extend the potential science applications of
``coronagraph`` to non-coronagraph-equipped telescopes, enabling studies
relevant to a far-infrared surveyor mission concept [@Cooray2017].

# Acknowledgements

This work was performed as part of NASA's Virtual Planetary Laboratory,
supported by the National Aeronautics and Space Administration through the NASA
Astrobiology Institute under solicitation NNH12ZDA002C and Cooperative Agreement
Number NNA13AA93A, and by the NASA Astrobiology Program under grant
80NSSC18K0829 as part of the Nexus for Exoplanet System Science (NExSS)
research coordination network.

# References
