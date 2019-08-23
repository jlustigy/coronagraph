.. coronagraph documentation master file, created by
   sphinx-quickstart on Sun Mar 11 11:30:49 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

coronagraph
===========

.. image:: https://img.shields.io/badge/GitHub-jlustigy%2Fcoronagraph-blue.svg?style=flat
    :target: https://github.com/jlustigy/coronagraph
.. image:: https://joss.theoj.org/papers/10.21105/joss.01387/status.svg
    :target: https://doi.org/10.21105/joss.01387
.. image:: http://img.shields.io/travis/jlustigy/coronagraph/master.svg?style=flat
    :target: http://travis-ci.org/jlustigy/coronagraph
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/jlustigy/coronagraph/blob/master/LICENSE
.. image:: https://coveralls.io/repos/github/jlustigy/coronagraph/badge.svg?branch=master&style=flat
    :target: https://coveralls.io/github/jlustigy/coronagraph?branch=master

A Python noise model for directly imaging exoplanets with a
coronagraph-equipped telescope. The `original IDL
code <https://github.com/tdrobinson/coronagraph_noise>`_ for this coronagraph
model was developed and published by Tyler Robinson and collaborators
(`Robinson, Stapelfeldt & Marley
2016 <http://adsabs.harvard.edu/abs/2016PASP..128b5003R>`_). This open-source
Python version has been expanded upon in a few key ways, most notably, the
:class:`Telescope`, :class:`Planet`, and :class:`Star` objects used for
reflected light coronagraph noise modeling can now be used for
transmission and emission spectroscopy noise modeling, making this model a
general purpose exoplanet noise model for many different types of observations.

To get started using the ``coronagraph`` noise model, take a look at the
`Quickstart <notebooks/quickstart.html>`_ tutorial and the
`examples <examples.html>`_. For more details about the full functionality of
the code see the `Application Programming Interface (API) <api.html>`_.

If you use this model in your own research please cite
`Robinson, Stapelfeldt & Marley (2016) <https://ui.adsabs.harvard.edu/abs/2016PASP..128b5003R/abstract>`_
and `Lustig-Yaeger, Robinson & Arney (2019) <http://joss.theoj.org/papers/29a123d0178ea95da358dafc0282e8f7>`_.

Documentation
=============

.. toctree::
   :maxdepth: 2

   .. overview
   install
   Quickstart <notebooks/quickstart.ipynb>
   examples
   scripts
   api
   Github <https://github.com/jlustigy/coronagraph>
   Submit an issue <https://github.com/jlustigy/coronagraph/issues>

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
