Todo
====

Core
----
    - fab deployment scripts
    - Central demo page linking to the demo pages of the various backends
    - More unified demosystem
    - In the core, add xml templates for getcapabilities and exceptions
    - Create aggdraw package:
        - Create distinct version in setup.py
        - Python setup.py sdist something
        - Put on package server
        - Put in buildout

Backends
--------
3di:
    - Height points and crossections, needs request signature discussion
    - Multithreaded pyramid construction
    - Caching strategy
        - Existing timestep:
            - Check timeout
        - New timestep:
            - Check max amount of cached timesteps
        - Existing layer:
            - Check timeout
        - New layer:
            - Check max amount of cached layers
            - Remove pyramids for removed layers
            - Make pyramid name a hash of selected .nc headers to
              automatically invalidate

Deltaportaal:
    - The proof of concept from threedi-server

Shape backend:
    - Separation of style and content
