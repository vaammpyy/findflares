# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and adheres to [Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-03-25
- Fixed a major injection recovery bug, flare removal was degrading the quality of the lightcurve with each injection recovery run.

## [0.1.0] - 2024-11-06
This is inital release of the findflare module, there might be some major changes depending on my current project directions.

## TODO
-- 2024-11-19
* [DONE] Currently the Rotation period of the star being stored is coming from the GLS method, final rotation model result is not being stored in the obj.star.prot.
-- 2024-12-12
* Add a post-processing module which will sort out all the stars for which data was found and pipeline was successfully completed.