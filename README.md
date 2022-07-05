# Kaizen
![Licence](https://img.shields.io/github/license/fuzailpalnak/kaizen)
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![Downloads](https://static.pepy.tech/personalized-badge/kaizen-mapping?period=total&units=international_system&left_color=yellow&right_color=grey&left_text=Downloads)](https://pepy.tech/project/kaizen-mapping)

A Library build with two propose, to *map match* road elements either with *probe trace or road elements from different
source* and help, tackle the problem of *roads and building intersecting or overlapping*, which are results of
inaccurate digitizing, snapping, or resource mismatch.

This Library, presents, my view on tackling the aforementioned problem, caused during map making, addressed 
using obstacle avoidance and map matching

<a href='https://ko-fi.com/fuzailpalnak' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi1.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

<table>
  <tr>
    <td>MapMatch</td>
    <td>Conflict Resolver</td>
    <td>ICT</td>

  </tr>
  <tr>
    <td><img src="https://user-images.githubusercontent.com/24665570/94099696-2f6d9580-fe49-11ea-95f4-c5b53443f4a6.gif" width=500 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/24665570/94338794-a86c1900-0012-11eb-9fad-434a1d6e6749.gif" width=500 height=200></td>
    <td><img src="https://user-images.githubusercontent.com/24665570/96708811-62a73400-13b7-11eb-970c-a4d8b96f9764.gif" width=500 height=200></td>

  </tr>
 </table>

## Installation
    
    pip install kaizen-mapping
    
## Requirements

- *_Rtree - [installation](https://anaconda.org/conda-forge/rtree)_*  
- *_Geopandas - [installation](https://anaconda.org/conda-forge/geopandas)_*
- *_Rasterio - [installation](https://anaconda.org/conda-forge/rasterio)_*
- *_GDAL 2.4.4 - [installation](https://anaconda.org/conda-forge/gdal)_*
- *_Fiona -  [installation](https://anaconda.org/conda-forge/fiona)_*
- *_Shapely -  [installation](https://anaconda.org/conda-forge/shapely)_*

 
The library uses [Rtree](https://rtree.readthedocs.io/en/latest/) which has a dependency on 
[libspatialindex](https://libspatialindex.org/), 
It is recommend to resolve the dependency through [conda](https://anaconda.org/conda-forge/libspatialindex)

*_LibSpatialIndex For Linux:_*

    $ sudo apt-get update -y
    $ sudo apt-get install -y libspatialindex-dev
        
*_LibSpatialIndex For Windows:_*

Experience is pretty slim, for Windows Installation, I recommend using conda, for trouble free installation. 

## Demo

Data for running the demo examples can be downloaded from [here](https://github.com/fuzailpalnak/kaizen/releases/download/0.0.1/data.zip)

## MapMatch 

Examples

1. [Map Matching Road Element with Line String](https://github.com/fuzailpalnak/kaizen/blob/master/examples/MapMatchingWithLineString.ipynb)
2. [Map Matching Road Element with List of Point](https://github.com/fuzailpalnak/kaizen/blob/master/examples/MapMatchingWithPoint.ipynb)

## Conflict Resolver

Examples 

1. [Solving Conflict Between Building and Road without additional Reference](https://github.com/fuzailpalnak/kaizen/blob/master/examples/ConflictResolver.ipynb)
2. [Complex Solving Conflict Between Building and Road without additional Reference](https://github.com/fuzailpalnak/kaizen/blob/master/examples/ConflictResolverComplex.ipynb)
3. [Solving Conflict Between Building and Road with matching the conflict with neighbouring data and finding 
associated reference points](https://github.com/fuzailpalnak/kaizen/blob/master/examples/ConflictResolverWithMapMatching.ipynb)

## ICT

Example

1. [Spatial ICT](https://github.com/fuzailpalnak/kaizen/blob/master/examples/Spatial_ICT.ipynb)


## References

1. [Fast Map Matching](https://people.kth.se/~cyang/bib/fmm.pdf)
2. [ST-Map Matching](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/Map-Matching20for20Low-Sampling-Rate20GPS20Trajectories-cameraReady.pdf)
3. [Game Programming](http://theory.stanford.edu/~amitp/GameProgramming/)
4. [Robot Navigation](https://github.com/AtsushiSakai/PythonRobotics)




