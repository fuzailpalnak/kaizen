from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "shapely == 1.7.1",
    "fiona >= 1.8.9.post2",
    "rtree >= 0.9.4",
    "affine == 2.3.0",
    "geopandas == 0.8.1",
    "matplotlib == 3.3.1",
    "networkx == 2.5",
    "numpy == 1.19.1",
    "opencv-python >= 4.1.1",
    "pandas == 1.1.2",
    "rasterio == 1.1.5",
    "scipy == 1.5.2",
    "visvalingamwyatt == 0.1.3",
    "GDAL == 2.4.4"
]

setup(
    name="kaizen",
    version="0.0.1",
    author="Fuzail Palnak",
    author_email="fuzailpalnak@gmail.com",
    url="https://github.com/fuzailpalnak/kaizen",
    description="Map Matching and road Building Conflict solving Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires='~=3.3',
    install_requires=install_requires,
    keywords=["GIS, Map Matching, Road Network, Building Footprint, Polygon, MultiPolygon, Geometry"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
)
