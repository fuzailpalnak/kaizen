from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [
    "affine == 2.3.0",
    "matplotlib == 3.3.1",
    "networkx == 2.5",
    "numpy == 1.19.1",
    "opencv-python >= 4.1.1",
    "pandas == 1.1.2",
    "scipy == 1.5.2",
    "visvalingamwyatt == 0.1.3",
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
    python_requires="~=3.3",
    install_requires=install_requires,
    keywords=[
        "GIS, Map Matching, Road Network, Building Footprint, Polygon, MultiPolygon, Geometry"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
    ],
)
