from setuptools import setup, find_packages

# Read the contents of README.md for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="findflares",
    version="0.1.0",  # Adjust version as needed
    author="Rohan Kumar",  # Replace with your name
    author_email="rohankumarprasad@yahoo.com",  # Replace with your email
    description="A Python library for detecting stellar flares",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vaammpyy/findflares",
    packages=find_packages(),  # Automatically finds sub-packages with __init__.py files
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="==3.11.*",  # Requires Python 3.11
    install_requires=[
        "celerite2",
        "pymc-ext",
        "exoplanet[pymc]",
        "lightkurve",
        "numpy",
        "matplotlib",
        "scipy",
        "astropy"
    ],
)