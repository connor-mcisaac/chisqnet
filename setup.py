import setuptools


setup_requires = ['numpy>=1.16.0']
install_requires = setup_requires + ['pycbc>=1.17.0',
                                     'tensorflow>=2.0.0',
                                     'tensorflow-probability>=0.12.0']

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="chisqnet",
    version="0.0.1",
    author="Connor McIsaac",
    author_email="connor.mcisaac@outlook.com",
    description="A package for tuning GW searches using ML.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/connor-mcisaac/chisqnet",
    setup_requires=setup_requires,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    scripts=["bin/filter_triggers",
             "bin/plan_training",
             "bin/preprocess_strain",
             "bin/merge_samples",
             "bin/train_model"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)