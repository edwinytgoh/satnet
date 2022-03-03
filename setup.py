"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

import pathlib

# Always prefer setuptools over distutils
# from Cython.Build import cythonize
from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="satnet",  # Required
    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#choosing-a-versioning-scheme
    # For a discussion on single-sourcing the version across setup.py and the
    # project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.1",  # Required
    description="Interplanetary Satellite Scheduling Baseline RL Implementation",
    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    # Often, this is the same as your README, so you can just read it in from
    # that file directly (as we have already done above)
    #
    # This field corresponds to the "Description" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-optional
    #     long_description=long_description,  # Optional
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    #     long_description_content_type='text/markdown',  # Optional (see note above)
    # This should be a valid link to your project's main homepage.
    #
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url="https://github.com/edwinytgoh/dsn_rl_benchmark",
    # This should be your name or the name of the organization which owns the
    # project.
    #     author='A. Random Developer',
    # This should be a valid email address corresponding to the author listed
    # above.
    #     author_email='author@example.com',
    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3 :: Only",
    ],
    keywords="deeprl, ray, rllib, reinforcement learning, scheduling, development, aerospace, satellite, operations, research",
    packages=find_packages(),  # Required
    python_requires=">=3.8, <3.9",
    install_requires=[
        "ray[rllib]==1.2.0",
        "aiohttp<3.8.0",  # aiohttp 3.8.0 is not compatible with ray[rllib]==1.2.0
        "aioredis<1.3.1",  # aioredis 1.3.1 is not compatible with ray[rllib]==1.2.0
        "tensorflow-gpu>=2.4.1, <=2.5.0",
        "gym < 0.22.0",  # https://github.com/ray-project/ray/issues/22622#issuecomment-1050388768
        "pytest>=6.0.1",
        "numba>=0.50.1",
        "pandas",
        "pydot",  # for plotting model
        "pyarrow",
        "gputil",
        "matplotlib>=3.4.0",  # for env.render()
        "sortedcontainers",
    ],
    extras_require={
        "dev": ["check-manifest"],
        "test": ["coverage"],
    },
    # If there are data files included in your packages that need to be
    # installed, specify them here.
    # package_data={
    #     "problems": ["data/*.json"],
    #     "maintenance": ["data/*maintenance*.parquet"],
    # },
    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/distutils/setupscript.html#installing-additional-files
    #
    # List additional URLs that are relevant to your project as a dict.
    #
    # This field corresponds to the "Project-URL" metadata fields:
    # https://packaging.python.org/specifications/core-metadata/#project-url-multiple-use
    #
    # Examples listed include a pattern for specifying where the package tracks
    # issues, where the source is hosted, where to say thanks to the package
    # maintainers, and where to support the project financially. The key is
    # what's used to render the link text on PyPI.
    #     project_urls={
    #         'Bug Reports': 'https://github.com/pypa/sampleproject/issues',
    #         'Funding': 'https://donate.pypi.org',
    #         'Say Thanks!': 'http://saythanks.io/to/example',
    #         'Source': 'https://github.com/pypa/sampleproject/',
    #     },
)
