"""setup.py file for packaging genie-worker-scoring."""

from setuptools import setup


with open('README.md', 'r') as readme_file:
    readme = readme_file.read()


setup(
    name='workerscoring',
    version='1.0.0',
    description='Automatic crowdsourcing spammer detection',
    long_description=readme,
    url='https://github.com/allenai/genie-worker-scoring',
    author='Nicholas Lourie',
    author_email='nicholasl@collaborator.allenai.org',
    keywords='GENIE crowdsourcing test questions spam detection'
             ' artificial intelligence ai machine learning ml',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    license='Apache',
    packages=['workerscoring'],
    package_dir={'': 'src'},
    scripts=[],
    install_requires=[
        'numpy >= 1.21.6',
        'scipy >= 1.7.3',
        'tqdm >= 4.64.0',
    ],
    setup_requires=[],
    tests_require=[],
    include_package_data=True,
    python_requires='>= 3.7',
    zip_safe=False,
)
