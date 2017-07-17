from setuptools import setup

setup(name='phyloinfer',
      version='0.3',
      description='An efficient Bayesian phylogenetic inference package',
      url='https://github.com/zcrabbit/PhyloInfer',
      author='Cheng Zhang',
      author_email='zc.rabbit@gmail.com',
      license='MIT',
      packages=['phyloinfer'],
      install_requires=[
          'ete3',
          'biopython',
      ],
      zip_safe=False)
