from setuptools import setup, find_packages

setup(
    name='tweetan',
    version='0.0.1',
    description='A simple PyTorch tool to analyze tweets based on TweetSentBR',
    author='Marcos Treviso',
    author_email='marcostreviso@usp.br',
    url='https://github.com/mtreviso/tweetan',
    license='MIT',
    packages=find_packages(exclude=('tests', 'docs')),
    data_files=['LICENSE'],
    zip_safe=False,
    keywords='evaluator',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ]
)

