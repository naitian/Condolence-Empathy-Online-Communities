# Condolence and Empathy in Online Communities

Github repository for EMNLP 2020 paper Condolence and Empathy in Online
Communities by Naitian Zhou and David Jurgens.

## Data

Due to the sensitive nature of this data, it will be made available upon
request to researchers.

## Code

### Pre-trained models

If you would like to use our pretrained models, you can install with `pip` with
```
pip install condolence-models
```

See [the Github repository](https://github.com/naitian/condolence-models) or
[the PyPi listing](https://pypi.org/project/condolence-models/) for more
details and documentation.

### Analysis

Notebooks for analysis on the final Reddit 2018 dataset are in the `analysis`
folder. Notebooks are named by the order in which they are to be run. You can
also view notebook dependencies at the top of each notebook.

Some requisite auxiliary data has also been included (namely, Reddit 2018 daily
volume).

### Crawling

Code for crawling through Reddit to heuristically label condolence and distress
are located in the `crawling` directory. The directory also includes the
labeling script used to get the final Reddit 2018 dataset used for analysis
(`grab_comments.py`).

Note: this code might be messy. Peruse at your own risk.

### Models

Notebooks and code for training models are in the `models` directory. This code
might be messy. Peruse at your own risk.


## Contact

Naitian Zhou (naitian@umich.edu)
