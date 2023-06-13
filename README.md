# be-datahive

[![image](https://img.shields.io/pypi/v/be_datahive.svg)](https://pypi.python.org/pypi/be_datahive)
 
Python library for the [BE-dataHIVE
API](https://be-datahive.com/documentation.html).

# Installation

`be_datahive` is available on
[PYPI](https://pypi.python.org/pypi/be_datahive/). Install with `pip`:

``` bash
pip install be_datahive
```

# Documentation

Create an `api` object for interacting with the API:

``` python
from be_datahive import be_datahive
api = be_datahive()
```

Obtain efficiency & bystander data:

``` python
efficiency_data = api.get_efficiency()
bystander_data = api.get_bystander()
```

Convert efficiency & bystander data into machine-ready arrays:

``` python
ef_features, ef_target = api.get_efficiency_ml_arrays(efficiency_data, encoding='one-hot')
by_features, by_target = api.get_bystander_ml_arrays(bystander_data, encoding='one-hot')
```

# Citation
When using the [BE-dataHIVE
API](https://be-datahive.com/documentation.html), please cite our paper as outlined below. 

```bibtex
@article{Schneider.2023,
    title = "BE-dataHIVE: a Base Editing Database",
    author = "Lucas Schneider, Peter Minary",
    year = "2023",
}
```
