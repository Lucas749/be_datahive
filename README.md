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

# Get Started

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
# API Documentation

The Python wrapper uses the BE-dataHive API which is accessible at the endpoint (https://be-server.herokuapp.com). 
The detailed API documentation, describing all endpoints, query parameters, and response schema, can be viewed [here](https://be-datahive.com/documentation.html).


# Troubleshooting

If you encounter any issues while using the `be_datahive` library, please refer to the common issues listed below:

1. **Installation issues**: Make sure you have the latest version of `pip` and Python installed. Use `pip install --upgrade pip` to upgrade pip if necessary.
2. **API Connection Errors**: Check your internet connection and ensure that the API endpoint (https://be-server.herokuapp.com) is reachable.
3. **Data Retrieval Issues**: Ensure that you are using the correct function names and parameters as outlined in the documentation.

If your issue is not listed here or you need further assistance, please open an issue on GitHub or reach out directly.

# Reporting Issues

If you have any problems, you can open an issue on GitHub in the following format:

**Title**: [Brief description of the issue]

**Description**:
1. **Summary**: A detailed description of the issue.
2. **Steps to Reproduce**: Step-by-step instructions to reproduce the issue.
3. **Expected Result**: What you expected to happen.
4. **Actual Result**: What actually happened.
5. **Software**: Operating system and Python version.
6. **Additional Information**: Any other information that may help us resolve the issue.

Alternatively, you can reach out directly to Lucas Schneider at [lucas.schneider@cs.ox.ac.uk](mailto:lucas.schneider@cs.ox.ac.uk).


# Citation
When using the [BE-dataHIVE
API](https://be-datahive.com/documentation.html), please cite our paper as outlined below. 

```bibtex
@article{Schneider.2023,
    title = "BE-dataHIVE: a Base Editing Database",
    author = "Lucas Schneider, Peter Minary",
    year = "2024",
}
```
