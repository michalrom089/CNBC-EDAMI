# CNBC0EDAMI

Python CNBC algorithm implementation.


## Installation

1. Download dependencies

```
pip install -r requirements.txt
```


## Test

1. Set PYTHONPATH to include **cnbc** directory:
```
# from the repo level
export PYTHONPATH=$PYTHONPATH:./cnbc/
```

2. Download dev dependencies
```
pip install -r dev-requirements.txt
```
3. Run tests:
```
pytest test
```


Bibliography:
* http://ceur-ws.org/Vol-1269/paper113.pdf
* www.comsis.org/pdf.php?id=672-1806
* https://pdfs.semanticscholar.org/4055/cadb408eb092c4b1e70ae7279536524384f6.pdf

NOTE: Algorithm description differs slightly from the implementation included in that repo. There are some inaccuracies that are resolved in the [Jave implementation](https://github.com/piotrlasek/clustering/blob/master/src/org/dmtools/clustering/algorithm/CNBC/CDNBCRTree.java) written by the author himself.