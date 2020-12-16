# EDAMI

1. pseudo-code


2. main assumptions (data structures, list/hashlist etc)


3. datasets (smaller testing, bigger for training)


4. how to visiualize the data


5. create github repo



Bibliography:
http://ceur-ws.org/Vol-1269/paper113.pdf
www.comsis.org/pdf.php?id=672-1806
by friday


## Preparation

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