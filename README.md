# StarGAN
StarGAN from scratch (pytorch)

> [Paper Review](https://inhopp.github.io/paper/Paper19/)

| Sample Image |
|:-:|
| ![sample_image](https://github.com/inhopp/inhopp/assets/96368476/369ebb99-ff41-41c3-aa72-3ad8f8ac4bbe)
 |


## Repository Directory 

``` python 
├── StarGAN
        ├── datasets
        │    
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── train.py
        ├── inference.py
        └── README.md
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/StarGAN.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --lr {}(default: 0.0002) \
    --n_epoch {}(default: 200) \
    --num_workers {}(default: 4) \
    --batch_size {}(default: 4) \ 
    --eval_batch_size {}(default: 4)
```

### testset inference
``` python
python3 inference.py
    --device {}(defautl: cpu) \
    --num_workers {}(default: 4) \
    --eval_batch_size {}(default: 4)
```


<br>


#### Main Reference
https://github.com/yunjey/stargan