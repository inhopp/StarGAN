# StarGAN
StarGAN from scratch (pytorch)

> [Paper Review](https://inhopp.github.io/paper/Paper19/)

| Original Hair | Black Hair |
|:-:| :-: |
| ![1](https://github.com/inhopp/inhopp/assets/96368476/a0269bfb-849c-4923-ac3c-aafc41996df2) | ![2](https://github.com/inhopp/inhopp/assets/96368476/63b6281c-e838-4bbd-90fc-dd08672dd38b)
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