# TeroACT: Constructing terpenoid-bioactivity knowledge graph for drug discovery

- Terpenoid-disease association (TerDA) prediction model as an effective prioritization tool for screening potential disease associations, and facilitating follow-up studies such as drug repositioning and the prediction of new indications for terpenoids.

- This repository contains the source code ,the data and trained models.




## Train
Model training can be started by running the `run.py` script:
```bash
python run.py --gpu 0 --epoch 300 --neg_ratio 1 
```
- `--neg_ratio` is required here to specify the ratio between negative and positive data

**Notes**:

You can utilize `./utils/condition.py` to generate the required molecular descriptor files in advance, such as: `./dataset/description.csv`



## Predict

- You can run the `prediction.py` script to reproduce the predicted results of compounds in web experiment as described in the article. Such as :

```bash
python prediction.py
```


**Notes**:

For the disease prediction, you can access our user-friendly web server ( [TeroACT](http://terokit.qmclab.com/teroact/)). 
