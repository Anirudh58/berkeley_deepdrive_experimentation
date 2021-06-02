# berkeley_deepdrive_experimentation

## Requirements
- PyTorch installed and configured with a decent GPU

## Approach
- [Dataset Preparation](https://github.com/Anirudh58/berkeley_deepdrive_experimentation/blob/master/dataset_preparation.ipynb) from [Berkeley Deep Drive](https://bdd-data.berkeley.edu/) to 
shortlist 1000 good quality images (900 for train and 100 for validation).
- [Fine-tuned](https://github.com/Anirudh58/berkeley_deepdrive_experimentation/blob/master/train.ipynb) a FasterRCNN model with specific target labels
- [Evaluate](https://github.com/Anirudh58/berkeley_deepdrive_experimentation/blob/master/video_queries.ipynb) by issuing sample queries to a test video taken from the same dataset 

## Demo

- Sample query: "Sample every 3 seconds and get all timestamps where there are more than 2 cars, 1 sign, 1 pedestrian"
- Momentary change in border color to green indicates the frame has satisfied the query

![alt text](https://github.com/Anirudh58/berkeley_deepdrive_experimentation/blob/master/model_test_1.gif "Demo")
