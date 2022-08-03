# GS-VQA
This repository contains the code of GS-VQA model implementation 
(version 2 with Unified estimator) described in 
*Graph Strategy for Interpretable Visual Question Answering* paper.



![text](./model_pic/gs-vqa_cad.png)

## Environment 
To set up the environment run the commands below.
```
git clone https://github.com/cds-mipt/x-vqa
```
1. Anaconda environment:

   ```
   conda create env --name gs-vqa
   conda activate gs-vqa
   conda install pytorch=1.10.2 torchvision=0.11.* cudatoolkit -c pytorch
   ```
2. Install dependencies for VL-BERT. Inside [vlbert](./vlbert) folder you can find the files from original repo with [VL-BERT](https://github.com/jackroos/VL-BERT) implementation.

   ```
   cd vlbert
   pip install -r requirements.txt
   pip install Cython
   pip install pyyaml==5.4.1
   ```
3. Build the library for VL-BERT:
   
   ```
   ./scripts/init.sh
   ```
 
 ## Data processing
To run UnCoRd-VL on custom dataset, you need to prepare the following data:
 1. **Question-to-graph** model checkpoint - to be placed in [ende_ctranslate2](./ende_ctranslate2) folder.
 2. **Faster-RCNN** model checkpoint - to be placed in [estimators](./estimators) folder.
 3. **VL-BERT** checkpoint - to be placed in [vlbert/model/pretrained_model](./vlbert/model/pretrained_model) folder.
 4. List of property names and their possible values
 
    You need to prepare a .txt file that lists all possible categorical properties in the dataset and the values they can take. Each line must be completed in the following format:
    
    ```
    property_name value_1 ... value_n
    ```
    
    [An example file with properties for CLEVR.](properties_file.txt)
 5. VL-BERT answers vocabulary

    Answers for VL-BERT is a set consisting of all possible values of all properties from the given dataset, as well as the words 'yes' and 'no'. [An example of VL-BERT vocabulary for CLEVR.](answer_vocab_file.txt)
 
 6. Image directory
 7. JSON file with indexes of questions, texts of questions and indexes of images that correspond to these questions (answers are optional for test mode).

    ```
    {'questions': [{'question_index': 0, 'question': 'What is the color of the large metal sphere?', 'image_index': 0, 'answer': 'brown'}, ... ]}
    ```
    Functions for extracting questions: [dataset.py](dataset.py).
    
In addition, to work with VL-BERT, you need to download pretrained [BERT](https://drive.google.com/file/d/14VceZht89V5i54-_xWiw58Rosa5NDL2H/view?usp=sharing) and [ResNet-101](https://drive.google.com/file/d/1qJYtsGw1SfAyvknDZeRBnp2cF4VNjiDE/view?usp=sharing) and put them in folders [vlbert/model/pretrained_model/bert-base-uncased](./vlbert/model/pretrained_model/bert-base-uncased) и [vlbert/model/pretrained_model](./vlbert/model/pretrained_model) respectively.
 
 ## Model evaluation and testing
 
The script should be launched from the root folder.

Testing model example:
 ```
 python main.py --image_dir IMAGE_DIR \
 --questions_file QUESTIONS_DIR/file_with_questions_and_images_indices.json \
 --test_mode True --device "cuda" \
 --answer_vocab_file answer_vocab_file.txt \
 --properties_file properties_file.txt
 ```
 The script outputs the file with model answers written line by line that can be 
used later for model evaluation.
 ## Pre-trained models

Here you can find pre-trained models for CLEVR question answering. 
Download them and place in the appropriate folders indicated in 
[Data processing](#data-processing).

- [CLEVR Question-to-graph](https://drive.google.com/file/d/1lBIaJ9ha8dPbCYbb52ezsJGcyPmGqpg-/view?usp=sharing)
- [CLEVR Object detector](https://drive.google.com/file/d/1B7d7ZNRxRKNR-4DwLHUMCEEAcQt5NGsF/view?usp=sharing)
- [CLEVR VL-BERT](https://drive.google.com/file/d/11tJzVUBqrQsXYd2Go0ZR3YwJDjzaE29l/view?usp=sharing)


