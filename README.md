# DuetCS

## Dependencies
- Python 3.5.2
- MongoDB 4.2.12
- pymongo 3.9.0
- ANTLR4 (Java target)
- TensorFlow 1.10.0
- Numpy 1.13.3
- Keras 2.2.2
- PyTorch 1.5.1

## Prepare data and generating feature embedding

encode the original raw code to initial vector

`python3 data_prepare.py`

generate the initial feature embedding

`python3 feature.py`

## Train siamese network for the classification task

`python3 siamese.py`

`python3 training.py`

## Create feature embedding for input code/code examples in target style

`python3 testing.py path_to_code`

## Output results through two modes

generate transferred code through LSTM

`python 3 generation.py`

retrieve transferred code for the database and compare with geneartion mode to output the final result

`sh translate_retrieval.sh path_to_input target_language`

