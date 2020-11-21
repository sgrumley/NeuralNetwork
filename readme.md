# Neural Network

A neural network built without any machine learning libraries

This project used the included MNIST data set to predict handwritten digits.


## Setup

Install dependencies  

```bash
pip install numpy
```

## Usage
run the program passing in the number of nodes in the three layers, training data and test data

```bash
python3 digitfinal.py 784 30 10 TrainDigitX.csv.gz TrainDigitY.csv.gz TestDigitX.csv.gz TestDigitY.csv.gz
```
For alternate configuration, see the bottom of NN.py 


## License
[MIT](https://choosealicense.com/licenses/mit/)