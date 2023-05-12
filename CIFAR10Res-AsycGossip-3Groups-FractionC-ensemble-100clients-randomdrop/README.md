This project implements an asynchronous federated learning algorithm, meaning it doesn't need to wait for the progress of other nodes, and new nodes can join or leave at any time. This implementation also validates its accuracy under several datasets.

## Usage
* Install requirements: `pip install -r requirements.txt`
* Run `python3 main.py`

## Config
Change config in parameter.py:
* `data_set`: The dataset to use. Currently, `CIFAR10` and `MNIST` is supported.
* `num_clients`: Number of clients in the network
* `iid`: Whether the data is distributed in an IID manner
* `epochs`: Number of epochs to train