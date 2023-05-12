This repository contains an implementation of a peer-to-peer (P2P) federated learning program. The program allows various clients to collaboratively learn a machine learning model while keeping all the training data on the original client, thus decoupling the ability to do machine learning from the need to store the data.

## Usage
Deploy to each node and run `python3 P2PCNN.py` to start the program. The program will automatically connect to other nodes in the network and start training the model.

## Dependencies
* Python 3.6+
* PyTorch 1.0+

## Config
The program can be configured by modifying the `config` file. The following parameters are available:

* `Protocol.Default`: The default protocol to use. Currently, `P2P` and `MQTT` is supported.
* `Protocol.P2P.Port`: The port to use for the P2P protocol.
* TBC