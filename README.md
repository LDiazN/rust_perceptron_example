# Perceptron example made with Rust

This is a simple perceptron example made with Rust, based on the perceptron Convergence algoi=rithm 
and powered by NDArray, a crate for fast n-dimensional array operations.

This sample is based in the MNIST data set, so it will try to tell which digit
corresponds to a 28x28 grayscale hand-written number. This is clearly not a linearly separable problem,
so a single-layered network may not be the best solution, but it's a good data sample to test 
the perceptron convergence algorithm. 

# Network architecture

The network consists of 10 neurons, one for each possible digit. Each neuron can take an input 784-vector,
where every coordinate corresponds to one pixel of the 28x28 image. if the i-th neuron returns a positive answer,
then i is returned by the network.

# How to run the project

You can simply cd into the 'perceptron' folder and test it with  
      ```cargo run --release data_file.csv```

This will run the perceptron with the default values. The program will select a portion 
of the sample data passed in the file to test the network precision.

### Available flags:
You can run the program without args to get a help message with the following information:
  * ```-l arg```: Specify the learning rate (see perceptron convergence algorithm) (default is 0.01)
  * ```-e arg```: Specify the number of epochs (trainning cycles) to use in the perceptron trainning (default is 50)
  * ```-p arg```: Specify (in percentage) how much of the data given in the data file use to test the perceptron (default is 20)
    * e.g.: ```cargo run --release data_file.csv -p 10``` 
    
The provided file should be in the MNIST data set format, each line corresponds to an input sample, the first number corresponds to the desired
output, the rest of the numbers corresponds to the pixels in the image, every number is separated by commas

# Obtained results
These are some results we obtained with a portion of the MNIST data set. Every test used 50 epochs of training:

| Learning rate | Precision |
| ------------- |:---------:|
| 0.001         |   75.65%  |
| 0.01          |   77.9%   |
| 0.1           |   76.6%   |
