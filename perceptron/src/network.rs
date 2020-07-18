/*
    This module provides the implementation of a network to decide
    which number corresponds to a 28*28 image in grayscale hand-written
    digit. It's built over the implementation of the 'perceptron' module
*/

use crate::perceptron;
use crate::util;

/// Base structure to represent our network, consisting on 
/// 10 neurons, each for each digit. When an input comes, the first
/// neuron to output 1 to this input corresponds to the response digit.
/// e.g. if the third neuron outputs 1, then the output will be 2 (0,1,2,...,9)
#[derive(Debug)]
pub struct Network{
    neurons: Vec<perceptron::Perceptron>
}

impl Network {

    /// Given an input vector consisting of 28*28 pixels, return the predicted digit for this input
    pub fn predict(&self, input: &util::NVector) -> Result<u8,&'static str> {
        if input.len() != 28*28 {panic!("Input length not matching required image size 28*28")}
        for (i,neuron) in self.neurons.iter().enumerate() {
            if (neuron.compute(input)) == 1. {
                return Ok(i as u8);
            }
        }

        return Err("Could not tell what number is this");
    }

    /// Create a new neuron from a perceptron vector. Panics if the dimension of the perceptron
    /// vector or the dimension of each perceptron doesn't fits the requirements of our 
    /// number-deciding network
    pub fn new(neurons: Vec<perceptron::Perceptron> ) -> Network {
        if neurons.len() != 10 {
            panic!("Error creating network, unvalid number of neurons. Expected: {}, found: {}", 
                    10, 
                    neurons.len());
        }
        for (i, neuron) in neurons.iter().enumerate() {
            if neuron.weight.len() != 28*28 {
                panic!("error creating network: Unvalid size for neuron weight,
required weight = 28*28, actual weight: {}, at neuron  {}", neuron.weight.len(), i);
            } 
        }

        Network {
            neurons
        }
    }
}