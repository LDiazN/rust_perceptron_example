use crate::perceptron;
use crate::util;

#[derive(Debug)]
pub struct Network{
    neurons: Vec<perceptron::Perceptron>
}

impl Network {
    pub fn predict(&self, input: &util::NVector) -> Result<u8,&'static str> {
        if input.len() != 28*28 {panic!("Input length not matching required image size 28*28")}
        for (i,neuron) in self.neurons.iter().enumerate() {
            if (neuron.compute(input)) == 1. {
                return Ok(i as u8);
            }
        }

        return Err("Could not tell what number is this");
    }

    pub fn new(neurons: Vec<perceptron::Perceptron> ) -> Network {
        if neurons.len() != 10 {
            panic!("Error creating network, unvalid number of neurons. Expected: {}, found: {}", 10, neurons.len());
        }
        for (i, neuron) in neurons.iter().enumerate() {
            if neuron.weight.len() != 28*28 {
                panic!("error creating network: Unvalid size for neuron weight,
required weight = 28*28, actual weight: {} at neuron  {}", neuron.weight.len(), i);
            } 
        }

        Network {
            neurons
        }
    }
}