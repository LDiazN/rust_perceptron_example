
use crate::util;

/// Base structure for a single perceptron
pub struct Perceptron {
    weight: util::NVector,
    bias: f64,
}

impl Perceptron {
    /// Compute the output of a single perceptron, as +1 or -1 depending on the 
    /// class
    pub fn compute(&self, input: &util::NVector ) -> f64 {
        if input.dim() != self.weight.dim() {
            panic!("Unmatching vector size");
        } 

        (input.dot(&self.weight) + self.bias).signum()
    }

    /// Creates a new perceptron biased & weighted to 0
    pub fn new() -> Perceptron {
        Perceptron {
            weight: ndarray::Array::from_elem(1, 0.),
            bias: 0.0
        }
    }

    /// Create a new perceptron consuming a vector with its weight and providing a bias
    pub fn from_vec(w: Vec<f64>, bias: f64) -> Perceptron {
        Perceptron {
            weight: ndarray::Array::from(w),
            bias
        }
    }
}
