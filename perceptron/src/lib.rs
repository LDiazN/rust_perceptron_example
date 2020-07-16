// DEFAULT VALUES //

// Default value por -l flag
const L_DEFAULT: f64 = 0.01; 

// Default value for -e flag
const E_DEFAULT: u8 = 50;

/////////////////////////////

// Base structure for a single perceptron
pub mod perceptron {

    /// utility type defining a n-vector 
    type NVector = ndarray::Array<f64, ndarray::Dim<[usize; 1]>>;

    pub struct Perceptron {
        weight: NVector,
        bias: f64,
    }

    impl Perceptron {
        /// Compute the output of a single perceptron, as +1 or -1 depending on the 
        /// class
        pub fn compute(&self, input: &NVector ) -> f64 {
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
}

/// Simple struct to manage configuration data
#[derive(Debug, PartialEq)]
pub struct Config {
    learning_rate: f64,
    data_file: String,
    epochs: u8
}

impl Config {
    /// Based on the input format: 
    ///     usage: 
    ///         perceptron filename [options]
    ///     where options:
    ///         -l arg: specify the learning rate 'arg' (should be a float in [0,1])
    ///         -e arg: Specify the number of epochs to train (should be an int in [0,255]) 
    pub fn new(params: &[String]) -> Result<Config, &'static str> {

        let learning_rate = {
            let l_flag = params
                            .iter()
                            .enumerate()
                            .find(|(_, param)| *param == &String::from("-l"));

            let err_noargs = "No argument for -l learning rate flag";
            let err_unvalid_args = "Unvalid argument for -l learning rate flag";
            let err_unvalid_lr = "Learning rate -l argument should be in range [0,1]";

            match l_flag {
                None        => L_DEFAULT,
                Some((i, _))  => {
                    if i == (params.len() - 1) {
                        return Result::Err(err_noargs);
                    }
                    else {
                        match params[i + 1].parse::<f64>() {
                            Ok(f) => if f > 1. || f < 0. {
                                        return Err(err_unvalid_lr)
                                    }
                                    else {
                                        f
                                    },
                            Err(_) => return Result::Err(err_unvalid_args)
                        }
                    }
                }
            }
        };

        let epochs = {
            let e_flag = params
                            .iter()
                            .enumerate()
                            .find(|(_, param)| *param == &String::from("-e"));
            
            let err_noargs = "No argument for -e learning rate flag";
            let err_unvalid_args = "Unvalid argument for -e learning rate flag";
            
            match e_flag {
                None => E_DEFAULT,
                Some((i,_)) => if i + 1 == params.len() {
                    return Err(err_noargs);
                } 
                else {
                    match params[i+1].parse::<u8>() {
                        Err(_)  => return Err(err_unvalid_args),
                        Ok(f)   => f
                    }
                }
            }
            
        };

        let data_file = {
            let err_no_filename_provided = "No training data filename provided";
            let err_no_such_file_or_dir = "Error openning training file: there's no such file";

            if params.len() < 2 {
                return Err(err_no_filename_provided)
            }
            use std::path;
            if !path::Path::new(&params[1]).exists() {
                return Err(err_no_such_file_or_dir) 
            }

            params[1].clone()
        };

        Result::Ok(
            Config {
                learning_rate,
                data_file,
                epochs
            }
        )  
        
    }    
}

mod test_suite {
    #[test]
    fn test_perceptron_computation(){
        let input = ndarray::Array::from(vec![1.,2.,3.]);
        let bias  = 10.;
        let p     = crate::perceptron::Perceptron::from_vec(vec![1.; 3], bias);

        assert_eq!((1. + 2. + 3. + bias).signum() , p.compute(&input) );
    }

    #[test]
    fn test_config_input1 () {
        let input = [
            String::from("perceptron"),
            String::from("test_data.csv"),
            String::from("-l"),
            String::from("0.01"),
            String::from("-e"),
            String::from("50")
        ];
        let expected: Result<crate::Config, &str> = Ok(crate::Config {
            learning_rate: 0.01,
            epochs: 50,
            data_file: String::from("test_data.csv")
        });

        assert_eq!(expected, crate::Config::new(&input));
    }
}