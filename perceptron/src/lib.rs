mod perceptron;
mod test_suite;
mod network;
use ndarray::Array;
use std::fs;
// DEFAULT VALUES //

// Default value por -l flag
const L_DEFAULT: f64 = 0.01; 

// Default value for -e flag
const E_DEFAULT: u8 = 50;

// Default ammount of data to reserve to test
const PERCENTAGE_TO_TEST: f64 = 20.;

/// Helpful module to import some utility types
pub mod util {
    /////////////////////////////
    // TYPES
    /// utility type defining a n-vector 
    pub type NVector = ndarray::Array<f64, ndarray::Dim<[usize; 1]>>;

    /// utility type defing a two-dimensional matrix
    pub type Matrix  = ndarray::Array<f64, ndarray::Dim<[usize; 2]>>; 
    /////////////////////////////
    
}
 

/// A module containing application configuration logic
pub mod config {

    /// Simple struct to manage configuration data
    #[derive(Debug, PartialEq)]
    pub struct Config {
        pub learning_rate: f64,
        pub data_file: String,
        pub epochs: u8
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
                    None        => crate::L_DEFAULT,
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
                
                let err_noargs = "No argument for -e epochs flag";
                let err_unvalid_args = "Unvalid argument for -e epochs flag";
                
                match e_flag {
                    None => crate::E_DEFAULT,
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
}

// Function to perform the data loading, parsing, training and result comparision
pub fn run(config: config::Config) -> Result<(),&'static str> {
    use std::time; //required to measure duration

    let err_openning_file = "Could not open provided file";
    println!("Reading CSV file...");
    //Overal time
    let overall = std::time::Instant::now();

    let input = match fs::read_to_string(config.data_file) {
        Err(_) => return Err(err_openning_file),
        Ok(s)   => s
    };

    //Parse the input
    let data = parse(input)?;
    
    println!("Data loaded in {} ms", overall.elapsed().as_millis());
    println!("Trainning...");
    let training_time = time::Instant::now();
    // Train the model 
    let network = train(&data, config.epochs, config.learning_rate);

    println!("Trainning ended in {} ms", training_time.elapsed().as_millis());
    
    println!("Testing...");
    let testing_time = time::Instant::now();

    //Test the network
    println!("{}% of test passed", test_network(&network, &data.test_data));
    println!("Testing ended in {} ms", testing_time.elapsed().as_millis());
    println!("Total time elapsed: {}", overall.elapsed().as_millis());

    Ok(())
}
/// Simple struct to represent a training set with its desired output
#[derive(Debug)]
struct TrainSet {
    input_data: util::Matrix,
    desired_result: util::NVector,
    test_data: Vec<(util::NVector, f64)>
}

/// Function to parse a csv from a string into a TrainSet struct
fn parse(input: String) -> Result<TrainSet, &'static str> {
    //Parse the string into a matrix of numbers
    let input: Vec<Vec<f64>> = input  
        .split_ascii_whitespace() //Split lines
        .map(| line | //convert each string slice line into a vec of f64
            {
                let mut line: Vec<f64> = line
                    .split_terminator(',')
                    .map(| num |
                        num.parse::<f64>().expect("Error: unvalid CSV format")/255.
                    ).collect();

                line.push(1.);
                line
            }
            ).collect();
    
    //Compute training set
    let n_entries = (input.len() as f64 * PERCENTAGE_TO_TEST/100.) as usize;
    let mut input_it = input.iter();
    let mut count = 0;
    let mut test_data = Vec::with_capacity(n_entries);

    for entry in &mut input_it {
        if count >= n_entries { break; }
        let ans = entry[0] * 255.;
        let input = ndarray::Array::from(Vec::from(&entry[1..785]));

        test_data.push((input,ans));

        count += 1;
    }
    //Compute desired output vector        
    let mut desired = vec![];
    let mut n_samples = 0;
    for row in input_it.clone() {
        desired.push(row[0] * 255.);
        n_samples += 1;
    }

    //Compute training matrix
    let input: Vec<f64> = input_it
        .map(| v | Vec::from(&v[1..]))
        .flatten()
        .collect();
    
    Ok(
        TrainSet{
            input_data: Array::from_shape_vec((n_samples, 785), input).expect("Error: unvalid CSV format"),
            desired_result: Array::from(desired),
            test_data
        } 
    )
}

/// Function to train a network based in a TrainSet, for a given number of epochs and learning_rate
fn train (ts: &TrainSet, epochs: u8, learning_rate: f64) -> network::Network {
    //Create weight matrix
    let mut weights = Array::from_shape_fn((785,10), |_| rand::random()); //@TODO la matriz tiene que inicializar con numeros random
    let desired_output = Array::from_shape_fn((ts.desired_result.len(), 10), |(i,j)| {
        if ts.desired_result[i] == (j as f64) {
            1.
        }
        else {
            -1.
        }
    });

    for _ in 0..epochs {
        let output = ts.input_data
                        .dot(&weights)
                        .map(| f | f.signum());
        
        for (j, col) in (0..10).map(| j | (j,output.column(j)) ) {
            for (i,val) in col.iter().enumerate() {
                let factor = learning_rate * (desired_output[[i,j]] - val);
                if factor != 0. {
                    weights
                        .column_mut(j)
                        .iter_mut()
                        .zip(ts.input_data.row(i))
                        .for_each(|(w,inp)| *w = *w + factor * inp )
                }
            }
        }
    } 

    let weights: Vec<perceptron::Perceptron> = 
    (0..10)
        .map(|i| {
            perceptron::Perceptron::from_vec(   
                weights
                    .slice(ndarray::s![..784,i])
                    .to_vec()
                , weights[[784, i]])
        })
        .collect();
    
    network::Network::new(weights)
} 
/// Test a trained network with a set of test and return the percentage of hits
fn test_network(network: &network::Network, test_data: &Vec<(util::NVector, f64)>) -> f64 {

    100. * test_data
        .iter()
        .map(|(v,y)| if (network.predict(v).unwrap_or(99) as f64) == *y { 1. } else { 0. })
        .sum::<f64>()/(test_data.len() as f64)
}