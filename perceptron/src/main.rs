use ::perceptron::{config, run};
use std::env;
fn main() {

    let args: Vec<String> = env::args().collect();
    let config = config::Config::new(&args);
    let usage = "Usage: 
        perceptron filename [options]
        where options:
            -l arg: Specify a learning 'arg' rate in the range [0,1]
            -e arg: Specify a number of epochs 'arg' in the range [0,255] to train
    ";
    
    match config {
        Err(e)     => {
            eprintln!("Error, unvalid usage: {}", e);
            eprintln!("{}", usage);
            std::process::exit(1);
        },
        Ok(config) => match run(config) {
            Err(e) => {
                eprintln!("Error running perceptron algorithm: {}", e);
                std::process::exit(1);
            },
            Ok(_) => { }
        }
    }
    //let mut m: perceptron::util::Matrix = ndarray::Array::zeros((10,10));
    //let v : perceptron::util::NVector = ndarray::Array::ones(10);
    //let mut col = m.column_mut(0);
    //
    //col.iter_mut().enumerate().map(| (i,x) | *x = *x + v[i]);

}
