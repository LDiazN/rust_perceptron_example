use ::perceptron::{config, run};
use std::env;

fn main() {

    let args: Vec<String> = env::args().collect();
    let usage = "Usage: 
        perceptron filename [options]
        where options:
            -l arg: Specify a learning 'arg' rate in the range [0,1]
            -e arg: Specify a number of epochs 'arg' in the range [0,255] to train
            -p arg: Specify a percentage 'arg' of sample inputs to use to test the network 
    ";
    let config = config::Config::new(&args);
    
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
}
