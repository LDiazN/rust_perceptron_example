use crate::*;

#[test]
fn test_perceptron_computation(){
    // Check correct computation (inner product)
    let input = ndarray::Array::from(vec![1.,2.,3.]);
    let bias  = 10.;
    let p     = crate::perceptron::Perceptron::from_vec(vec![1.; 3], bias);

    assert_eq!((1. + 2. + 3. + bias).signum() , p.compute(&input) );
}

#[test]
fn test_config_input1 () {
    //Check correct case 1
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

#[test]
fn test_config_input2 () {
    // check correct case with other order
    let input = [
        String::from("perceptron"),
        String::from("test_data.csv"),
        String::from("-e"),
        String::from("50"),
        String::from("-l"),
        String::from("0.01")
    ];
    let expected: Result<crate::Config, &str> = Ok(crate::Config {
        learning_rate: 0.01,
        epochs: 50,
        data_file: String::from("test_data.csv")
    });

    assert_eq!(expected, crate::Config::new(&input));
}

#[test]
fn test_config_input3 () {
    // Check no arg error
    let input = [
        String::from("perceptron"),
        String::from("test_data.csv"),
        String::from("-e"),
        String::from("50"),
        String::from("0.01"),
        String::from("-l"),
    ];
    let expected = Err("No argument for -l learning rate flag");

    assert_eq!(expected, crate::Config::new(&input));
}

#[test]
fn test_config_input4 () {
    // Check no file error
    let input = [
        String::from("perceptron"),
        String::from("nofile.csv"),
        String::from("-e"),
        String::from("50"),
        String::from("-l"),
        String::from("0.01"),
    ];
    let expected = Err("Error openning training file: there's no such file");

    assert_eq!(expected, crate::Config::new(&input));
}

#[test]
fn test_config_input5 () {
    // Check unvalid learning rate range
    let input = [
        String::from("perceptron"),
        String::from("test_data.csv"),
        String::from("-e"),
        String::from("50"),
        String::from("-l"),
        String::from("10.0"),
    ];
    let expected = Err("Learning rate -l argument should be in range [0,1]");

    assert_eq!(expected, crate::Config::new(&input));
}
