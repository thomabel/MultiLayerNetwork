/*
    Programming #1
    Neural Network

    Name: Thomas &Abel
    Date: 2022-07-10
    Class: Machine Learning
*/
mod constants;
mod read;
mod layer;
mod network;
mod print_data;

use constants::*;
use ndarray::prelude::*;
use network::Network;


// MAIN
fn main() {
    let path_index = 0;
    let path = [
        "./data/test.csv",
        "./data/mnist_test - Copy.csv",
        "./data/mnist_test_short.csv",
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
    ];
    let result = read_file(path[path_index]);
    let input;
    match result {
        Ok(o) => {
            input = o;
        },
        Err(e) => {
            println!("{}", e);
            return;
        }
    }
    let _n = gradient_descent(&input);
    // Use network from here to test data.
}

// Reads from a .csv file to get input values.
fn read_file(path: &str) -> Result<(Array2<f32>, Array1<f32>), &'static str> {
    println!("Reading data, please be patient...");
    let inputs = INPUT;
    let input 
        = read::read_csv(path, inputs);
    match input {
        Ok(v) => {
            println!("SUCCESS: Data read");
            Ok(v)
        }
        Err(_e) => {
            let str = "Could not read.";
            Err(str)
        }
    }
}

// Performs training on the network.
fn gradient_descent(input: &(Array2<f32>, Array1<f32>)) -> Network {
    println!("Performing gradient descent... \n");
    let sizes = vec![
        layer::LayerSize::new(HIDDEN, INPUT, BATCHES),
        layer::LayerSize::new(OUTPUT, HIDDEN, BATCHES)
    ];
    let mut network = Network::new(&sizes, RATE, MOMENTUM, BATCHES);
    network.set_weights(0.1);
    network.gradient_descent(input);
    println!("\nEnding training.");
    network
}