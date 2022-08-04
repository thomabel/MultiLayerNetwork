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
use rand::seq::SliceRandom;

use crate::print_data::{_print_matrix, _print_correct};


// MAIN
fn main() {
    let path_index = 4;
    let path = [
        "./data/test.csv",
        "./data/mnist_test - Copy.csv",
        "./data/mnist_test_short.csv",
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
    ];
    println!("Reading data, please be patient...");
    let result = read::read_csv(path[path_index], INPUT);
    let input;
    match result {
        Ok(mut o) => {
            o.0 /= DIVIDE;
            input = o;
        },
        Err(e) => {
            println!("{}", e);
            return;
        }
    }

    let _n = train_network(&input);

}

// Performs training on the network.
fn train_network(input: &(Array2<f32>, Array1<f32>)) -> Network {
    println!("Performing gradient descent... \n");

    // Create the network.
    let sizes = vec![
        layer::LayerSize::new(HIDDEN, INPUT, BATCH_SIZE),
        layer::LayerSize::new(OUTPUT, HIDDEN, BATCH_SIZE)
    ];
    let mut network = Network::new(&sizes, BATCH_SIZE);
    network.weight_randomize(WEIGHT_LOW, WEIGHT_HIGH);

    // Train the network.
    for e in 0..EPOCH {
        // Create index array for randomization of inputs.
        //let index_vec = random_index(input.1.len());

        // Perform the algorithm with all training input data.
        println!("===== EPOCH {} =====", e + 1);
        let confusion = network.gradient_descent(input, LEARNING_RATE, MOMENTUM_RATE);
        _print_matrix(&confusion.view(), "CONFUSION");
        _print_correct(&confusion.view());
    }
    
    println!("\nEnding training.");
    network
}

// Creates a vector of indices and shuffles them.
fn random_index(size: usize) -> Vec<usize> {
    let mut vec = Vec::new();
    for i in 0..size {
        vec.push(i);
    }
    vec.shuffle(&mut rand::thread_rng());
    vec
}
