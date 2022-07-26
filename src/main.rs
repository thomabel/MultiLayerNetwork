/*
    Programming #1
    Neural Network

    Name: Thomas &Abel
    Date: 2022-07-10
    Class: Machine Learning
*/
mod constants;
mod data;
mod network;
mod print_data;
mod read;

use constants::*;
use data::Input;
use ndarray::prelude::*;


// MAIN
fn main() {
    let result = read_toy_input();
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
    gradient_descent(&input);
}

fn read_toy_input() -> Result<(Array2<f32>, Array1<f32>), &'static str> {
    // Read file for data.
    println!("Reading data, please be patient...");
    let path = "./data/test.csv";
    let input = read::read_toy(path, 2);
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

fn read_image_input() -> Result<Array1<Input>, &'static str> {
    // Read file for data.
    println!("Reading data, please be patient...");
    let path_index = 1;
    let path = [
        "./data/mnist_test - Copy.csv",
        "./data/mnist_test_short.csv",
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
    ];
    let inputs = INPUT;
    let dividend = DIVIDE;
    let temp
        = read::read_image(path[path_index], inputs, dividend);

    match temp {
        Ok(v) => {
            println!("SUCCESS: Data read");
            Ok(Array1::<Input>::from_vec(v))
        }
        Err(_e) => {
            let str = "Could not read.";
            Err(str)
        }
    }
}



fn gradient_descent(input: &(Array2<f32>, Array1<f32>)) {
    println!("Performing gradient descent... \n");
    // Goes Output, Input
    // Add +1 on input for bias at the end
    // Currently on 2 layers
    let sizes = vec![
    network::LayerSize::new(HIDDEN, INPUT + 1, BATCHES),
    network::LayerSize::new(OUTPUT, HIDDEN + 1, BATCHES)];
    let mut network = network::Network::new(&sizes);
    network.set_weights(0.1);

    let mut total = 0;
    for i in input.0.rows() {
        total += 1;
        println!("=== BATCH: {}, INPUT: {} ===", network.batch, total);
        network.forward(&i);

        network.batch += 1;
        if network.batch == BATCHES {
            // Do backprop
            network.backward(&input.1);
            network.batch = 0;
        }
    }

    println!("\nEnding training.");
}