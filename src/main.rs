/*
    Programming #1
    Neural Network

    Name: Thomas &Abel
    Date: 2022-07-10
    Class: Machine Learning
*/
mod constants;
mod data;
mod neural_network;
mod print_data;
mod read;
mod network;
mod propogation;
use constants::*;
use data::Input;
use ndarray::prelude::*;


// MAIN
fn main() {
    let result = read_input();
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

fn read_input() -> Result<Array1<Input>, &'static str> {
    // Read file for data.
    println!("Reading data, please be patient...");
    let path_index = 1;
    let path = [
        "./data/mnist_test - Copy.csv",
        "./data/mnist_test_short.csv",
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
    ];
    let input;
    let inputs = INPUT;
    let dividend = DIVIDE;
    let temp
        = read::read_image(path[path_index], inputs, dividend);

    match temp {
        Ok(v) => {
            input = v;
            println!("SUCCESS: Data read");
            Ok(Array1::<Input>::from_vec(input))
        }
        Err(_e) => {
            let str = "Could not read.";
            Err(str)
        }
    }
}

fn gradient_descent(input: &Array1<Input>) {
    println!("Performing gradient descent...");
    // Goes Output, Input
    // Add +1 for bias at the end
    // Currently on 2 layers
    let sizes = vec![
    network::LayerSize::new(HIDDEN, INPUT + 1),
    network::LayerSize::new(OUTPUT, HIDDEN + 1)];
    
    let mut network = network::Network::new(&sizes, BATCHES);
    let mut batch = 0;
    let mut total = 0;
    for i in input {
        total += 1;
        println!("BATCH: {}, INPUT: {}", batch, total);
        propogation::forward(&mut network, &i.data.view(), batch);
        println!();

        // Track number in batch of inputs.
        if batch >= BATCHES - 1 {
            batch = 0;
        }
        else {
            batch += 1;
        }
    }

    println!("\nEnding training.");
}