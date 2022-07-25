use crate::constants::START;

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
use constants::*;

// MAIN
fn main() {
    // Read file for data.
    println!("Reading data, please be patient...");
    let path_index = 1;
    let path = [
        "./data/mnist_test - Copy.csv",
        "./data/mnist_test_short.csv",
        "./data/mnist_train.csv",
        "./data/mnist_test.csv",
    ];
    let input: Vec<data::Input>;
    let temp = read::read(path[path_index]);
    match temp {
        Ok(v) => {
            input = v;
            println!("SUCCESS: Data read")
        }
        Err(e) => {
            print!("ERROR: {}", e);
            return;
        }
    }

    println!("Performing gradient descent...");

    let mut start = 0;
    let mut end = input.len();

    if START != None {
        start = START.unwrap();
    }
    if END != None {
        end = END.unwrap();
    }
    let mut network = data::Network_::default();
    neural_network::gradient_descent(
        &input[start..end],
        constants::EPOCH,
        &mut network
    );

    println!("\nEnding training.");
}
