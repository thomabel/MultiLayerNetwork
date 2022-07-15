use crate::constants::START;

/*
    Programming #1
    Neural Network

    Name: Thomas &Abel
    Date: 2022-07-10
    Class: Machine Learning
*/
mod constants;
mod neural_network;
mod read;
mod print_data;
mod data;

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
    let input: Vec<read::Input>;
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
    neural_network::gradient_descent(&input[start..end], constants::EPOCH);

    println!("\nEnding training.");
}
