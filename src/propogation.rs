use crate::network::*;
use ndarray::prelude::*;

// Sigmoid function + derivative
fn sigmoid(input: f32) -> f32 {
    1. / (1. + f32::exp(-input))
}
fn sigmoid_derivative(input: f32, factor: f32) -> f32 {
    input * (1.0 - input) * factor
}

// Passes input into the chain of hidden layers until the final layer us used.
pub fn forward(network: &mut Network, input: & ArrayView1<f32>, batch: usize) {
    let mut iter = network.layers.iter_mut();
    
    let mut layer_prev = iter.next().unwrap();
    let mut input_inner;
    let mut weights = &layer_prev.weights;
    let mut result 
            = layer_prev.results.index_axis_mut(Axis(1), batch);
    print!("Initial: ");
    feed(input, &weights.view(), &mut result);

    for layer in iter{
        input_inner = layer_prev.results.index_axis(Axis(1), batch);
        weights = &layer.weights;
        result = layer.results.index_axis_mut(Axis(1), batch);
        print!("Layer: ", );
        feed(&input_inner, &weights.view(), &mut result);
        layer_prev = layer;
    }
}

// Finds all results between two layers.
fn feed(
    input: &ArrayView1<f32>, 
    weight: &ArrayView2<f32>, 
    output: &mut ArrayViewMut1<f32>,
) {
    let len = input.len() - 1;
    // For each output node
    for j in 0..output.len() {
        // Dot product with input values
        let mut solution: f32 = 0.0;
        for i in 0..len {
            let wi = weight[[j, i]] * input[i];
            solution += wi;
        }
        // Bias
        solution += weight[[j, len]];
        // use sigmoid
        solution = sigmoid(solution);
        print!("{}, ", solution);

        output[j] = solution;
    }
    println!();
}