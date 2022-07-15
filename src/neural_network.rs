use crate::constants::*;
use crate::print_data::*;
use crate::read::Input;
use crate::data::*;
use ndarray::prelude::*;


// Main function that performs the actual gradent descent needed to train the neural network.
pub fn gradient_descent(input: &[Input], epochs: usize) {
    let mut network = Network::default();
    //_print_matrix(&network.weight_oh, "weight_oh");

    let mut correct_all_epochs = 0;
    let mut total_inputs_all_epochs = 0;
    for _e in 0..epochs {
        // Run an epoch.
        let mut correct_cur_epoch = 0;
        let mut batch = 0;
        for i in 0..input.len() {
            //_print_vector(&input[i].data.view());

            // Input to Hidden layer
            forward_propagation(
                &input[i].data.view(),
                &network.weight_hi.view(),
                &mut network.result_h.index_axis_mut(Axis(0), batch),
            );

            // Hidden to Output layer
            forward_propagation(
                &network.result_h.index_axis(Axis(0), batch),
                &network.weight_oh.view(),
                &mut network.result_o.index_axis_mut(Axis(0), batch),
            );

            if false {
                _print_matrix(&network.result_h, "result_h");
                _print_matrix(&network.result_o, "result_o");
            }

            // Determine what the network predicted.
            network.outcome[batch] = Outcome {
                target: input[i].t,
                prediction: find_prediction(&network.result_o.index_axis(Axis(0), batch)),
            };

            // If we've hit the batch limit, backpropagate.
            if batch >= BATCHES - 1 {
                let start = i - (BATCHES - 1);
                let end = i + 1;

                // Check results
                if false {
                    _print_matrix(&network.result_h, "result_h");
                    _print_matrix(&network.result_o, "result_o");
                }
                backward_propagation(&mut network, &input[start..end]);

                // Print predictions
                let c = batch_correct(&network.outcome);
                correct_cur_epoch += c;
                correct_all_epochs += c;
                total_inputs_all_epochs += BATCHES as i32;

                let total_inputs: i32 = i as i32 + 1;
                let batch_number = i / BATCHES + 1;

                println!("\nAFTER BATCH {}:", batch_number);
                _print_outcome(&network.outcome);
                print!("Total ");
                _print_total_error(correct_cur_epoch, total_inputs);

                batch = 0;
            } else {
                batch += 1;
            }
        }
        // End of Epoch
        print!("Total for Epoch {} ", _e + 1);
        _print_total_error(correct_all_epochs, total_inputs_all_epochs)
    }
}

// Returns the number of correct predictions in a batch.
fn batch_correct(batch: &Array1<Outcome>) -> i32 {
    let mut result = 0;
    for b in batch {
        if b.is_correct() {
            result += 1;
        }
    }
    result
}

// Sigmoid function
fn sigmoid(input: f32) -> f32 {
    1. / (1. + f32::exp(-input))
}
// Feed forwards the data through the network.
fn forward_propagation(input: &ArrayView1<f32>, weight: &ArrayView2<f32>, output: &mut ArrayViewMut1<f32>) {
    let print = false;
    if print {
        println!("\nFORWARD PROPAGATION: ")
    }
    let inlen = input.len();

    // For each output node
    for j in 0..output.len() {
        let mut sum: f32 = 0.0;
        // Dot product with input values
        for i in 0..inlen {
            let wi = weight[[j, i]] * input[i];
            sum += wi;
            //print!("{}, ", wi);
        }
        // Bias weight exists at end of array.
        let bias = weight[[j, inlen]];
        //println!("bias: {}", bias);
        if print {
            print!("{}, ", sum);
        }
        sum += bias;

        output[j] = sigmoid(sum);
        //if print {print!("sig: {}, ", output[j]);}
    }
    if print {
        println!();
    }
}
// Predicts the class from the outputs.
fn find_prediction(input: &ArrayView1<f32>) -> char {
    let mut index = 0;
    let mut value = -1.0;
    for i in 0..input.len() {
        if input[i] > value {
            index = i;
            value = input[i];
        }
    }
    TARGET[index]
}

// Error functions
fn sigmoid_derivative(input: f32, factor: f32) -> f32 {
    input * (1.0 - input) * factor
}
fn determine_error_o(outcome: &ArrayView1<Outcome>, result: &ArrayView2<f32>) -> Array2<f32> {
    let mut error = Array2::<f32>::zeros((BATCHES, OUTPUT));
    //println!("error_o:");
    for b in 0..BATCHES {
        // ground truth == predicted class, outcome
        /*
        outcome: array[image0, image1, ....]
        image0 = target: 1, predicted: 7
        outcome: [(1, 7)]
        gt/outcome [0.9, 0.1, 0.1 ....]
        result [0.3, 0.5 .... ]
        */
        //print!("Batch {}: ", b);
        for o in 0..OUTPUT {
            let t: f32 = if outcome[b].target == TARGET[o] {
                0.9
            } else {
                0.1
            };
            let bo = result[[b, o]];
            error[[b, o]] = sigmoid_derivative(bo, t - bo);
        }
        //println!();
    }
    //println!();
    error
}
fn determine_error_h(result: &ArrayView2<f32>, weight: &ArrayView2<f32>, error_o: &Array2<f32>) -> Array2<f32> {
    let mut error = Array2::<f32>::zeros((BATCHES, HIDDEN));
    for b in 0..BATCHES {
        for j in 0..HIDDEN {
            // Sum all weights times their output to this hidden node.
            let mut sum: f32 = 0.0;
            for k in 0..OUTPUT {
                sum += weight[[k, j]] * error_o[[b, k]];
            }
            let h = result[[b, j]];
            error[[b, j]] = sigmoid_derivative(h, sum);
        }
    }
    error
}

// Looks at the errors and updates weights backwards through the network.
fn backward_propagation(network: &mut Network, input: &[Input]) {
    // Find error values.
    let error_o = determine_error_o(&network.outcome.view(), &network.result_o.view());
    let error_h = determine_error_h(
        &network.result_h.view(),
        &network.weight_oh.view(),
        &error_o,
    );

    let rb = RATE / (BATCHES as f32);
    // Update output layer weights
    for k in 0..OUTPUT {
        for j in 0..HIDDEN {
            let mut sum = 0.0;
            // Gather error and result from each input in batch.
            for b in 0..BATCHES {
                sum += error_o[[b, k]] * network.result_h[[b, j]];
            }
            update_weight(
                sum * rb,
                network.weight_ohd[[k, j]],
                &mut network.weight_oh[[k, j]],
                &mut network.weight_ohd[[k, j]],
            );
        }
        // Bias weight
        let mut sum = 0.0;
        for b in 0..BATCHES {
            sum += error_o[[b, k]] * network.weight_oh[[k, HIDDEN]];
        }
        update_weight(
            sum * rb,
            network.weight_ohd[[k, HIDDEN]],
            &mut network.weight_oh[[k, HIDDEN]],
            &mut network.weight_ohd[[k, HIDDEN]],
        );
    }

    // Update hidden layer weights
    for j in 0..HIDDEN {
        for i in 0..INPUT {
            let mut sum = 0.0;
            for b in 0..BATCHES {
                sum += error_h[[b, j]] * input[b].data[i];
            }
            update_weight(
                sum * rb,
                network.weight_hid[[j, i]],
                &mut network.weight_hi[[j, i]],
                &mut network.weight_hid[[j, i]],
            );
        }
        // Bias weight
        let mut sum = 0.0;
        for b in 0..BATCHES {
            sum += error_h[[b, j]] * network.weight_hi[[j, INPUT]];
        }
        update_weight(
            sum * rb,
            network.weight_hid[[j, INPUT]],
            &mut network.weight_hi[[j, INPUT]],
            &mut network.weight_hid[[j, INPUT]],
        );
    }
}
fn update_weight(delta: f32, delta_old: f32, weight: &mut f32, weight_dyn: &mut f32) {
    let delta_new = delta + MOMENTUM * delta_old;
    *weight -= delta_new;
    *weight_dyn = delta_new;
    //println!("{}, ", *weight_dyn);
}
