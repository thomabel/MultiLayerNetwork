use ndarray::prelude::*;
use crate::{layer::*, print_data::*};

pub struct Network {
    pub batch: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub momentum_rate: f32,
    // Stores all the layers of the network
    pub layers: Array1<Layer>,
}
impl Network {
    pub fn new(sizes: &[LayerSize], learning_rate: f32, momentum_rate: f32, batch_total: usize) -> Network {
        let mut temp = Vec::<Layer>::new();
        for size in sizes {
            temp.push(Layer::new(*size));
        }
        let layers = Array1::<Layer>::from_vec(temp);

        Network {
            batch: 0,
            batch_size: batch_total,
            learning_rate,
            momentum_rate,
            layers,
        }
    }
    pub fn _weight_set(&mut self, weight: f32) {
        for layer in &mut self.layers {
            layer._weight_set(weight);
        }
    }
    pub fn weight_randomize(&mut self, low: f32, high: f32) {
        for layer in &mut self.layers {
            layer.weight_randomize(low, high);
        }
    }

    // The main function for training the network.
    pub fn gradient_descent(&mut self, input: &(Array2<f32>, Array1<f32>)) {
        let mut batch = 0;
        let mut total = 0;
        let mut correct: u32 = 0;
        for i in input.0.rows() {
            println!("=== BATCH: {}, INPUT: {}, TOTAL: {} ===", batch + 1, self.batch + 1, total + 1);

            // Forward Propogation
            self.forward(&i);
            /* 
            for layer in &self.layers {
                _print_matrix(&layer.result, "RESULT");
            }
            */
            self.batch += 1;
            total += 1;

            // Backward propogation
            if self.batch == self.batch_size {
                // For slicing the array over the bach.
                let start = total - self.batch_size;
                let end = total;

                // Slice the arrays and classify targets
                let target = &input.1.slice(s![start..end]);
                let result = &self.layers.last().unwrap().result;
                let target_class = Network::classify_targets(target, result);
                let slice = &input.0.slice(s![start..end, ..]);

                _print_vector(target, "TARGET");
                _print_vector(&target_class.1.view(), "PREDICT");

                for t in 0..target.len() {
                    if target[t] == target_class.1[t] {
                        correct += 1;
                    }
                }
                _print_total_error(correct, total as u32);

                // Get the error and update the weights.
                self.find_error(&target_class.0.view());
                self.update_weights(slice);
                
                // Update counter variables.
                self.batch = 0;
                batch += 1;
            }
            println!();
        }
    }

    // Passes input into the chain of hidden layers until the final layer us used.
    fn forward(&mut self, target: &ArrayView1<f32>) {
        let mut iter = self.layers.iter_mut();
        let mut layer_prev = iter.next().unwrap();
        let mut input;

        // Feed forward the initial input layer.
        layer_prev.feed_forward(target, self.batch);

        // Feed it through every other layer.
        for layer in iter {
            input = layer_prev.result.index_axis(Axis(1), self.batch);
            layer.feed_forward(&input, self.batch);
            layer_prev = layer;
        }
    }

    // Classify targets as either 0.9 or 0.1 for backprop.
    fn classify_targets(target: &ArrayView1<f32>, result: &Array2<f32>) -> (Array1<f32>, Array1<f32>) {
        let mut target_weight = Array1::<f32>::zeros(target.len());
        let mut prediction = Array1::<f32>::zeros(target.len());

        // Cycle through all target values in batch.
        for t in 0..target.len() {
            // Look through each result given that batch and find the largest value.
            let res = result.column(t);
            let mut index: usize = 1;
            for o in 2..res.len() {
                if res[o] > res[index] {
                    index = o;
                }
            }
            index -= 1;
            target_weight[t] = if target[t] == index as f32 { 0.9 } else { 0.1 };
            prediction[t] = index as f32;
        }
        (target_weight, prediction)
    }

    // Finds the error values given each output.
    fn find_error(&mut self, target: &ArrayView1<f32>) {
        let mut iter = self.layers.iter_mut().rev();
        let mut layer_prev = iter.next().unwrap();

        // Find error values for output layer.
        layer_prev.find_error_output(target);

        // Find error values for each inner layer.
        for layer in iter {
            layer.find_error_layer(layer_prev);
            layer_prev = layer;
        }
    }
    
    // Updates the weights given the input batch and current error values.
    fn update_weights(&mut self, input: &ArrayView2<f32>) {
        let mut iter = self.layers.iter_mut();
        let mut layer_prev = iter.next().unwrap();

        // Update the first layer using the input values.
        layer_prev.update_weights(
            input, true,
            self.learning_rate, self.momentum_rate);

        // Update the weights of every other layer.
        for layer in iter {
            layer.update_weights(
                &layer_prev.result.view(), false, 
                self.learning_rate, self.momentum_rate);
            layer_prev = layer;
        }
    }
}
