use ndarray::prelude::*;
use crate::layer::*;

pub struct Network {
    pub batch: usize,
    pub batch_total: usize,
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
            batch_total,
            learning_rate,
            momentum_rate,
            layers,
        }
    }
    pub fn set_weights(&mut self, weight: f32) {
        for layer in &mut self.layers {
            layer.set_weights(weight);
        }
    }

    // The main function for training the network.
    pub fn gradient_descent(&mut self, input: &(Array2<f32>, Array1<f32>)) {
        let mut batch = 0;
        let mut total = 0;
        for i in input.0.rows() {
            println!("=== BATCH: {}, INPUT: {}, TOTAL: {} ===", batch + 1, self.batch + 1, total + 1);
            self.forward(&i);
            
            self.batch += 1;
            total += 1;

            // Do back propogation
            if self.batch == self.batch_total {
                let start = total - self.batch_total;
                let end = total;
                self.find_error(&input.1.slice(s![start..end]));
                self.update_weights(&input.0.slice(s![start..end, ..]));
                self.batch = 0;
                batch += 1;
            }
            println!();
        }
    }

    // Passes input into the chain of hidden layers until the final layer us used.
    pub fn forward(&mut self, target: &ArrayView1<f32>) {
        let mut iter = self.layers.iter_mut();
        let mut layer_prev = iter.next().unwrap();
        let mut input;

        // Feed forward the initial input layer.
        println!("Forward:\nInitial:");
        layer_prev.feed_forward(target, self.batch);

        // Feed it through every other layer.
        for layer in iter {
            println!("Layer:");
            input = layer_prev.result.index_axis(Axis(1), self.batch);
            layer.feed_forward(&input, self.batch);
            layer_prev = layer;
        }
        println!();
    }

    // Finds the error values given each output.
    pub fn find_error(&mut self, target: &ArrayView1<f32>) {
        let mut iter = self.layers.iter_mut().rev();
        let mut layer_prev = iter.next().unwrap();

        // Find error values for output layer.
        println!("Error:\nOutput:");
        layer_prev.find_error_output(target);
        println!();

        // Find error values for each inner layer.
        for layer in iter {
            println!("Layer:");
            layer.find_error_layer(layer_prev);
            layer_prev = layer;
            println!();
        }
        println!();
    }
    
    // Updates the weights given the input batch and current error values.
    pub fn update_weights(&mut self, input: &ArrayView2<f32>) {
        let mut iter = self.layers.iter_mut();
        let mut layer_prev = iter.next().unwrap();

        // Update the first layer using the input values.
        println!("Update:\nInitial:");
        layer_prev.update_weights(
            input, true,
            self.learning_rate, self.momentum_rate);

        // Update the weights of every other layer.
        for layer in iter {
            println!("Layer:");
            layer.update_weights(
                &layer_prev.result.view(), false, 
                self.learning_rate, self.momentum_rate);
            layer_prev = layer;
            println!();
        }
        println!();
    }
    
}
