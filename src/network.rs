use ndarray::prelude::*;
use crate::layer::*;

pub struct Network {
    pub batch: usize,
    pub learning_rate: f32,
    pub momentum_rate: f32,
    pub layers: Array1<Layer>,       // Stores all the layers of the network
}
impl Network {
    pub fn new(sizes: &[LayerSize], learning_rate: f32, momentum_rate: f32) -> Network {
        let mut temp = Vec::<Layer>::new();
        for size in sizes {
            let layer = Layer::new(size);
            temp.push(layer);
        }
        Network {
            batch: 0,
            learning_rate,
            momentum_rate,
            layers: Array1::<Layer>::from_vec(temp)
        }
    }
    pub fn set_weights(&mut self, weight: f32) {
        for layer in &mut self.layers {
            layer.set_weights(weight);
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
            println!("Layer: ");
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
        //let mut input;

        // Find all error values.
        println!("Error:\nOutput:");
        layer_prev.find_error_output(target);
        println!();
        for layer in iter {
            print!("Layer: ");
            layer.find_error_layer(layer_prev);
            println!();
            layer_prev = layer;
        }
        println!();
    }
    
    // Updates the weights given the input batch and current error values.
    pub fn update_weights(&mut self, input: &ArrayView2<f32>) {
        let mut iter = self.layers.iter_mut();
        let mut layer_prev = iter.next().unwrap();
        println!("Update:\nInitial:");
        layer_prev.update_weights(
            input, true,
            self.learning_rate, self.momentum_rate);

        // Feed it through every other layer.
        for layer in iter {
            println!("Layer:");
            layer.update_weights(
                &layer_prev.result.view(), false, 
                self.learning_rate, self.momentum_rate);
            layer_prev = layer;
        }
        println!();
    }
    
}
