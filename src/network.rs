use ndarray::prelude::*;
use crate::{layer::*, print_data::*};

pub struct Network {
    layers: Array1<Layer>,
    batch_size: usize,
}
impl Network {
    pub fn new(sizes: &[LayerSize], batch_total: usize) -> Network {
        let mut temp = Vec::<Layer>::new();
        for size in sizes {
            temp.push(Layer::new(*size));
        }
        let layers = Array1::<Layer>::from_vec(temp);

        Network {
            batch_size: batch_total,
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
    pub fn gradient_descent(
        &mut self, 
        input: &(Array2<f32>, Array1<f32>), 
        learning_rate: f32, 
        momentum_rate: f32
    ) -> Array2<u32>
    {
        let batch_total = input.0.len_of(Axis(0)) / self.batch_size;
        let confusion_size = self.layers.last().unwrap().size.output;
        let mut confusion = Array2::<u32>::zeros((confusion_size, confusion_size));
        let mut total = 0;
        let mut correct: u32 = 0;

        // Loops for each batch.
        for i in 0..batch_total {
            println!("BATCH: {}, TOTAL: {}", i + 1, total);

            // Forward Propogation
            // Go through the next n inputs as a batch.
            for j in 0..self.batch_size {
                let index = i + j;
                self.forward( 
                    &input.0.row(index).view(), 
                    j);
                total += 1;
            }

            // Backward propogation
            // For slicing the array over the batch.
            let start = total - self.batch_size;
            let end = total;

            // Slice the arrays and classify targets
            let target = &input.1.slice(s![start..end]);
            let result = &self.layers.last().unwrap().result;
            let target_class = Network::classify_targets(target, result);

            // Print all of the target and predicted values along with correct percentage
            for t in 0..target.len() {
                let j = target[t] as usize;
                let i = target_class.1[t] as usize;
                confusion[[j, i]] += 1;
                if j == i {
                    correct += 1;
                }
            }
            _print_vector(target, "TARGET");
            _print_vector(&target_class.1.view(), "PREDICT");
            _print_total_error(correct, total as u32);

            // Get the error and update the weights.
            self.find_error(
                &target_class.0.view()
            );
            self.update_weights(
                &input.0.slice(s![start..end, ..]), 
                learning_rate, 
                momentum_rate,
            );
        }
        confusion
    }

    // Passes input into the chain of hidden layers until the final layer us used.
    fn forward(&mut self, target: &ArrayView1<f32>, batch: usize){
        let mut iter = self.layers.iter_mut();
        let mut layer_prev = iter.next().unwrap();
        let mut input;

        // Feed forward the initial input layer.
        layer_prev.feed_forward(target, batch);

        // Feed it through every other layer.
        for layer in iter {
            input = layer_prev.result.index_axis(Axis(1), batch);
            layer.feed_forward(&input, batch);
            layer_prev = layer;
        }
    }

    // Classify targets as either 0.9 or 0.1 for backprop.
    fn classify_targets(target: &ArrayView1<f32>, result: &Array2<f32>) 
        -> (Array1<f32>, Array1<f32>){
        let len = target.len();
        let mut target_weight = Array1::<f32>::zeros(len);
        let mut prediction = Array1::<f32>::zeros(len);

        // Cycle through all target values in batch.
        for t in 0..len {
            // Look through each result given that batch and find the largest value.
            let res_col = result.column(t);
            let r_len = res_col.len();
            let mut index: usize = 1;
            for o in 1..r_len {
                //print!("{:.3}, ", res_col[o]);
                if res_col[o] > res_col[index] {
                    index = o;
                }
            }
            //println!();
            index -= 1;
            let pre = index as f32;
            target_weight[t] = if target[t] == pre { 0.9 } else { 0.1 };
            prediction[t] = pre;
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
    fn update_weights(
        &mut self,
        input: &ArrayView2<f32>,
        learning_rate: f32,
        momentum_rate: f32,
    ){
        let mut iter = self.layers.iter_mut();
        let mut layer_prev = iter.next().unwrap();

        // Update the first layer using the input values.
        layer_prev.update_weights(
            input, true,
            learning_rate, momentum_rate);

        // Update the weights of every other layer.
        for layer in iter {
            layer.update_weights(
                &layer_prev.result.view(), false, 
                learning_rate, momentum_rate);
            layer_prev = layer;
        }
    }

}
