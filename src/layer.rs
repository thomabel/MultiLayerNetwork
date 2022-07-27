use ndarray::prelude::*;
use ndarray_rand::rand_distr::Distribution;
use rand::distributions::Uniform;
use crate::constants::*;

pub struct LayerSize {
    pub outputs: usize,
    pub inputs: usize,
    pub batches: usize,
}
impl LayerSize {
    pub fn new(o: usize, i:usize, b: usize) -> LayerSize{
        LayerSize { outputs: o, inputs: i, batches: b }
    }
}

pub struct Layer {
    pub outputs:     usize,
    pub inputs:      usize,
    pub batches:     usize,
    pub weight:      Array2<f32>,
    pub weight_last: Array2<f32>,
    pub result:      Array2<f32>,
    pub error:       Array2<f32>,
}
impl Layer {
    // Constructors
    pub fn new(size: &LayerSize) -> Layer {
        let weight 
            = Array2::<f32>::zeros((size.outputs, size.inputs));
        let weight_last 
            = Array2::<f32>::zeros((size.outputs, size.inputs));
        let result 
            = Array2::<f32>::zeros((size.outputs, size.batches));
        let error
            = Array2::<f32>::zeros((size.outputs, size.batches));
        
        Layer{ 
            outputs: size.outputs, 
            inputs: size.inputs, 
            batches: size.batches,
            weight, 
            weight_last, 
            result, 
            error
        }
    }
    pub fn set_weights(&mut self, weight: f32) {
        for w in &mut self.weight {
            *w = weight;
        }
    }
    pub fn randomize_weights(&mut self) {
        let between = Uniform::from(LOW..HIGH);
        let mut rng = rand::thread_rng();

        for w in &mut self.weight {
            *w = between.sample(&mut rng);
        }
    }

    // Sigmoid function + derivative
    fn sigmoid(input: f32) -> f32 {
        1. / (1. + f32::exp(-input))
    }
    fn sigmoid_derivative(input: f32, factor: f32) -> f32 {
        input * (1.0 - input) * factor
    }

    // Finds all results for this layer given an input vector.
    pub fn feed_forward(&mut self, input: &ArrayView1<f32>, batch: usize) {
        let len = self.inputs - 1;
        // For each output node
        for j in 0..self.outputs {
            // Dot product with input values
            let mut solution: f32 = 0.0;
            for i in 0..len {
                let wi = self.weight[[j, i]] * input[i];
                solution += wi;
            }
            // Bias
            solution += self.weight[[j, len]];
            // use sigmoid
            solution = Layer::sigmoid(solution);
            self.result[[j, batch]] = solution;
            print!("{:.3}, ", solution);
        }
        println!();
    }

    // Error functions
    pub fn find_error_output(&mut self, target: &ArrayView1<f32>) {
        for node in 0..self.outputs {     
            for batch in 0..self.batches {
                let input = self.result[[node, batch]];
                let factor = target[batch] - input;
                let error = Layer::sigmoid_derivative(input, factor);
                self.error[[node, batch]] = error;
                print!("{:.3}, ", error);
            }
        }
    }
    pub fn find_error_layer(&mut self, layer_prev: &Layer) {
        for hidden in 0..self.outputs {
            for batch in 0..self.batches {
                let input = self.result[[hidden, batch]];
                let col_w = &layer_prev.weight.column(hidden);
                let col_e = &layer_prev.error.column(batch);
                let factor = col_w.dot(col_e);
                let error = Layer::sigmoid_derivative(input, factor);
                self.error[[hidden, batch]] = error;
                print!("{:.3}, ", error);
            }
        }
    }

    // Update the weights given the current error values and some input.
    pub fn update_weights(
        &mut self, 
        input: &ArrayView2<f32>, 
        read_column: bool, 
        learning_rate: f32, 
        momentum_rate: f32
    ) {
        let batch_size = self.batches as f32;
        let index = self.inputs - 1;

        for j in 0..self.outputs {
            let row_e = self.error.row(j);
            
            for i in 0..index {
                let row_i;
                if read_column {
                    row_i = input.column(i);
                }
                else {
                    row_i = input.row(i);
                }
                // Weight update
                let weight = &mut self.weight[[j, i]];
                let error = row_i.dot(&row_e) / batch_size;
                let delta = learning_rate * error;
                let momentum = self.weight_last[[j, i]] * momentum_rate;
                *weight += delta + momentum;
                print!("{:.4} * {:.4} + {:.4} * {:.4} = ", learning_rate, error, momentum_rate, self.weight_last[[j, i]]);
                println!("{:.4}, ", *weight);
            }

            // Bias weight update
            let weight = &mut self.weight[[j, index]];
            let delta = learning_rate * row_e.sum() / batch_size;
            let momentum = self.weight_last[[j, index]];
            *weight += delta + momentum;
            print!("{:.4}, ", *weight);

            println!();
        }
        println!();
    }
}
