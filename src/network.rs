use ndarray::prelude::*;
use rand::prelude::*;
use rand::distributions::*;
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
pub struct Network {
    pub batch: usize,
    pub layers: Array1<Layer>,       // Stores all the layers of the network
}
impl Network {
    pub fn new(sizes: &[LayerSize]) -> Network {
        let mut temp = Vec::<Layer>::new();
        for size in sizes {
            let layer = Layer::new(size);
            temp.push(layer);
        }
        Network {
            batch: 0,
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
        println!("Forward: ");
        print!("Initial: ");
        layer_prev.feed_forward(target, self.batch);

        // Feed it through every other layer.
        for layer in iter {
            print!("Layer: ");
            input = layer_prev.result.index_axis(Axis(1), self.batch);
            layer.feed_forward(&input, self.batch);
            layer_prev = layer;
        }
        println!();
    }

    pub fn backward(&mut self, target: &Array1<f32>) {
        let mut iter = self.layers.iter_mut().rev();
        let mut layer_prev = iter.next().unwrap();
        //let mut input;

        // Find all error values.
        println!("Backward: ");
        print!("Output: ");
        layer_prev.find_error_output(target);
        println!();
        for layer in iter {
            print!("Layer: ");
            layer.find_error_layer(layer_prev);
            println!();
            layer_prev = layer;
        }

        // Reset iterator
        iter = self.layers.iter_mut().rev();
        layer_prev = iter.next().unwrap();

        // Feed backward through the output layer.
        // Feed it through every other layer.
        for layer in iter {
        }
        println!();
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
    fn new(size: &LayerSize) -> Layer {
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
    fn set_weights(&mut self, weight: f32) {
        for w in &mut self.weight {
            *w = weight;
        }
    }
    fn randomize_weights(&mut self) {
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
    fn feed_forward(&mut self, input: &ArrayView1<f32>, batch: usize) {
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
    fn find_error_output(&mut self, target: &Array1<f32>) {
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
    fn find_error_layer(&mut self, layer_prev: &Layer) {
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

    // 
    fn feed_backward() {

    }
}
