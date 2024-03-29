use ndarray::prelude::*;
use ndarray_rand::rand_distr::Distribution;
use rand::distributions::Uniform;

#[derive(Clone, Copy)]
pub struct LayerSize {
    pub output: usize,
    pub input: usize,
    pub batch: usize,
}
impl LayerSize {
    pub fn new(o: usize, i:usize, b: usize) -> LayerSize{
        LayerSize { output: o, input: i, batch: b }
    }
}

pub struct Layer {
    pub result:  Array2<f32>,       // [output + 1, batch]
    pub size:    LayerSize,         // Sizes
    
    weight:      Array2<f32>,       // [output, input + 1]
    weight_last: Array2<f32>,       // [output, input + 1]
    error:       Array2<f32>,       // [output, batch]
}
impl Layer {
    // Constructors
    pub fn new(size: LayerSize) -> Layer {
        // The weight between the output nodes and their inputs.
        let weight 
            = Array2::<f32>::zeros((size.output, size.input + 1));
        // For storing the last used weight delta for momentum term.
        let weight_last 
            = Array2::<f32>::zeros((size.output, size.input + 1));
        // Stores the results of forward propogation. Add 1 to row for bias input on next layer.
        let result 
            = Array2::<f32>::zeros((size.output + 1, size.batch));
        // Stores the error value for back propogation.
        let error
            = Array2::<f32>::zeros((size.output, size.batch));
        
        Layer{ 
            size,
            weight, 
            weight_last, 
            result, 
            error
        }
    }
    pub fn _weight_set(&mut self, weight: f32) {
        for w in &mut self.weight {
            *w = weight;
        }
    }
    pub fn weight_randomize(&mut self, low: f32, high: f32) {
        let between = Uniform::from(low..high);
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
        // Set Bias as first term for next layer.
        self.result[[0, batch]] = 1.;
        // Calculate result values.
        for j in 0..self.size.output {
            // Dot product with input values
            let dot_product = self.weight.row(j).dot(input);
            self.result[[j + 1, batch]] = Layer::sigmoid(dot_product);
        }
    }

    // Error functions
    pub fn find_error_output(&mut self, target: &ArrayView1<f32>) {
        for output in 0..self.size.output {     
            for batch in 0..self.size.batch {
                let input = self.result[[output + 1, batch]];
                let factor = target[batch] - input;
                self.error[[output, batch]] = Layer::sigmoid_derivative(input, factor);
            }
        }
    }
    pub fn find_error_layer(&mut self, layer_prev: &Layer) {
        for output in 0..self.size.output {
            for batch in 0..self.size.batch {
                let col_w = &layer_prev.weight.column(output);
                let col_e = &layer_prev.error.column(batch);
                let factor = col_w.dot(col_e);

                let input = self.result[[output + 1, batch]];
                self.error[[output, batch]] = Layer::sigmoid_derivative(input, factor);
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
        let batch_size = self.size.batch as f32;

        for j in 0..self.size.output {
            let row_e = self.error.row(j);
            
            for i in 0..=self.size.input {
                // Use column or row depending on if input is original or layer.
                let row_i =
                    if read_column  { input.column(i) }
                    else            { input.row(i) };

                // Weight update
                let delta = learning_rate * row_i.dot(&row_e) / batch_size 
                    + self.weight_last[[j, i]] * momentum_rate;

                self.weight[[j, i]] += delta;
                self.weight_last[[j, i]] = delta;
            }
        }
    }
}
