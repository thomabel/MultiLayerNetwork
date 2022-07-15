use ndarray::prelude::*;
use rand::prelude::*;
use rand::distributions::*;

use crate::constants::*;

pub struct LayerSize {
    pub outputs: usize,
    pub inputs: usize,
}
impl LayerSize {
    pub fn new(o: usize, i:usize) -> LayerSize{
        LayerSize { outputs: o, inputs: i }
    }
}
pub struct Network {
    pub layers: Array1<Layer>,       // Stores all the layers of the network
}
impl Network {
    pub fn new(sizes: &[LayerSize], batch: usize) -> Network {
        let mut temp = Vec::<Layer>::new();
        for size in sizes {
            let lay = Layer::new(size.outputs, size.inputs, batch);
            temp.push(lay);
        }
        Network {
            layers: Array1::<Layer>::from_vec(temp)
        }
    }
}

pub struct Layer {
    pub outputs: usize,
    pub inputs: usize,
    pub weights: Array2<f32>,
    pub weights_last: Array2<f32>,
    pub results: Array2<f32>,
}
impl Layer {
    fn new(outputs: usize, inputs: usize, batch: usize) -> Layer {
        let weights = Array2::<f32>::zeros((outputs, inputs));
        let weights_last = Array2::<f32>::zeros((outputs, inputs));
        let results = Array2::<f32>::zeros((outputs, batch));
        
        let mut layer = Layer{ outputs, inputs, weights, weights_last, results };
        layer.randomize();
        layer
    }
    fn randomize(&mut self) {
        let between = Uniform::from(LOW..HIGH);
        let mut rng = rand::thread_rng();

        for w in &mut self.weights {
            *w = between.sample(&mut rng);
        }
    }
}
