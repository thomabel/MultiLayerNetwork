// 
use ndarray::prelude::*;
use crate::constants::*;
use ndarray_rand::rand_distr::*;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub struct Input {
    pub t: char,
    pub data: Array1<f32>,
}
impl Default for Input {
    fn default() -> Input {
        Input {
            t: '0',
            data: Array::zeros(INPUT)
        }
    }
}

// Stores the prediction of the network along with the target value.
#[derive(Clone, Copy)]
pub struct Outcome {
    pub target: char,
    pub prediction: char,
}
pub trait Correctness {
    fn is_correct(&self) -> bool;
}
impl Correctness for Outcome {
    fn is_correct(&self) -> bool {
        self.target == self.prediction
    }
}
/* Data matrices needed:
- weights between nodes, 2D
    - hidden <- input 1
    - output <- hidden 2
- results from forward propagation, batched, 2D
    - hidden 3
    - output 4
- predicted outcomes from each output, 1D
    - inputs as part of batch 5
- weight change from last batch change, for momentum, 2D
    - hidden <- input 6
    - output <- hidden 7
*/
pub struct Network_ {
    pub weight_hi: Array2<f32>,
    pub weight_oh: Array2<f32>,
    pub result_h: Array2<f32>,
    pub result_o: Array2<f32>,
    pub outcome: Array1<Outcome>,
    pub weight_hid: Array2<f32>,
    pub weight_ohd: Array2<f32>,
}
impl Default for Network_ {
    fn default() -> Self {
        // Shapes of the arrays.
        // Add 1 to columns for bias weights.
        let shape_hi = (HIDDEN, INPUT + 1);
        let shape_oh = (OUTPUT, HIDDEN + 1);
        let shape_rh = (BATCHES, HIDDEN);
        let shape_ro = (BATCHES, OUTPUT);
        let distribution = Uniform::new_inclusive(LOW, HIGH);
        //let distribution = Normal::<f32>::new(0., 1.).unwrap();

        Network_ {
            // Assign weights randomly with uniform distribution.
            weight_hi: Array::random(shape_hi, distribution),
            weight_oh: Array::random(shape_oh, distribution),

            // Weight deltas used for momentum.
            weight_hid: Array2::zeros(shape_hi),
            weight_ohd: Array2::zeros(shape_oh),

            // Stores BATCHES results from that number of inputs forward propagating.
            result_h: Array2::zeros(shape_rh),
            result_o: Array2::zeros(shape_ro),

            // Stores outcomes of the predictions with their targets as (target, outcome)
            outcome: Array::<Outcome, _>::from_elem(
                BATCHES,
                Outcome {
                    target: '0',
                    prediction: '0',
                })
        }
    }
}
