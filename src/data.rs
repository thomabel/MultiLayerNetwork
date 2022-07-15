// 
use ndarray::prelude::*;
use crate::constants::*;
use ndarray_rand::rand_distr::*;
use ndarray_rand::RandomExt;

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
pub struct Network {
    pub weight_hi: Array2<f32>,
    pub weight_oh: Array2<f32>,
    pub result_h: Array2<f32>,
    pub result_o: Array2<f32>,
    pub outcome: Array1<Outcome>,
    pub weight_hid: Array2<f32>,
    pub weight_ohd: Array2<f32>,
}
impl Default for Network {
    fn default() -> Self {
        let shape_hi = (HIDDEN, INPUT + 1);
        let shape_oh = (OUTPUT, HIDDEN + 1);
        //let distribution = Uniform::new_inclusive(LOW, HIGH);
        let distribution = Normal::<f32>::new(0., 1.).unwrap();
        Network {
            // Assign weights randomly with uniform distribution.
            weight_hi: Array::random(shape_hi, distribution),
            weight_oh: Array::random(shape_oh, distribution),
            // Stores BATCHES results from that number of inputs forward propagating.
            result_h: Array2::zeros((BATCHES, HIDDEN)),
            result_o: Array2::zeros((BATCHES, OUTPUT)),
            // Stores outcomes of the predictions with their targets as (target, outcome)
            outcome: Array::<Outcome, _>::from_elem(
                BATCHES,
                Outcome {
                    target: '0',
                    prediction: '0',
                },
            ),
            weight_hid: Array2::zeros(shape_hi),
            weight_ohd: Array2::zeros(shape_oh),
        }
    }
}
