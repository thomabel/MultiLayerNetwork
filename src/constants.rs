#![allow(dead_code)]
// Constant values for tweaking the algorithm.
pub const DIVIDE: f32 = 255.0;
pub const LOW: f32 = -0.05;
pub const HIGH: f32 = 0.05;
pub const RATE: f32 = 0.2;
pub const MOMENTUM: f32 = 0.9;
pub const EPOCH: usize = 100;

// Array sizes
pub const BATCHES: usize = 1;
//pub const INPUT: usize = 784;
pub const INPUT: usize = 2;
pub const HIDDEN: usize = 2;
pub const OUTPUT: usize = 1;

// Current target values
pub const TARGET: [char; 10] = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

pub const START: Option<usize>
    //= Some(0);
    = None;
pub const END: Option<usize>
    = Some(1);
    //= None;
