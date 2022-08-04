#![allow(dead_code)]
// Constant values for tweaking the algorithm.
pub const DIVIDE: f32 = 255.0;
pub const WEIGHT_LOW: f32 = -0.05;
pub const WEIGHT_HIGH: f32 = 0.05;
pub const LEARNING_RATE: f32 = 0.1;
pub const MOMENTUM_RATE: f32 = 0.9;
pub const EPOCH: usize = 50;

// Array sizes
pub const BATCH_SIZE: usize = 100;
pub const INPUT: usize = 784;
pub const HIDDEN: usize = 20;
pub const OUTPUT: usize = 10;
