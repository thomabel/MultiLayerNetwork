#![allow(dead_code)]
// Constant values for tweaking the algorithm.
pub const DIVIDE: f32 = 255.0;
pub const LOW: f32 = -0.05;
pub const HIGH: f32 = 0.05;
pub const RATE: f32 = 0.1;
pub const MOMENTUM: f32 = 0.9;
pub const EPOCH: usize = 10;

// Array sizes
pub const BATCHES: usize = 10;
pub const INPUT: usize = 784;
pub const HIDDEN: usize = 20;
pub const OUTPUT: usize = 10;
