// Printing functions for debugging
use ndarray::prelude::*;
/* 
pub fn _print_outcome(outcome: &Array1<Outcome>) {
    // Check our work.
    let mut correct = 0;
    let total = BATCHES as i32;
    
    println!("Target values:");
    for c in outcome {
        print!("{}, ", c.target);
        if c.is_correct() {
            correct += 1;
        }
    }
    println!("\nPredicted values:");
    for c in outcome {
        print!("{}, ", c.prediction);
    }
    _print_total_error(correct, total);
}
*/

pub fn _percentage(num: i32, denom: i32) -> String {
    let per = num as f64 / denom as f64 * 100.0;
    format!("{:.1}%", per)
}

pub fn _print_total_error(correct: i32, total: i32) {
    println!(
        "Correct: {} / {} = {}",
        correct,
        total,
        _percentage(correct, total)
    );
    println!("\n");
}

pub fn _print_matrix(input: &Array2<f32>, name: &str) {
    println!("{} values:", name);
    for row in input.rows() {
        for col in row {
            print!("{}, ", col);
        }
        println!();
    }
    println!("\n");
}

pub fn _print_vector(input: &ArrayView1<f32>) {
    for i in input {
        print!("{}, ", i);
    }
    println!();
}
