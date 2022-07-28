// Printing functions for debugging
use ndarray::prelude::*;

pub fn _percentage(num: u32, denom: u32) -> String {
    let percent = num as f64 / denom as f64 * 100.0;
    format!("{:.1}%", percent)
}

pub fn _print_total_error(correct: u32, total: u32) {
    println!("Correct: {} / {} = {}\n\n", correct, total, _percentage(correct, total));
}

pub fn _print_matrix<T: std::fmt::Debug>(input: &ArrayView2<T>, name: &str) {
    println!("{}:", name);
    for row in input.rows() {
        for col in row {
            print!("{:02?}, ", col);
        }
        println!();
    }
    println!();
}

pub fn _print_vector<T: std::fmt::Debug>(input: &ArrayView1<T>, name: &str) {
    println!("{}:", name);
    for i in input {
        print!("{:.0?}, ", i);
    }
    println!();
}
