use crate::constants::DIVIDE;
use crate::constants::INPUT;
use ndarray::prelude::*;
use std::error::Error;

#[derive(Debug)]
pub struct Input {
    pub t: char,
    pub data: Array1<f32>,
}
impl Default for Input {
    fn default() -> Input {
        Input {
            t: '0',
            data: Array::zeros(INPUT.f()),
        }
    }
}
// Reads .csv file input.
pub fn read(path: &str) -> Result<Vec<Input>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut output = Vec::new();

    for result in reader.records() {
        let record = result?;

        // Create test entry and find true type t
        let mut entry = Input {
            t: record[0].parse().unwrap(),
            ..Default::default()
        };

        // Gets the pixel intensity value
        for i in 1..INPUT {
            let num = record[i].parse::<i16>();
            match num {
                Ok(c) => {
                    entry.data[i] = c as f32 / DIVIDE;
                }
                Err(e) => println!("ERROR: {}", e),
            }
        }
        output.push(entry);
    }
    Ok(output)
}
