use crate::data::Input;
use std::error::Error;
use ndarray::prelude::*;

// Reads .csv file input.
pub fn read_image(path: &str, inputs: usize, dividend: f32) -> Result<Vec<Input>, Box<dyn Error>> {
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
        for i in 1..inputs {
            let num = record[i].parse::<u8>();
            match num {
                Ok(c) => {
                    entry.data[i] = c as f32 / dividend;
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }
        output.push(entry);
    }
    Ok(output)
}

pub fn read_toy (path: &str, inputs: usize) -> Result<Array2<f32>, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut output = Vec::new();
    let mut nrow = 0;

    for result in reader.records() {
        let record = result?;
        // Parse each entry in the input.
        for i in 1..inputs {
            let num = record[i].parse::<f32>();
            match num {
                Ok(c) => {
                    output.push(c);
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
            }
        }
        nrow += 1;
    }

    let arr2 = Array2::from_shape_vec((nrow, inputs), output)?;
    Ok(arr2)
}