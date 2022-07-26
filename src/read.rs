use std::error::Error;
use ndarray::prelude::*;

pub fn read_toy (path: &str, inputs: usize) -> Result<(Array2<f32>, Array1<f32>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut output = Vec::new();
    let mut target = Vec::new();
    let mut nrow = 0;

    for result in reader.records() {
        let record = result?;
        // Parse first entry as the target value.
        let num = record[0].parse::<f32>();
        match num {
            Ok(c) => {
                target.push(c);
            }
            Err(e) => {
                return Err(Box::new(e));
            }
        }
        // Parse each entry in the input.
        for i in 1..=inputs {
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

    let out_array 
        = Array2::from_shape_vec((nrow, inputs), output)?;
    let target_array
        = Array1::from_vec(target);
    Ok((out_array, target_array))
}