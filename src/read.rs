use crate::constants::INPUT;
use crate::constants::DIVIDE;
use crate::data::Input;
use std::error::Error;

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
            let num = record[i].parse::<u8>();
            match num {
                Ok(c) => {
                    entry.data[i] = c as f32 / DIVIDE;
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

pub fn read_toy (path: &str) -> Result<Vec<Input>, Box<dyn Error>> {
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
            let num = record[i].parse::<u8>();
            match num {
                Ok(c) => {
                    entry.data[i] = c as f32 / DIVIDE;
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