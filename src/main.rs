use std::collections::VecDeque;
use std::env;
use std::error::Error;
use argh::FromArgs;
use tokio::io::AsyncReadExt;
use tokio::net::TcpSocket;
use tracing::{error, warn};

use std::sync::{Arc};
use std::time::Duration;
use bushtelegram_dsp::{FFT_OUTPUT, SAMPLE_SIZE, SAMPLING_FREQUENCY_HZ, SPLIT_AT_LEFT};
use bushtelegram_dsp::fft::apply_fft;
use bushtelegram_dsp::filters::{apply_butterworth_bandpass_filter, energy_in_target_band};
use log::info;
use textplots::{Chart, ColorPlot, Plot, Shape};
use tokio::sync::RwLock;
use tracing_subscriber::fmt::format;

const GREEN: rgb::RGB8 = rgb::RGB8::new(0x00, 0xFF, 0x00);
const RED: rgb::RGB8 =  rgb::RGB8::new(0xFF, 0x00, 0x00);
const BLUE: rgb::RGB8 =  rgb::RGB8::new(0x00, 0x00, 0xFF);

#[derive(FromArgs)]
/// Connects to a TCP socket and plots 16 bit values
pub struct Args {
    /// optional verbosity level. Default 0
    #[argh(option, short = 'v', default = "0")]
    verbose: u64,

    /// the server to connect to
    #[argh(option, short = 's', default = "String::from(\"192.168.88.248:1234\")")]
    server: String,
}

pub const DEFAULT_FILTER_ENV: &str = "LOG_LEVEL";


#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dotenv_vars = dotenv::dotenv();
    let args: Args = argh::from_env();
    unsafe { setup_log(args.verbose) };
    if let Err(e) = dotenv_vars {
        warn!(?e, "Error getting values from .env file, ignoring");
    }

    // hide the cursor so we don't see it flying all over
    let term = console::Term::stdout();
    term.hide_cursor().unwrap();
    term.clear_screen().unwrap();

    let addr = args.server.parse()?;
    let socket = TcpSocket::new_v4()?;
    let mut stream = socket.connect(addr).await?;
    info!("Connected to {}", addr);

    // let vals = VecDeque::<f32>::new();
    let vals = [0f32; SAMPLE_SIZE].to_vec();
    let vals = Arc::new(RwLock::new(vals));

    // Read from the TCP socket and queue the values into the VecDequeue
    let vals_for_writing = vals.clone();
    let footstep_detected = Arc::new(RwLock::new(false));
    let footstep_detected_for_filter = footstep_detected.clone();
    let energy = Arc::new(RwLock::new(0f32));
    let energy_clone = energy.clone();
    tokio::spawn(async move {
        loop {
            let mut new_val = [0f32; SAMPLE_SIZE];
            for i in 0..SAMPLE_SIZE {

                new_val[i] = match stream.read_f32_le().await {
                    Ok(val) => val,
                    Err(e) => {
                        // error!(?e, "Error reading from stream");
                        continue;
                        // break;
                    }
                };
            }

            // let mut new_val = match apply_butterworth_bandpass_filter(&mut new_val, SAMPLING_FREQUENCY_HZ as f32, 20.0) {
            //     Ok(v) => v,
            //     Err(e) => {
            //         error!(?e, "Error with filter");
            //         &mut new_val
            //     }
            // };
            let mut new_val = &mut new_val;
            let mut new_val = apply_fft(new_val);
            // let mut new_val  = new_val.as_slice().first_chunk().expect("Could not get half of the FFT array");
            let mut new_val = match apply_butterworth_bandpass_filter(&mut new_val, SAMPLING_FREQUENCY_HZ as f32, 100.0) {
                Ok(nv) => nv,
                Err(e) => new_val
            };
            // info!("Read {SAMPLE_SIZE} f32");
            // let new_val = apply_fft(&mut new_val);
            // let new_val = new_val.as_slice()[0..FFT_OUTPUT].to_vec();
            // let mut new_val = new_val.as_slice()[0..FFT_OUTPUT-48].to_vec();
            // let x_offset = 22.0;
            // let k1 = 0.0086;
            // let k2 = 0.04;
            // let offset = 0.0;
            //
            // // new_val.iter_mut().enumerate().for_each(|(i, y)| {
            // //     let x = i as f32;
            // //     let x = x_offset - (x.powi(2)*k1 + x * k2 + 0.00001);
            // //     // *y = *y - x;
            // //     *y = *y - x + offset;
            // //     // *y = *y / x + offset;
            // //     // *y = x;
            // // });

            {
                let mut w = vals_for_writing.write().await;
                let (_l, r) = new_val.split_at(SPLIT_AT_LEFT);
                *w = r.to_vec()
            }
            {
                let detected = footstep_detected_for_filter.read().await;
                if *detected {
                    continue;
                }
            }
            // {
            //     let mut w = footstep_detected_for_filter.write().await;
            //     let mut ww = energy_clone.write().await;
            //     let energy_in_band = energy_in_target_band(new_val, SAMPLING_FREQUENCY_HZ as f32, 4.0, 20.0);
            //     match energy_in_band {
            //         Ok(energy) => {
            //             *w = energy > 50.0;
            //             *ww = energy
            //         },
            //         Err(e) => error!(?e, "Error with footstep detection")
            //     }
            // }
            //
            // if let Err(e) = val {
            //     error!(?e, "Error reading from streaam");
            //     break;
            //
            // }
            // let val = val.unwrap();
            // {
            //     let mut w = vals_for_writing.write().await;
            //     w.push_back(val);
            //     if w.len() > FFT_OUTPUT {
            //         w.pop_front();
            //     }
            // }
        }
    });

    // The actual plotter
    tokio::spawn(async move {
        let mut max = 0_f32;
        let mut min = 0_f32;
        let mut current_max = 0_f32;
        let mut current_min = 0_f32;
        let mut max_energy = 0_f32;

        let mut average = 0.0;
        loop {
            current_max = 0.0;
            current_min = 0.0;
            average = 0.0;

            let buf = {
                let r = vals.read().await;
                r.clone()
            };
            // Map the 16 bit values to F32
            let buf = buf.iter()
                .enumerate()
                .map(|(k, v)| {
                    if current_max < *v {
                        current_max = *v;
                    }
                    if current_min > *v {
                        current_min = *v;
                    }
                    average += *v;
                (k as f32, *v as f32)
            } )
                .collect::<Vec<_>>();

            if current_max > max {
                max = current_max;
            }
            if current_min < min {
                min = current_min;
            }

            average /= buf.len() as f32;



            term.move_cursor_to(0, 1).unwrap();
            term.clear_screen().unwrap();

            let detected = {
                let r = footstep_detected.read().await;
                if *r { "detected" } else { " "}
            };
            let energy = {
                let r = energy.read().await;
                if *r > max_energy {
                    max_energy = *r;
                }
                format!("{:.2}", *r)
            };
            let max_energy = format!("{:.2}", max_energy);

            println!("⬆️{current_max}({max}), ⬇️:{current_min}({min}) ↔️:{average} 🦶🏻:{detected} 🔋:{energy} ({max_energy})");
            Chart::new_with_y_range(200, 100, 0., buf.len() as f32, min as f32, max as f32)
                .linecolorplot(&Shape::Lines(buf.as_slice()), GREEN)
                .linecolorplot(&Shape::Lines(&[(0.0, average), (buf.len() as f32, average)]), RED)
                .display();
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }).await?;

    Ok(())
}

pub unsafe fn setup_log(verbose: u64) {
    use tracing_subscriber::prelude::*;
    use tracing_subscriber::{fmt, EnvFilter};

    if env::var("RUST_LIB_BACKTRACE").is_err() {
        env::set_var("RUST_LIB_BACKTRACE", "1")
    }
    let default_filter = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    }
        .to_string();

    let default_filter = {
        match env::var("LOG_LEVEL") {
            Ok(v) => v,
            Err(_) => {
                env::set_var("LOG_LEVEL", default_filter.clone());
                default_filter
            }
        }
    };

    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", default_filter)
    }

    let fmt_layer = fmt::layer().with_target(false);
    let filter_layer = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(fmt_layer)
        // .with(ErrorLayer::default())
        .init();

    color_eyre::install().unwrap();
}