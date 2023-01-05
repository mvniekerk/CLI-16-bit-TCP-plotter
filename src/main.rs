use std::collections::VecDeque;
use std::env;
use std::error::Error;
use argh::FromArgs;
use tokio::io::AsyncReadExt;
use tokio::net::TcpSocket;
use tracing::{error, warn};

use std::sync::{Arc};
use std::time::Duration;
use log::info;
use textplots::{Chart, ColorPlot, Shape};
use tokio::sync::RwLock;

const GREEN: rgb::RGB8 = rgb::RGB8::new(0x00, 0xFF, 0x00);

#[derive(FromArgs)]
/// Connects to a TCP socket and plots 16 bit values
pub struct Args {
    /// optional verbosity level. Default 0
    #[argh(option, short = 'v', default = "0")]
    verbose: u64,
}

pub const DEFAULT_FILTER_ENV: &str = "LOG_LEVEL";

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dotenv_vars = dotenv::dotenv();
    let args: Args = argh::from_env();
    setup_log(args.verbose);
    if let Err(e) = dotenv_vars {
        warn!(?e, "Error getting values from .env file, ignoring");
    }

    // hide the cursor so we don't see it flying all over
    let term = console::Term::stdout();
    term.hide_cursor().unwrap();
    term.clear_screen().unwrap();

    let addr = env::var("SERVER").unwrap_or("192.168.0.105:1234".to_string());
    info!("Going to connect to {addr:}");
    let addr = addr.parse().unwrap();
    let socket = TcpSocket::new_v4()?;
    info!("Connecting");
    let mut stream = socket.connect(addr).await?;
    info!("Connected?");

    let vals = VecDeque::<u16>::new();
    let vals = Arc::new(RwLock::new(vals));

    // Read from the TCP socket and queue the values into the VecDequeue
    let vals_for_writing = vals.clone();
    tokio::spawn(async move {
        loop {
            let val = stream.read_u16().await;
            if let Err(e) = val {
                error!(?e, "Error reading from stream");
                break;

            }
            let val = val.unwrap();
            {
                let mut w = vals_for_writing.write().await;
                w.push_back(val);
                if w.len() > 1024 {
                    w.pop_front();
                }
            }
        }
    });

    // The actual plotter
    tokio::spawn(async move {
        let mut max = 0_u16;
        let mut min = 0_u16;
        loop {
            let buf: VecDeque<u16> = {
                let r = vals.read().await;
                r.clone()
            };
            let buf = buf.iter()
                .enumerate()
                .map(|(k, v)| {
                    if max < *v {
                        max = *v;
                    }
                    if min > *v {
                        min = *v;
                    }
                (k as f32, *v as f32)
            } )
                .collect::<Vec<_>>();

            term.move_cursor_to(0, 0).unwrap();
            Chart::new_with_y_range(400, 200, 0., buf.len() as f32, min as f32, max as f32)
                .linecolorplot(&Shape::Lines(buf.as_slice()), GREEN)
                .display();
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }).await?;

    Ok(())
}

pub fn setup_log(verbose: u64) {
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