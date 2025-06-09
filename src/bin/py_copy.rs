use byteorder::{LittleEndian, ReadBytesExt};
use chrono::Local;
use clap::Parser;
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};
use eframe::egui;
use egui_plot::{Line, Legend, Plot, PlotPoints, PlotBounds};
use log::{debug, error, info, warn};
use num_complex::Complex;
use rustfft::{FftPlanner};
use std::{
    collections::VecDeque,
    io::{Cursor, Read, Write}, // Added Write for env_logger format
    net::TcpStream,
    sync::Arc, // Arc for Fft trait object
    thread,
    time::{Duration, Instant}, // Added Instant here
};
use bushtelegram_dsp::fft::apply_fft;
use bushtelegram_dsp::SAMPLE_SIZE;
use color_eyre::owo_colors::OwoColorize;
use eframe::emath::Pos2;
use egui::Rect;
use rustfft::num_traits::Zero;

// --- Configuration ---
// const SAMPLE_SIZE: usize = 256;
// For a real input of SAMPLE_SIZE, RFFT gives SAMPLE_SIZE/2 + 1 complex values.
const FFT_RAW_OUTPUT_LEN: usize = SAMPLE_SIZE / 2 + 1;

// User wants original bin indices 2 to 25 (inclusive)
const SLICE_START_BIN_INDEX: usize = 2; // 0-indexed
const SLICE_END_BIN_INDEX: usize = 25;   // 0-indexed, inclusive
// const SLICE_END_BIN_INDEX: usize = 50;   // 0-indexed, inclusive

// Calculated length of the data that will actually be plotted for FFT
const PLOT_POINTS_COUNT: usize = if SLICE_END_BIN_INDEX >= SLICE_START_BIN_INDEX {
    SLICE_END_BIN_INDEX - SLICE_START_BIN_INDEX + 1
} else {
    0 // This case should be caught by startup checks
};

const TCP_RECONNECT_DELAY_SECONDS: u64 = 5;
const CHANNEL_CAPACITY: usize = 5; // Store a few recent datasets

// --- PlotStatistics Struct ---
#[derive(Debug)]
struct PlotStatistics {
    overall_max_value_fft: f64,
    frame_averages_history_fft: VecDeque<f64>,
    current_moving_avg_of_frame_avgs_fft: f64,
    current_frame_max_value_fft: f64,
    current_frame_sum: f64,
}

impl PlotStatistics {
    fn new() -> Self {
        PlotStatistics {
            overall_max_value_fft: 0.0,
            frame_averages_history_fft: VecDeque::with_capacity(20), // Python: maxlen=20
            current_moving_avg_of_frame_avgs_fft: 0.0,
            current_frame_max_value_fft: 0.0,
            current_frame_sum: 0.0,
        }
    }

    fn update(&mut self, current_fft_values: &[f64]) {
        if current_fft_values.is_empty() {
            self.current_frame_max_value_fft = 0.0;
            self.current_frame_sum = 0.0;
            self.frame_averages_history_fft.push_back(0.0); // As per Python logic
        } else {
            self.current_frame_max_value_fft = current_fft_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            if self.current_frame_max_value_fft == f64::NEG_INFINITY { // All values were NaN or empty
                self.current_frame_max_value_fft = 0.0;
            }
            self.current_frame_sum = current_fft_values.iter().sum();
            let current_frame_average_fft = if !current_fft_values.is_empty() {
                self.current_frame_sum / current_fft_values.len() as f64
            } else { 0.0 };
            self.frame_averages_history_fft.push_back(current_frame_average_fft);
        }

        if self.current_frame_max_value_fft > self.overall_max_value_fft {
            self.overall_max_value_fft = self.current_frame_max_value_fft;
        }

        // Maintain maxlen for history
        while self.frame_averages_history_fft.len() > 20 {
            self.frame_averages_history_fft.pop_front();
        }

        if !self.frame_averages_history_fft.is_empty() {
            self.current_moving_avg_of_frame_avgs_fft =
                self.frame_averages_history_fft.iter().sum::<f64>()
                    / self.frame_averages_history_fft.len() as f64;
        } else {
            self.current_moving_avg_of_frame_avgs_fft = 0.0;
        }
    }
}

// --- TCP Data Receiver Thread ---
fn tcp_data_receiver_thread(
    server_host: String,
    server_port: u16,
    tx: Sender<Vec<f64>>,
) {
    let mut planner = FftPlanner::<f32>::new();
    let r2c_fft = planner.plan_fft_forward(SAMPLE_SIZE);
    // let mut scratch_buffer = vec![Complex::zero(); r2c_fft.get_inplace_scratch_len()];
    // let mut scratch_buffer = vec![Complex::zero(); SAMPLE_SIZE];

    // info!("TCP Receiver Thread started. FFT Scratch buffer size: {}", scratch_buffer.len());

    loop { // Reconnection loop
        info!("Attempting to connect to {}:{}...", server_host, server_port);
        match TcpStream::connect((server_host.as_str(), server_port)) {
            Ok(mut stream) => {
                info!("Connected to {}:{}", server_host, server_port);
                let mut read_buffer = [0u8; SAMPLE_SIZE * 4]; // 4 bytes per f32
                let mut val_buffer = [0f32; SAMPLE_SIZE];

                loop { // Data reading loop
                    match stream.read_exact(&mut read_buffer) {
                        Ok(_) => {
                            // let mut current_samples_f32_unpacked = Vec::with_capacity(SAMPLE_SIZE);
                            let mut rdr = Cursor::new(&read_buffer);
                            let mut unpack_ok = true;
                            for i in 0..SAMPLE_SIZE {
                                match rdr.read_f32::<LittleEndian>() {
                                    Ok(val) => val_buffer[i] = val,
                                    Err(e) => {
                                        error!(
                                            "Error unpacking f32 data: {}. Read {} bytes. Skipping packet.",
                                            e,
                                            read_buffer.len()
                                        );
                                        unpack_ok = false;
                                        break;
                                    }
                                }
                            }

                            if !unpack_ok {
                                continue; // Skip this corrupted/incomplete packet
                            }

                            let fft_magnitudes = apply_fft(&mut val_buffer);

                            // Slice the FFT output
                            if SLICE_END_BIN_INDEX >= fft_magnitudes.len() {
                                error!(
                                    "Slicing error: SLICE_END_BIN_INDEX ({}) is out of bounds for FFT magnitudes length ({}). Skipping frame.",
                                    SLICE_END_BIN_INDEX, fft_magnitudes.len()
                                );
                                continue;
                            }
                            let processed_fft_data = fft_magnitudes
                                [SLICE_START_BIN_INDEX..=SLICE_END_BIN_INDEX]
                                .iter()
                                .map(|v| *v as f64)
                                .collect::<Vec<_>>();

                            // Send data to GUI thread
                            match tx.try_send(processed_fft_data) {
                                Ok(_) => { /* Sent successfully */ }
                                Err(TrySendError::Full(_data_dropped)) => {
                                    warn!("Data queue full. Dropping current FFT frame to avoid lag.");
                                }
                                Err(TrySendError::Disconnected(_)) => {
                                    info!("GUI thread receiver disconnected. TCP thread exiting.");
                                    return; // Exit thread
                                }
                            }
                        }
                        Err(e) => {
                            error!("TCP connection error: {}. Attempting to reconnect...", e);
                            break; // Break from data reading loop, go to reconnection logic
                        }
                    }
                }
            }
            Err(e) => {
                error!(
                    "Failed to connect to {}:{}: {}. Retrying in {} seconds...",
                    server_host, server_port, e, TCP_RECONNECT_DELAY_SECONDS
                );
                thread::sleep(Duration::from_secs(TCP_RECONNECT_DELAY_SECONDS));
            }
        }
    }
}
// --- egui Application ---
struct FftPlotterApp {
    rx_fft_data: Receiver<Vec<f64>>,
    latest_fft_data: Option<Vec<f64>>,
    stats: PlotStatistics,
    flash_plot_background_end_time: Option<Instant>, // New field for flash timing
}

impl FftPlotterApp {
    fn new(_cc: &eframe::CreationContext<'_>, rx_fft_data: Receiver<Vec<f64>>) -> Self {
        Self {
            rx_fft_data,
            latest_fft_data: None,
            stats: PlotStatistics::new(),
            flash_plot_background_end_time: None, // Initialize to None
        }
    }
}

impl eframe::App for FftPlotterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Try to receive new data from the TCP thread
        match self.rx_fft_data.try_recv() {
            Ok(new_fft_data) => {
                // Update stats based on the newly received data, whether it's empty or not.
                // PlotStatistics::update handles empty slices correctly (sum becomes 0).
                self.stats.update(&new_fft_data);

                // Now check the condition based on the updated stats
                if self.stats.current_frame_sum > -200.0 {
                    self.flash_plot_background_end_time =
                        Some(Instant::now() + Duration::from_millis(300));
                    info!("FOOTSTEP!");
                }
                self.latest_fft_data = Some(new_fft_data);
            }
            Err(TryRecvError::Empty) => { /* No new data, plot with old data if available. Flash timer continues independently. */ }
            Err(TryRecvError::Disconnected) => {
                error!("TCP receiver thread disconnected! Plot will no longer update.");
                // Optionally, close the app or show a persistent error message
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Live FFT Data from TCP Stream");
            ui.separator();

            // FFT Plot
            ui.label("FFT Magnitude Spectrum"); // Title for the plot

            // --- Logic for flashing plot background ---
            let original_plot_bg_fill = ui.visuals().widgets.noninteractive.bg_fill;
            let mut restore_visuals = false;
            let mut paint_footstep = false;

            if let Some(end_time) = self.flash_plot_background_end_time {
                if Instant::now() < end_time {
                    paint_footstep = true;
                } else {
                    // Time has passed, reset the flash state for the next frame
                    self.flash_plot_background_end_time = None;
                    restore_visuals = true;
                }
            }
            // --- End of flash logic ---

            let plot = Plot::new("fft_spectrum_plot")
                .legend(Legend::default())
                .height(300.0) // Adjust as needed
                .x_axis_label("Frequency Bin Index (Post-Slice)")
                .y_axis_label("Magnitude")
                .show_x(true)
                .show_y(true)
                .show_background(true);

            if paint_footstep {
                // Paint background of the plot red
                ui.painter().rect_filled(
                    Rect {
                        min: Pos2 {
                            x: 0.0,
                            y: 0.0
                        },
                        max: Pos2 {
                            x: 800.0,
                            y: 350.0
                        },
                    },
                    0.0, // Corner radius
                    egui::Color32::from_rgb(255, 0, 0).linear_multiply(0.2), // Semi-transparent red
                );
            }
            plot.show(ui, |plot_ui| {

                let mut x_max_plot = (PLOT_POINTS_COUNT.saturating_sub(1)) as f64;

                if let Some(data) = &self.latest_fft_data {
                    if !data.is_empty() {
                        // FFT data line
                        let fft_points: PlotPoints = data
                            .iter()
                            .enumerate()
                            .map(|(i, &val)| [i as f64, val])
                            .collect();
                        plot_ui.line(Line::new(fft_points).name("FFT Data").color(egui::Color32::LIGHT_BLUE));

                        x_max_plot = (data.len().saturating_sub(1)) as f64;

                        // Moving average line
                        let mov_avg_val = self.stats.current_moving_avg_of_frame_avgs_fft;
                        if data.len() > 1 { // Draw line only if there are at least two points for x-axis
                            let moving_avg_points = PlotPoints::new(vec![
                                [0.0, mov_avg_val],
                                [x_max_plot, mov_avg_val],
                            ]);
                            plot_ui.line(
                                Line::new(moving_avg_points)
                                    .name("Moving Avg (Frame Avgs)")
                                    .color(egui::Color32::YELLOW)
                                    .style(egui_plot::LineStyle::dashed_loose()),
                            );
                        } else if data.len() == 1 { // Draw single point for moving average if only one data point
                            plot_ui.points(egui_plot::Points::new(PlotPoints::new(vec![[0.0, mov_avg_val]]))
                                .name("Moving Avg (Frame Avgs)")
                                .color(egui::Color32::YELLOW)
                                .radius(2.0)
                            );
                        }
                    }
                }
                // Set fixed Y-axis bounds [0, 50] as per Python script's active setting
                // Ensure x_max_plot is at least 0 for valid bounds
                let current_x_max = if x_max_plot < 0.0 { 0.0 } else { x_max_plot };
                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                    [0.0, 0.0], // x_min, y_min
                    [current_x_max, 50.0],  // x_max, y_max
                ));
            });

            // Restore original visuals if they were changed for the plot
            if restore_visuals {
                ui.visuals_mut().widgets.noninteractive.bg_fill = original_plot_bg_fill;
            }

            ui.separator();
            ui.label("FFT Statistics:");
            ui.label(format!(
                "Frame Max: {:.2}",
                self.stats.current_frame_max_value_fft
            ));
            ui.label(format!("Frame Sum: {:.2}", self.stats.current_frame_sum)); // This sum triggers the flash
            ui.label(format!(
                "Overall Max: {:.2}",
                self.stats.overall_max_value_fft
            ));
            ui.label(format!(
                "Mov.Avg (Frame Avgs): {:.2}",
                self.stats.current_moving_avg_of_frame_avgs_fft
            ));
        });

        // Request redraw for continuous updates (e.g., ~20 FPS)
        ctx.request_repaint_after(Duration::from_millis(50));
    }
}

// --- Command Line Arguments ---
#[derive(Parser, Debug)]
#[clap(author, version, about = "Rust FFT Plotter from TCP Stream", long_about = None)]
struct CliArgs {
    #[clap(short, long, default_value = "192.168.88.250:1234", help = "Server address and port (e.g., 192.168.88.250:1234)")]
    server: String,

    #[clap(short, long, action = clap::ArgAction::Count, help = "Verbosity level (-v for DEBUG)")]
    verbose: u8,
}

fn main() -> Result<(), eframe::Error> {
    let args = CliArgs::parse();

    // Logging Setup (mimicking Python script's format and levels)
    let log_level = if args.verbose > 0 {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    env_logger::Builder::new()
        .format(|buf, record| {
            writeln!(
                buf,
                "{} - {} - {} - {}",
                Local::now().format("%Y-%m-%d %H:%M:%S%.3f"), // Adjusted for milliseconds
                std::thread::current().name().unwrap_or("unknown"),
                record.level(),
                record.args()
            )
        })
        .filter_level(log_level)
        .init();

    // --- Sanity checks for FFT slicing configuration ---
    if PLOT_POINTS_COUNT == 0 {
        error!(
            "Calculated PLOT_POINTS_COUNT is 0. Check SLICE_START_BIN_INDEX ({}) and SLICE_END_BIN_INDEX ({}). Application will exit.",
            SLICE_START_BIN_INDEX, SLICE_END_BIN_INDEX
        );
        return Ok(()); // Exit gracefully
    }
    if SLICE_END_BIN_INDEX >= FFT_RAW_OUTPUT_LEN {
        error!(
            "Configuration error: SLICE_END_BIN_INDEX ({}) is out of bounds for FFT_RAW_OUTPUT_LEN ({}). Application will exit.",
            SLICE_END_BIN_INDEX, FFT_RAW_OUTPUT_LEN
        );
        return Ok(());
    }
    info!("Plotting FFT bins from index {} to {} (inclusive). Total points: {}", SLICE_START_BIN_INDEX, SLICE_END_BIN_INDEX, PLOT_POINTS_COUNT);


    // Parse server host and port
    let parts: Vec<&str> = args.server.split(':').collect();
    if parts.len() != 2 {
        error!("Invalid server format. Use HOST:PORT (e.g., 127.0.0.1:1234)");
        return Ok(());
    }
    let server_host = parts[0].to_string();
    let server_port: u16 = match parts[1].parse() {
        Ok(p) => p,
        Err(_) => {
            error!("Invalid server port number: {}", parts[1]);
            return Ok(());
        }
    };

    // Create a channel for (FFT data)
    let (tx_fft_data, rx_fft_data) = crossbeam_channel::bounded::<Vec<f64>>(CHANNEL_CAPACITY);

    // Start TCP receiver thread
    let _tcp_thread_handle = thread::Builder::new()
        .name("TCPReceiverThread".to_string())
        .spawn(move || {
            tcp_data_receiver_thread(server_host, server_port, tx_fft_data);
        })
        .expect("Failed to spawn TCP receiver thread");

    info!("Starting eframe GUI...");
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0]) // Initial window size
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Rust FFT Plotter",
        native_options,
        Box::new(move |cc| Box::new(FftPlotterApp::new(cc, rx_fft_data))),
    )
}
