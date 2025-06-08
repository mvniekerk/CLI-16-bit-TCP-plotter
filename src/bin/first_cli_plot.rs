use std::collections::VecDeque;
use std::env;
use std::error::Error;
use argh::FromArgs;
use tokio::io::AsyncReadExt;
use tokio::net::TcpSocket;
use tracing::{error, warn};

use std::sync::{Arc};
use std::time::Duration;
use bushtelegram_dsp::{CURVE_SIZE_INCORPORATED, FFT_OUTPUT, SAMPLE_SIZE, SAMPLING_FREQUENCY_HZ, SPLIT_AT_LEFT, SPLIT_AT_RIGHT};
use bushtelegram_dsp::fft::apply_fft;
use bushtelegram_dsp::filters::{apply_butterworth_bandpass_filter, energy_in_target_band};
use color_eyre::owo_colors::OwoColorize;
use log::info;
// use textplots::{Chart, ColorPlot, Plot, Shape};
use tokio::sync::{OnceCell, RwLock};
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
    #[argh(option, short = 's', default = "String::from(\"192.168.88.250:1234\")")]
    server: String,
}

pub const DEFAULT_FILTER_ENV: &str = "LOG_LEVEL";

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

use std::time::{Instant};

use color_eyre::Result;
use lazy_static::lazy_static;
use ratatui::{
    crossterm::event::{self, Event, KeyCode},
    layout::{Constraint, Layout, Rect},
    style::{Color, Modifier, Style, Stylize},
    symbols::{self, Marker},
    text::{Line, Span},
    widgets::{Axis, Block, Chart, Dataset, GraphType, LegendPosition},
    DefaultTerminal, Frame,
};
use ratatui::text::Text;
// use ratatui::crossterm::style::Stylize;

lazy_static!(
    pub static ref VALUES: Arc<RwLock<Vec<f64>>> = Arc::default();
);

#[tokio::main]
async fn main() -> Result<()> {
    // color_eyre::install()?;

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


    // Read from the TCP socket and queue the values into the VecDequeue
    let footstep_detected = Arc::new(RwLock::new(false));
    let footstep_detected_for_filter = footstep_detected.clone();
    let energy = Arc::new(RwLock::new(0f32));
    let energy_clone = energy.clone();
    let values = VALUES.clone();
    tokio::spawn(async move {
        loop {
            let mut new_val = [0f32; SAMPLE_SIZE];
            for i in 0..SAMPLE_SIZE {

                new_val[i] = match stream.read_f32_le().await {
                    Ok(val) => val as f32,
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

            {
                let mut w = values.write().await;
                let (_l, r) = new_val.split_at(SPLIT_AT_LEFT);
                let (l, _r) = r.split_at(r.len() - SPLIT_AT_RIGHT);
                *w = l.iter().map(|s| *s as f64).collect::<Vec<_>>();
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
    let terminal = ratatui::init();
    let app_result = App::new().run(terminal).await;
    ratatui::restore();
    app_result
}

struct App {
    pub max: f64,
    pub min: f64,
    pub current_max: f64,
    pub current_min: f64,
    pub max_energy: f64,
    pub averages: Vec<f64>,
    pub averages_average: f64
}

impl App {
    fn new() -> Self {
        Self {
            max: 0.0,
            min: 0.0,
            current_max: 0.0,
            current_min: 0.0,
            max_energy: 0.0,
            averages: Vec::new(),
            averages_average: 0.0
        }
    }

    async fn run(mut self, mut terminal: DefaultTerminal) -> Result<()> {
        let tick_rate = Duration::from_millis(10);
        let mut last_tick = Instant::now();
        loop {
            let values = {
                let r = VALUES.read().await;
                r.clone()
            };

            let mut current_max = 0.0;
            let mut current_min = 2000.0;
            let mut average = 0.0;

            values
                .iter()
                .for_each(|v| {
                    if current_max < *v {
                        current_max = *v;
                    }
                    if current_min > *v {
                        current_min = *v;
                    }
                    average += *v;
                });

            if current_max > self.max {
                self.max = current_max;
            }
            if current_min < self.min {
                self.min = current_min;
            }

            if self.averages.len() > 20 {
                self.averages.reverse();
                self.averages.pop();
                self.averages.reverse();
            }

            self.averages.push(average / values.len() as f64);

            let average: f64 = self.averages.iter().sum();
            self.averages_average = average / self.averages.len() as f64;

            terminal.draw(|frame|
                self.draw(frame, values))?;

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') {
                        return Ok(());
                    }
                    if key.code == KeyCode::Down {
                        self.max = 0.0;
                        self.current_max = 0.0;
                        self.averages_average = 0.0;
                        self.averages = Vec::new();
                    }
                }
            }
            if last_tick.elapsed() >= tick_rate {
                self.on_tick();
                last_tick = Instant::now();
            }
        }
    }

    fn on_tick(&mut self) {
    }

    fn draw(&self, frame: &mut Frame, values: Vec<f64>) {
        // let [animated_chart, bar_chart] =
        //     Layout::horizontal([Constraint::Fill(1), Constraint::Length(29)]).areas(frame.area());
        let [top, bottom] = Layout::vertical([Constraint::Fill(1), Constraint::Length(10)]).areas(frame.area());
        let [animated_chart, bar_chart] =
            Layout::horizontal([Constraint::Fill(1), Constraint::Length(29)]).areas(top);
        // let [line_chart, scatter] = Layout::horizontal([Constraint::Fill(1); 2]).areas(bottom);

        self.render_animated_chart(frame, animated_chart, values);
        render_barchart(frame, bar_chart);
        let t = Text::from(format!("Max {:.2} Min {:.2}", self.max, self.min));
        frame.render_widget(t, bottom);
        // render_line_chart(frame, line_chart);
        // render_scatter(frame, scatter);
    }

    fn render_animated_chart(&self, frame: &mut Frame, area: Rect, values: Vec<f64>) {
        let x_labels = vec![
            Span::styled(
                format!("{}", 0.0),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("{}", values.len() / 2)),
            Span::styled(
                format!("{}", values.len()),
                Style::default().add_modifier(Modifier::BOLD),
            ),
        ];
        let values = values.into_iter().enumerate().map(|(i, v)| (i as f64, v)).collect::<Vec<_>>();
        let averages_line = [(0.0, self.averages_average), (values.len() as f64, self.averages_average)];
        let datasets = vec![
            Dataset::default()
                .name("avg")
                .marker(symbols::Marker::Dot)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Yellow))
                .data(&averages_line),
            Dataset::default()
                .name("data")
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(Color::Cyan))
                .data(&values),
            // Dataset::default()
            //     .name("data3")
            //     .marker(symbols::Marker::Braille)
            //     .style(Style::default().fg(Color::Yellow))
            //     .data(&self.data2),
        ];

        let y_label_0 = format!("{:.2}", self.averages_average / 2.0).bold();
        let y_label_1 = format!("{:.2}", self.max).bold();
        let y_label_2_avg = format!("{:.2}", self.averages_average).bold();

        let chart = Chart::new(datasets)
            .block(Block::bordered())
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .labels(x_labels)
                    .bounds([0.0, values.len() as f64]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Color::Gray))
                    .labels([y_label_0, y_label_2_avg, y_label_1])
                    .bounds([self.averages_average / 2.0, self.max]),
            );

        frame.render_widget(chart, area);
    }
}

fn render_barchart(frame: &mut Frame, bar_chart: Rect) {
    let dataset = Dataset::default()
        .marker(symbols::Marker::HalfBlock)
        .style(Style::new().fg(Color::Blue))
        .graph_type(GraphType::Bar)
        // a bell curve
        .data(&[
            (0., 0.4),
            (10., 2.9),
            (20., 13.5),
            (30., 41.1),
            (40., 80.1),
            (50., 100.0),
            (60., 80.1),
            (70., 41.1),
            (80., 13.5),
            (90., 2.9),
            (100., 0.4),
        ]);

    let chart = Chart::new(vec![dataset])
        .block(Block::bordered().title_top(Line::from("Bar chart").cyan().bold().centered()))
        .x_axis(
            Axis::default()
                .style(Style::default().gray())
                .bounds([0.0, 100.0])
                .labels(["0".bold(), "50".into(), "100.0".bold()]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().gray())
                .bounds([0.0, 100.0])
                .labels(["0".bold(), "50".into(), "100.0".bold()]),
        )
        .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));

    frame.render_widget(chart, bar_chart);
}

fn render_line_chart(frame: &mut Frame, area: Rect) {
    let datasets = vec![Dataset::default()
        .name("Line from only 2 points".italic())
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Yellow))
        .graph_type(GraphType::Line)
        .data(&[(1., 1.), (4., 4.)])];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(Line::from("Line chart").cyan().bold().centered()))
        .x_axis(
            Axis::default()
                .title("X Axis")
                .style(Style::default().gray())
                .bounds([0.0, 5.0])
                .labels(["0".bold(), "2.5".into(), "5.0".bold()]),
        )
        .y_axis(
            Axis::default()
                .title("Y Axis")
                .style(Style::default().gray())
                .bounds([0.0, 5.0])
                .labels(["0".bold(), "2.5".into(), "5.0".bold()]),
        )
        .legend_position(Some(LegendPosition::TopLeft))
        .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));

    frame.render_widget(chart, area);
}

fn render_scatter(frame: &mut Frame, area: Rect) {
    let datasets = vec![
        Dataset::default()
            .name("Heavy")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Style::new().yellow())
            .data(&HEAVY_PAYLOAD_DATA),
        Dataset::default()
            .name("Medium".underlined())
            .marker(Marker::Braille)
            .graph_type(GraphType::Scatter)
            .style(Style::new().magenta())
            .data(&MEDIUM_PAYLOAD_DATA),
        Dataset::default()
            .name("Small")
            .marker(Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Style::new().cyan())
            .data(&SMALL_PAYLOAD_DATA),
    ];

    let chart = Chart::new(datasets)
        .block(Block::bordered().title(Line::from("Scatter chart").cyan().bold().centered()))
        .x_axis(
            Axis::default()
                .title("Year")
                .bounds([1960., 2020.])
                .style(Style::default().fg(Color::Gray))
                .labels(["1960", "1990", "2020"]),
        )
        .y_axis(
            Axis::default()
                .title("Cost")
                .bounds([0., 75000.])
                .style(Style::default().fg(Color::Gray))
                .labels(["0", "37 500", "75 000"]),
        )
        .hidden_legend_constraints((Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)));

    frame.render_widget(chart, area);
}

// Data from https://ourworldindata.org/space-exploration-satellites
const HEAVY_PAYLOAD_DATA: [(f64, f64); 9] = [
    (1965., 8200.),
    (1967., 5400.),
    (1981., 65400.),
    (1989., 30800.),
    (1997., 10200.),
    (2004., 11600.),
    (2014., 4500.),
    (2016., 7900.),
    (2018., 1500.),
];

const MEDIUM_PAYLOAD_DATA: [(f64, f64); 29] = [
    (1963., 29500.),
    (1964., 30600.),
    (1965., 177_900.),
    (1965., 21000.),
    (1966., 17900.),
    (1966., 8400.),
    (1975., 17500.),
    (1982., 8300.),
    (1985., 5100.),
    (1988., 18300.),
    (1990., 38800.),
    (1990., 9900.),
    (1991., 18700.),
    (1992., 9100.),
    (1994., 10500.),
    (1994., 8500.),
    (1994., 8700.),
    (1997., 6200.),
    (1999., 18000.),
    (1999., 7600.),
    (1999., 8900.),
    (1999., 9600.),
    (2000., 16000.),
    (2001., 10000.),
    (2002., 10400.),
    (2002., 8100.),
    (2010., 2600.),
    (2013., 13600.),
    (2017., 8000.),
];

const SMALL_PAYLOAD_DATA: [(f64, f64); 23] = [
    (1961., 118_500.),
    (1962., 14900.),
    (1975., 21400.),
    (1980., 32800.),
    (1988., 31100.),
    (1990., 41100.),
    (1993., 23600.),
    (1994., 20600.),
    (1994., 34600.),
    (1996., 50600.),
    (1997., 19200.),
    (1997., 45800.),
    (1998., 19100.),
    (2000., 73100.),
    (2003., 11200.),
    (2008., 12600.),
    (2010., 30500.),
    (2012., 20000.),
    (2013., 10600.),
    (2013., 34500.),
    (2015., 10600.),
    (2018., 23100.),
    (2019., 17300.),
];