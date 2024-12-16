use std::{
    fs::File,
    i16,
    path::{self, PathBuf},
};

use anyhow::{bail, Context};
use clap::{Args, Parser, Subcommand};
use linfa::{
    dataset::DatasetBase,
    traits::{Fit, Predict},
};
use linfa_ica::fast_ica::{FastIca, GFunc};
use minimp3::Frame;
use mp3lame_encoder::FlushNoGap;
use ndarray::{array, concatenate, Array, Axis};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use plotters::prelude::*;
use rand_xoshiro::Xoshiro256Plus;

const FONT: &str = "arial";
const BACKGROUND_COLOR: &RGBColor = &WHITE;
const MARGIN: i32 = 5;
const LABEL_AREA: i32 = 50;
const CAPTION_STYLE: (&str, i32, &RGBColor) = (FONT, 30, &BLACK);
const NUM_X_LABEL: usize = 20;
const NUM_Y_LABEL: usize = 10;
const LABEL_STYLE: (&str, i32) = ("arial", 20);
const LINE_WIDTH: u32 = 2;
const LEGEND1: ShapeStyle = ShapeStyle {
    stroke_width: LINE_WIDTH,
    color: RGBAColor(255, 0, 0, 1.0),
    filled: true,
};
const LEGEND2: ShapeStyle = ShapeStyle {
    stroke_width: LINE_WIDTH,
    color: RGBAColor(0, 0, 255, 1.0),
    filled: true,
};

#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// path to the plot output
    #[arg(short, long, default_value = "plot.png")]
    output: PathBuf,

    #[command(flatten)]
    resolution: Resolution,
}

#[derive(Subcommand, Clone, Debug)]
enum Commands {
    /// FastICA example using sine and box curves
    Example(CliExample),

    /// FastICA example using two speech signals
    Speech(CliSpeech),
}

#[derive(Debug, Args, Clone)]
struct CliExample {
    /// the number of samples used to create the original signals
    #[arg(short = 'n', long, default_value = "2000")]
    num_samples: usize,

    /// the start time of the original signals in seconds
    #[arg(short = 's', long, default_value = "0.")]
    time_start: f64,

    /// the start time of the original signals in seconds
    #[arg(short = 'e', long, default_value = "10.")]
    time_end: f64,
}

#[derive(Debug, Args, Clone)]
struct CliSpeech {
    /// path to first speaker file (mp3)
    #[arg(short = '1', long, default_value = "jane.mp3")]
    speaker_1: PathBuf,

    /// path to second speaker file (mp3)
    #[arg(short = '2', long, default_value = "bill.mp3")]
    speaker_2: PathBuf,

    #[arg(short = 'o', long, default_value = "mix.mp3")]
    mix_output: PathBuf,
}

#[derive(Debug, Copy, Clone, Parser)]
struct Resolution {
    /// width
    #[arg(short, default_value = "1440")]
    x: u32,

    /// height
    #[arg(short, default_value = "1080")]
    y: u32,
}

#[derive(Debug, Clone, Copy)]
enum Axes {
    S,
    X,
    Y,
}

#[derive(Debug, Clone, Copy)]
enum Graphs {
    S1,
    S2,
    X1,
    X2,
    Y1,
    Y2,
}

struct PlotArgs {
    output: String,
    res: Resolution,
    x: Vec<f64>,
    s_1: Vec<f64>,
    s_2: Vec<f64>,
    x_1: Vec<f64>,
    x_2: Vec<f64>,
    y_1: Vec<f64>,
    y_2: Vec<f64>,
}

impl PlotArgs {
    pub fn range(&self, axis: Axes) -> anyhow::Result<((f64, f64), (f64, f64))> {
        let x = (self.min(None)?, self.max(None)?);
        match axis {
            Axes::S => {
                let s_1_min = self.min(Some(Graphs::S1))?;
                let s_2_min = self.min(Some(Graphs::S2))?;
                let s_1_max = self.max(Some(Graphs::S1))?;
                let s_2_max = self.max(Some(Graphs::S2))?;

                Ok((x, (s_1_min.min(s_2_min), s_1_max.max(s_2_max))))
            }
            Axes::X => {
                let x_1_min = self.min(Some(Graphs::X1))?;
                let x_2_min = self.min(Some(Graphs::X2))?;
                let x_1_max = self.max(Some(Graphs::X1))?;
                let x_2_max = self.max(Some(Graphs::X2))?;

                Ok((x, (x_1_min.min(x_2_min), x_1_max.max(x_2_max))))
            }
            Axes::Y => {
                let y_1_min = self.min(Some(Graphs::Y1))?;
                let y_2_min = self.min(Some(Graphs::Y2))?;
                let y_1_max = self.max(Some(Graphs::Y1))?;
                let y_2_max = self.max(Some(Graphs::Y2))?;

                Ok((x, (y_1_min.min(y_2_min), y_1_max.max(y_2_max))))
            }
        }
    }

    pub fn graph(&self, graph: Graphs) -> Vec<(f64, f64)> {
        let mut vec = Vec::new();

        match graph {
            Graphs::S1 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.s_1.clone())
                    .for_each(|v| vec.push(v));
            }
            Graphs::S2 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.s_2.clone())
                    .for_each(|v| vec.push(v));
            }
            Graphs::X1 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.x_1.clone())
                    .for_each(|v| vec.push(v));
            }
            Graphs::X2 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.x_2.clone())
                    .for_each(|v| vec.push(v));
            }
            Graphs::Y1 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.y_1.clone())
                    .for_each(|v| vec.push(v));
            }
            Graphs::Y2 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.y_2.clone())
                    .for_each(|v| vec.push(v));
            }
        }

        vec
    }

    fn vec(&self, graph: Option<Graphs>) -> Vec<f64> {
        match graph {
            Some(Graphs::S1) => self.s_1.clone(),
            Some(Graphs::S2) => self.s_2.clone(),
            Some(Graphs::X1) => self.x_1.clone(),
            Some(Graphs::X2) => self.x_2.clone(),
            Some(Graphs::Y1) => self.y_1.clone(),
            Some(Graphs::Y2) => self.y_2.clone(),
            None => self.x.clone(),
        }
    }

    fn min(&self, graph: Option<Graphs>) -> anyhow::Result<f64> {
        let vec = self.vec(graph);
        vec.into_iter()
            .reduce(|v, acc| acc.min(v))
            .context("failed to find min")
    }

    fn max(&self, graph: Option<Graphs>) -> anyhow::Result<f64> {
        let vec = self.vec(graph);
        vec.into_iter()
            .reduce(|v, acc| acc.max(v))
            .context("failed to find max")
    }
}

fn plot(args: PlotArgs) -> anyhow::Result<()> {
    let root_area = BitMapBackend::new(&args.output, (args.res.x, args.res.y)).into_drawing_area();
    root_area.fill(BACKGROUND_COLOR)?;

    let rows = root_area.split_evenly((3, 1));

    let ranges = [Axes::S, Axes::X, Axes::Y];
    let captions = [
        "Original Signals",
        "Original Signals (mixed)",
        "FastICA Result",
    ];
    let graphs = [
        (Graphs::S1, Graphs::S2),
        (Graphs::X1, Graphs::X2),
        (Graphs::Y1, Graphs::Y2),
    ];
    let labels = [("s1", "s2"), ("x1", "x2"), ("y1", "y2")];

    for (i, row) in rows.iter().enumerate() {
        let ((x_start, x_end), (y_start, y_end)) = args.range(ranges[i])?;

        let mut cc = ChartBuilder::on(row)
            .margin(MARGIN)
            .margin_right(4 * MARGIN)
            .margin_left(8 * MARGIN)
            .set_left_and_bottom_label_area_size(LABEL_AREA)
            .caption(captions[i], CAPTION_STYLE)
            .build_cartesian_2d(x_start..x_end, y_start..y_end)?;

        cc.configure_mesh()
            .x_labels(NUM_X_LABEL)
            .y_labels(NUM_Y_LABEL)
            .label_style(LABEL_STYLE)
            .y_label_formatter(&|y| format!("{:1.3e}", y))
            .x_label_formatter(&|x| format!("{:1.3e}", x))
            .draw()?;

        cc.draw_series(LineSeries::new(args.graph(graphs[i].0), &RED))?
            .label(labels[i].0)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], LEGEND1));

        cc.draw_series(LineSeries::new(args.graph(graphs[i].1), &BLUE))?
            .label(labels[i].1)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], LEGEND2));

        cc.configure_series_labels()
            .border_style(BLACK)
            .label_font(LABEL_STYLE)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    root_area.present().context("failed saving plot")?;

    println!("saved plot to {}", args.output);

    Ok(())
}

fn read_audio_file<P>(path: P) -> anyhow::Result<Frame>
where
    P: AsRef<path::Path>,
{
    let mut frame = Frame {
        data: Vec::new(),
        sample_rate: 0,
        channels: 0,
        layer: 0,
        bitrate: 0,
    };

    let mut decoder = minimp3::Decoder::new(File::open(path)?);

    loop {
        match decoder.next_frame() {
            Ok(f) => {
                frame.data.extend_from_slice(&f.data);
                frame.sample_rate = f.sample_rate;
                frame.channels = f.channels;
                frame.layer = f.layer;
                frame.bitrate = f.bitrate;
            }
            Err(minimp3::Error::Eof) => break,
            Err(err) => bail!("mp3 decode: {err}"),
        }
    }

    Ok(frame)
}

fn x_from_frame(frame: &Frame) -> Vec<f64> {
    frame
        .data
        .iter()
        .enumerate()
        // TODO determine timestamps of sample
        .map(|(i, _)| i as f64)
        .collect()
}

fn encode_frame<P>(path: P, frame: Frame) -> anyhow::Result<()>
where
    P: AsRef<path::Path>,
{
    let mut encoder = mp3lame_encoder::Builder::new().context("new encoder")?;
    encoder
        .set_num_channels(frame.channels as u8)
        .expect("set channels");
    encoder
        .set_sample_rate(frame.sample_rate as u32)
        .expect("set sample rate");
    encoder
        .set_brate(mp3lame_encoder::Bitrate::Kbps96)
        .expect("set bitrate");
    let mut encoder = encoder.build().expect("build encoder");

    let input = mp3lame_encoder::MonoPcm(&frame.data);
    let mut output = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(input.0.len()));

    let encoded_size = encoder
        .encode(input, output.spare_capacity_mut())
        .expect("encoding");

    unsafe {
        output.set_len(output.len().wrapping_add(encoded_size));
    }

    let encoded_size = encoder
        .flush::<FlushNoGap>(output.spare_capacity_mut())
        .expect("flush");

    unsafe {
        output.set_len(output.len().wrapping_add(encoded_size));
    }

    std::fs::write(path, output)?;

    Ok(())
}

// fn mix_frames(f1: &Frame, f2: &Frame)

fn speech(args: CliSpeech) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let signal_1 = read_audio_file(&args.speaker_1)?;
    let signal_2 = read_audio_file(&args.speaker_2)?;

    let one = signal_1.data.clone();
    let two = signal_2.data.clone();

    let mix = Frame {
        data: one
            .iter()
            .zip(two)
            .map(|(one, two)| one + two)
            .collect::<Vec<i16>>(),
        sample_rate: signal_1.sample_rate,
        channels: signal_1.channels,
        layer: signal_1.layer,
        bitrate: signal_1.bitrate,
    };
    // TODO apply mixing matrix first
    // TODO save mixed signals separately or combined in one?
    // TODO rodio crate to directly play signals?
    encode_frame(&args.mix_output, mix)?;

    let x = x_from_frame(&signal_1);

    Ok((
        x,
        signal_1.data.iter().map(|v| v.to_owned() as f64).collect(),
        signal_2.data.iter().map(|v| v.to_owned() as f64).collect(),
    ))
}

fn example(args: CliExample) -> anyhow::Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let base = Array::linspace(args.time_start, args.time_end, args.num_samples);

    let s_1 = base.mapv(|x| (2. * x).sin());

    let s_2 = base.mapv(|x| match (4. * x).sin() > 0. {
        true => 1.,
        false => -1.,
    });

    let mut s = concatenate![
        Axis(1),
        s_1.clone().insert_axis(Axis(1)),
        s_2.clone().insert_axis(Axis(1))
    ];
    let mut rng = Xoshiro256Plus::seed_from_u64(1234);
    s += &Array::random_using((args.num_samples, 2), Uniform::new(-0.05, 0.05), &mut rng);

    Ok((base.to_vec(), s.column(0).to_vec(), s.column(1).to_vec()))
}

fn main() -> anyhow::Result<()> {
    let now = std::time::Instant::now();
    let cli = Cli::parse();

    let (base, s_1, s_2) = match cli.command {
        Commands::Example(ex) => example(ex)?,
        Commands::Speech(sp) => speech(sp)?,
    };

    let (s_1, s_2) = (Array::from_iter(s_1), Array::from_iter(s_2));

    let s = concatenate![
        Axis(1),
        s_1.clone().insert_axis(Axis(1)),
        s_2.clone().insert_axis(Axis(1))
    ];

    let a = array![[0.75, 1.5], [0.5, 2.]];
    let x = s.dot(&a.t());

    let ica = FastIca::params().gfunc(GFunc::Logcosh(1.0));
    let ica = ica.fit(&DatasetBase::from(x.view()))?;
    let y = ica.predict(&x);

    plot(PlotArgs {
        output: cli.output.to_str().context("invalid path")?.to_string(),
        res: cli.resolution,
        x: base,
        s_1: s.column(0).to_vec(),
        s_2: s.column(1).to_vec(),
        x_1: x.column(0).to_vec(),
        x_2: x.column(1).to_vec(),
        y_1: y.column(0).to_vec(),
        y_2: y.column(1).to_vec(),
    })?;

    println!("Runtime: {:?}", now.elapsed());

    Ok(())
}
