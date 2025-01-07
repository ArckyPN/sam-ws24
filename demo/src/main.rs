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

// Plotting constants
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

/// type alias for clarity
type SignalData = (Vec<f64>, Vec<f64>, Vec<f64>, Option<Frame>);

/// Command Line Interface
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// which type of plot to create
    #[command(subcommand)]
    command: Commands,

    /// path to the plot output
    #[arg(short, long, default_value = "plot.png")]
    output: PathBuf,

    /// plot output resolution [pixels]
    #[command(flatten)]
    resolution: Resolution,
}

/// Available Commands
#[derive(Subcommand, Clone, Debug)]
enum Commands {
    /// FastICA example using sine and box waves
    Example(CliExample),

    /// FastICA example using two speech signals (mp3 files)
    Speech(CliSpeech),
}

/// Additional flags for the example plot
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

/// Additional flags for the speech plot
#[derive(Debug, Args, Clone)]
struct CliSpeech {
    /// path to first speaker file (mp3)
    #[arg(short = '1', long, default_value = "speaker1.mp3")]
    speaker_1: PathBuf,

    /// path to second speaker file (mp3)
    #[arg(short = '2', long, default_value = "speaker2.mp3")]
    speaker_2: PathBuf,
}

/// Plot output resolution flags
#[derive(Debug, Copy, Clone, Parser)]
struct Resolution {
    /// width [pixels]
    #[arg(short, default_value = "1440")]
    x: u32,

    /// height [pixels]
    #[arg(short, default_value = "1080")]
    y: u32,
}

/// Plotting helper
#[derive(Debug, Clone, Copy)]
enum Signal {
    Original,
    Mixed,
    Reconstructed,
}

/// Plotting helper
#[derive(Debug, Clone, Copy)]
enum SignalGraphs {
    Original1,
    Original2,
    Mixed1,
    Mixed2,
    Reconstructed1,
    Reconstructed2,
}

/// Plotting function arguments
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
    speech: bool,
}

impl PlotArgs {
    /// returns the x and y number ranges for the given signal as tuple:
    ///
    /// ((x_min, x_max), (y_min, y_max))
    pub fn range(&self, axis: Signal) -> anyhow::Result<((f64, f64), (f64, f64))> {
        // x ranges are the same for all
        let x = (self.min(None)?, self.max(None)?);

        match axis {
            Signal::Original => {
                let s_1_min = self.min(Some(SignalGraphs::Original1))?;
                let s_2_min = self.min(Some(SignalGraphs::Original2))?;
                let s_1_max = self.max(Some(SignalGraphs::Original1))?;
                let s_2_max = self.max(Some(SignalGraphs::Original2))?;

                Ok((x, (s_1_min.min(s_2_min), s_1_max.max(s_2_max))))
            }
            Signal::Mixed => {
                let x_1_min = self.min(Some(SignalGraphs::Mixed1))?;
                let x_2_min = self.min(Some(SignalGraphs::Mixed2))?;
                let x_1_max = self.max(Some(SignalGraphs::Mixed1))?;
                let x_2_max = self.max(Some(SignalGraphs::Mixed2))?;

                Ok((x, (x_1_min.min(x_2_min), x_1_max.max(x_2_max))))
            }
            Signal::Reconstructed => {
                let y_1_min = self.min(Some(SignalGraphs::Reconstructed1))?;
                let y_2_min = self.min(Some(SignalGraphs::Reconstructed2))?;
                let y_1_max = self.max(Some(SignalGraphs::Reconstructed1))?;
                let y_2_max = self.max(Some(SignalGraphs::Reconstructed2))?;

                Ok((x, (y_1_min.min(y_2_min), y_1_max.max(y_2_max))))
            }
        }
    }

    /// returns the x,y number pairs to plot the graphs
    ///
    /// [(x, y),...]
    pub fn graph(&self, graph: SignalGraphs) -> Vec<(f64, f64)> {
        // initial empty vector
        let mut vec = Vec::new();

        // push the respective graph data into `vec`
        match graph {
            SignalGraphs::Original1 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.s_1.clone())
                    .for_each(|v| vec.push(v));
            }
            SignalGraphs::Original2 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.s_2.clone())
                    .for_each(|v| vec.push(v));
            }
            SignalGraphs::Mixed1 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.x_1.clone())
                    .for_each(|v| vec.push(v));
            }
            SignalGraphs::Mixed2 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.x_2.clone())
                    .for_each(|v| vec.push(v));
            }
            SignalGraphs::Reconstructed1 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.y_1.clone())
                    .for_each(|v| vec.push(v));
            }
            SignalGraphs::Reconstructed2 => {
                self.x
                    .clone()
                    .into_iter()
                    .zip(self.y_2.clone())
                    .for_each(|v| vec.push(v));
            }
        }

        vec
    }

    /// returns a clone of the specified graph or x if None given
    fn vec(&self, graph: Option<SignalGraphs>) -> Vec<f64> {
        match graph {
            Some(SignalGraphs::Original1) => self.s_1.clone(),
            Some(SignalGraphs::Original2) => self.s_2.clone(),
            Some(SignalGraphs::Mixed1) => self.x_1.clone(),
            Some(SignalGraphs::Mixed2) => self.x_2.clone(),
            Some(SignalGraphs::Reconstructed1) => self.y_1.clone(),
            Some(SignalGraphs::Reconstructed2) => self.y_2.clone(),
            None => self.x.clone(),
        }
    }

    /// returns the minimum of the given graph
    fn min(&self, graph: Option<SignalGraphs>) -> anyhow::Result<f64> {
        let vec = self.vec(graph);
        vec.into_iter()
            .reduce(|v, acc| acc.min(v))
            .context("failed to find min")
    }

    /// returns the maximum of the given graph
    fn max(&self, graph: Option<SignalGraphs>) -> anyhow::Result<f64> {
        let vec = self.vec(graph);
        vec.into_iter()
            .reduce(|v, acc| acc.max(v))
            .context("failed to find max")
    }
}

/// creates the plot
fn plot(args: PlotArgs) -> anyhow::Result<()> {
    // create the canvas and fill it with the background color const
    let root_area = BitMapBackend::new(&args.output, (args.res.x, args.res.y)).into_drawing_area();
    root_area.fill(BACKGROUND_COLOR)?;

    // split the canvas into three equal subplot rows
    let rows = root_area.split_evenly((3, 1));

    // define iterable variable for plotting
    let ranges = [Signal::Original, Signal::Mixed, Signal::Reconstructed];
    let captions = [
        "Original Signals",
        "Original Signals (mixed)",
        "FastICA Result",
    ];
    let graphs = [
        (SignalGraphs::Original1, SignalGraphs::Original2),
        (SignalGraphs::Mixed1, SignalGraphs::Mixed2),
        (SignalGraphs::Reconstructed1, SignalGraphs::Reconstructed2),
    ];
    let labels = [("s1", "s2"), ("x1", "x2"), ("y1", "y2")];

    // iterate over the subplots by element and index (to access the values above)
    for (i, row) in rows.iter().enumerate() {
        // retrieve the plot ranges for current subplot
        let ((x_start, x_end), (y_start, y_end)) = args.range(ranges[i])?;

        // create the subplot drawer on the current row
        let mut cc = ChartBuilder::on(row)
            .margin(MARGIN)
            .margin_right(4 * MARGIN)
            .margin_left(8 * MARGIN)
            .set_left_and_bottom_label_area_size(LABEL_AREA)
            .caption(captions[i], CAPTION_STYLE)
            .build_cartesian_2d(x_start..x_end, y_start..y_end)?;

        // style the subplot
        cc.configure_mesh()
            .x_labels(NUM_X_LABEL)
            .y_labels(NUM_Y_LABEL)
            .label_style(LABEL_STYLE)
            .y_label_formatter(&|y| format!("{:1.3e}", y))
            .x_label_formatter(&|x| format!("{:1.3e}", x))
            .draw()?;

        // draw the first graph
        cc.draw_series(LineSeries::new(args.graph(graphs[i].0), &RED))?
            .label(labels[i].0)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], LEGEND1));

        // draw the second graph
        cc.draw_series(LineSeries::new(args.graph(graphs[i].1), &BLUE))?
            .label(labels[i].1)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], LEGEND2));

        // set the legend position based on command
        let pos = if args.speech {
            SeriesLabelPosition::UpperRight
        } else {
            SeriesLabelPosition::MiddleRight
        };

        // style the legend
        cc.configure_series_labels()
            .border_style(BLACK)
            .label_font(LABEL_STYLE)
            .position(pos)
            .draw()?;
    }

    // save the plot
    root_area.present().context("failed saving plot")?;

    println!("saved plot to {}", args.output);

    Ok(())
}

/// read an mp3 file and return its data
fn read_audio_file<P>(path: P) -> anyhow::Result<Frame>
where
    P: AsRef<path::Path>,
{
    // init empty audio data
    let mut frame = Frame {
        data: Vec::new(),
        sample_rate: 0,
        channels: 0,
        layer: 0,
        bitrate: 0,
    };

    // init mp3 decoder
    let mut decoder = minimp3::Decoder::new(File::open(path)?);

    loop {
        // decode all available frames until end of file or error
        match decoder.next_frame() {
            // on success overwrite the audio data
            Ok(f) => {
                frame.data.extend_from_slice(&f.data);
                frame.sample_rate = f.sample_rate;
                frame.channels = f.channels;
                frame.layer = f.layer;
                frame.bitrate = f.bitrate;
            }
            // break loop on end of file
            Err(minimp3::Error::Eof) => break,
            // throw error otherwise
            Err(err) => bail!("mp3 decode: {err}"),
        }
    }

    Ok(frame)
}

/// returns x values based on the audio data
fn x_from_frame(frame: &Frame) -> Vec<f64> {
    frame
        .data
        .iter()
        .enumerate()
        .map(|(i, _)| i as f64 * 1. / frame.sample_rate as f64)
        .collect()
}

/// encodes audio data back to an mp4 file
fn save_audio_to_file<P>(path: P, frame: Frame, tag: Option<&str>) -> anyhow::Result<()>
where
    P: AsRef<path::Path>,
{
    // init mp3 encoder
    let mut encoder = mp3lame_encoder::Builder::new().context("new encoder")?;
    // set number of channels
    encoder
        .set_num_channels(frame.channels as u8)
        .expect("set channels");
    // set sampling rate
    encoder
        .set_sample_rate(frame.sample_rate as u32)
        .expect("set sample rate");
    // set bitrate
    encoder
        .set_brate(mp3lame_encoder::Bitrate::Kbps96)
        .expect("set bitrate");
    // finish building encoder
    let mut encoder = encoder.build().expect("build encoder");

    // prepare data
    let input = mp3lame_encoder::MonoPcm(&frame.data);
    let mut output = Vec::with_capacity(mp3lame_encoder::max_required_buffer_size(input.0.len()));

    // encode audio data
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

    let path = match tag {
        Some(tag) => {
            let path = path.as_ref();
            let ext = path
                .extension()
                .context("missing file extension")?
                .to_str()
                .context("invalid extension string")?;
            let file = path
                .file_name()
                .context("missing file name")?
                .to_str()
                .context("invalid file name string")?;
            let path_str = path.to_str().context("invalid path string")?;

            let replace = file.replace(&format!(".{}", ext), &format!("_{}.{}", tag, ext));
            let new_path = path_str.replace(file, &replace);
            PathBuf::from(new_path)
        }
        None => path.as_ref().to_path_buf(),
    };

    // write encoded data to file
    std::fs::write(&path, output)?;
    println!("saved audio to {:?}", path);

    Ok(())
}

/// prepares the data of the speech command
///
/// returns x values, audio signal 1 values, audio signal 2 values and audio meta data
fn speech(
    args: CliSpeech,
    mix: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>>,
) -> anyhow::Result<SignalData> {
    // read both audio files
    let signal_1 = read_audio_file(&args.speaker_1)?;
    let signal_2 = read_audio_file(&args.speaker_2)?;

    // clone the audio data and convert values to floats
    let one = signal_1
        .data
        .clone()
        .iter()
        .map(|e| *e as f64)
        .collect::<Vec<f64>>();
    let two = signal_2
        .data
        .clone()
        .iter()
        .map(|e| *e as f64)
        .collect::<Vec<f64>>();

    // make sure both signals are of the same length
    // slice both to the length of the shorter one
    let boundary = one.len().min(two.len());
    let one = one[..boundary].to_vec();
    let two = two[..boundary].to_vec();

    // apply the mixing matrix
    let s = concatenate![
        Axis(1),
        Array::from_iter(one.clone()).insert_axis(Axis(1)),
        Array::from_iter(two.clone()).insert_axis(Axis(1))
    ];
    let x = s.dot(&mix.t());

    // extract mixed signals and convert values back to integers
    let one_i16 = x
        .column(0)
        .to_vec()
        .iter()
        .map(|e| *e as i16)
        .collect::<Vec<i16>>();
    let two_i16 = x
        .column(1)
        .to_vec()
        .iter()
        .map(|e| *e as i16)
        .collect::<Vec<i16>>();

    // create audio signals from the mixed signals
    let base = Frame {
        data: Vec::new(),
        sample_rate: signal_1.sample_rate,
        channels: signal_1.channels,
        layer: signal_1.layer,
        bitrate: signal_1.bitrate,
    };
    let mut mix_1 = base.clone();
    mix_1.data = one_i16;
    let mut mix_2 = base.clone();
    mix_2.data = two_i16;

    // write audio signals to disk
    save_audio_to_file(&args.speaker_1, mix_1, Some("mix"))?;
    save_audio_to_file(&args.speaker_2, mix_2, Some("mix"))?;

    Ok((x_from_frame(&signal_1), one, two, Some(base)))
}

/// prepares the data of the example command
///
/// returns x values, sine wave values, box wave values and always None
fn example(args: CliExample) -> anyhow::Result<SignalData> {
    // create the base values, evenly spaced between given args
    let base = Array::linspace(args.time_start, args.time_end, args.num_samples);

    // create the sine wave using base
    let s_1 = base.mapv(|x| (2. * x).sin());

    // create the box wave using base
    let s_2 = base.mapv(|x| match (4. * x).sin() > 0. {
        true => 1.,
        false => -1.,
    });

    // join the two signals as columns
    let mut s = concatenate![
        Axis(1),
        s_1.clone().insert_axis(Axis(1)),
        s_2.clone().insert_axis(Axis(1))
    ];
    // add some random noise to the waves
    let mut rng = Xoshiro256Plus::seed_from_u64(1234);
    s += &Array::random_using((args.num_samples, 2), Uniform::new(-0.05, 0.05), &mut rng);

    // extract the signals from the columns
    Ok((
        base.to_vec(),
        s.column(0).to_vec(),
        s.column(1).to_vec(),
        None,
    ))
}

fn main() -> anyhow::Result<()> {
    let now = std::time::Instant::now();
    let cli = Cli::parse();

    // select different Mixing Matrices depending on the command
    let a = match cli.command {
        Commands::Example(_) => {
            // original mixing matrix (plot on slide 4)
            // array![[0.75, 1.5], [0.5, 2.]]
            // ambiguity plot example (plot on slide 9)
            array![[2., 1.5], [0.5, 0.75]]
        }
        Commands::Speech(_) => array![[2., 0.5], [0.5, 1.5]],
    };

    // get the signal data based on the command used
    let (base, s_1, s_2, audio_data) = match cli.command {
        Commands::Example(ref ex) => example(ex.clone())?,
        Commands::Speech(ref sp) => speech(sp.clone(), a.clone())?,
    };

    // convert vectors to ndarrays
    let (s_1, s_2) = (Array::from_iter(s_1), Array::from_iter(s_2));

    // join the signals as columns
    let s = concatenate![
        Axis(1),
        s_1.clone().insert_axis(Axis(1)),
        s_2.clone().insert_axis(Axis(1))
    ];

    // apply the mixing matrix
    let x = s.dot(&a.t());

    // Fitting the ICA model using the logcosh G Function with Alpha 1
    let ica = FastIca::params().gfunc(GFunc::Logcosh(1.0));
    let ica = ica.fit(&DatasetBase::from(x.view()))?;

    // reconstruct the original signals using FastICA
    let y = ica.predict(&x);

    // extract signals from matrix
    let (y_1, y_2) = (y.column(0).to_vec(), y.column(1).to_vec());

    // save signals for speech command
    if let Commands::Speech(ref sp) = cli.command {
        // get audio data
        let Some(audio) = audio_data else {
            unreachable!("always Some for Speech Commands")
        };

        // amplify signals to make them audible
        let mut speaker_1 = audio.clone();
        speaker_1.data = y_1.clone().iter().map(|e| (*e * 1e6) as i16).collect();
        let mut speaker_2 = audio;
        speaker_2.data = y_2.clone().iter().map(|e| (*e * 1e6) as i16).collect();

        // save to disk
        save_audio_to_file(sp.speaker_1.clone(), speaker_1, Some("ica"))?;
        save_audio_to_file(sp.speaker_2.clone(), speaker_2, Some("ica"))?;
    }

    let is_speech = match cli.command {
        Commands::Example(_) => false,
        Commands::Speech(_) => true,
    };

    // plot the data
    plot(PlotArgs {
        output: cli.output.to_str().context("invalid path")?.to_string(),
        res: cli.resolution,
        x: base,
        s_1: s.column(0).to_vec(),
        s_2: s.column(1).to_vec(),
        x_1: x.column(0).to_vec(),
        x_2: x.column(1).to_vec(),
        y_1,
        y_2,
        speech: is_speech,
    })?;

    println!("Runtime: {:?}", now.elapsed());

    Ok(())
}
