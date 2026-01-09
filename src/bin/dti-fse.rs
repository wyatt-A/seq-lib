use std::arch::naked_asm;
use std::io::Write;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read};
use std::path::{Path, PathBuf};
use std::process::exit;
use std::rc::Rc;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use seq_struct::waveform::Waveform;
use seq_lib::grad_pulses::{ramp_down, ramp_up, trapezoid};
use seq_lib::PulseSequence;
use seq_lib::rf_pulses::{hardpulse, hardpulse_composite};
use mrs_ppl::compile::{build_seq, compile_seq};
use seq_lib::q_calc::{binary_solve, calc_b_matrix, grad_solve, load_bvecs};
use array_lib::io_mrd::read_mrd;
use dft_lib::common::{FftDirection, NormalizationType};
use nalgebra::{sup, ComplexField, DMatrix, DVector};
use num_complex::Complex32;
use rayon::prelude::*;
use dft_lib::rs_fft;

// shorthand types
type GW = Rc<Waveform>;
type RF = Rc<RfPulse>;
type GS = Rc<EventControl<FieldGrad>>;
type RFP = Rc<EventControl<f64>>;
type RFPhase = Rc<EventControl<Angle>>;
type Lookup = Rc<LUT>;

// loop names
const VIEW:&str = "view";
const ECHO:&str = "echo";
const EXPERIMENT:&str = "experiment";

// rf pulses
const EXC:&str = "exc_pulse";
const REF:&str = "ref_pulse";
const REFT:&str = "reft_pulse";

// diffusion pulses
const DIFF1:&str = "diff_1";
const DIFF2:&str = "diff_2";

// first phase encode, readout, acq
const PE: &str = "pe";
const RE: &str = "re";
const RO_RU:&str = "ro_ru";
const RO_RD:&str = "ro_rd";
const ACQ: &str = "acq";

// echo-loop phase encode, rewind, crusher, readout, acq
const ROT_RU: &str = "rot_ru";
const ROT_RD: &str = "rot_rd";
const ACQT: &str = "acqt";
const PETL: &str = "pet_left";
const PETR: &str = "pet_right";



struct DTIFse {
    mode: Mode,
    bandwidth_khz: f64,
    n_samples: usize,
    fov_x_mm: f64,
    fov_y_mm: f64,
    fov_z_mm: f64,
    rf_duration_us: usize,
    diffusion_ramp_time_us: usize,
    imaging_ramp_time_us: usize,
    phase_enc_dur_ms: f64,
    little_delta_ms: f64,
    g_limit_tpm: f64,
    n_echoes: usize,
    cs_table: PathBuf,
    bvec_table: PathBuf,
    target_bvalues: Vec<f64>,
}

impl Default for DTIFse {
    fn default() -> Self {
        DTIFse {
            bandwidth_khz: 100.,
            n_samples: 512,
            fov_x_mm: 25.6,
            fov_y_mm: 12.8,
            fov_z_mm: 12.8,
            rf_duration_us: 100,
            diffusion_ramp_time_us: 500,
            imaging_ramp_time_us: 200,
            phase_enc_dur_ms: 0.5,
            little_delta_ms: 1.7,
            g_limit_tpm: 2.3,
            n_echoes: 6,
            cs_table: PathBuf::from(r"stream_CS256_8x_pa18_pb54"),
            bvec_table: PathBuf::from(r"../26.wang.06/bvecs.txt"),
            target_bvalues: vec![1_000.,3_000.,5_000.,8_000.,10_000.,12_000.],
            mode: Mode::Tune {n: 1_000},
        }
    }
}

struct Waveforms {
    ramp_up: GW,
    ramp_down: GW,
    diffusion: GW,
    phase_enc: GW,
    rf90: RF,
    rf180: RF,
    imaging_ramp_time:Time,
    pe_dur:Time,
    rf_dur:Time,
}

impl Waveforms {
    fn build(params:&DTIFse) -> Waveforms {

        let imaging_ramp_time = Time::us(params.imaging_ramp_time_us);
        let diffusion_ramp_time = Time::us(params.diffusion_ramp_time_us);
        let diffusion_pulse_plat_dur = Time::ms(params.little_delta_ms);
        let pe_dur = Time::ms(params.phase_enc_dur_ms);
        let rf_dur = Time::us(params.rf_duration_us);

        // waveform time-base
        let grad_dt = Time::us(2);
        let rf_dt = Time::us(2);

        // ramps for readout window
        let ramp_up = ramp_up(imaging_ramp_time,grad_dt).to_shared();
        let ramp_down = ramp_down(imaging_ramp_time,grad_dt).to_shared();

        // trapezoid pulse for diffusion encoding
        let diffusion = trapezoid(diffusion_ramp_time,diffusion_pulse_plat_dur,grad_dt).to_shared();

        // trapezoid pulse for phase encoding
        let phase_enc = trapezoid(imaging_ramp_time,pe_dur,grad_dt).to_shared();

        // rf pulses for excitation and refocusing
        let rf90 = hardpulse(rf_dur,rf_dt,Nuc1H).to_shared();
        let rf180 = hardpulse_composite(rf_dur.scale(2),rf_dt,Nuc1H).to_shared();

        Waveforms {
            ramp_up,
            ramp_down,
            diffusion,
            phase_enc,
            rf90,
            rf180,
            imaging_ramp_time,
            pe_dur,
            rf_dur,
        }

    }
}

struct EventControllers {
    ex_pow:RFP,
    ref_pow:RFP,
    ref_phase:RFPhase,
    readout: GS,
    diffusion_x: GS,
    diffusion_y: GS,
    diffusion_z: GS,
    phase_enc_y: GS,
    phase_enc_z: GS,
    phase_rewind_y: GS,
    phase_rewind_z: GS,

    // phase_enc2_y: GS,
    // phase_enc2_z: GS,
    // phase_rewind2_y: GS,
    // phase_rewind2_z: GS,

    prephase_x: GS,
    crush_left_x: GS,
    crush_right_x: GS,
}

#[derive(Clone)]
enum Mode {
    /// used to acquire image support region for each echo to measure phase errors.
    /// The y-z FOV should match the size of the object to gather accurate phase information due to very low matrix size in y-z
    /// "r" is the y-z sampling radius e.g. r=2 results in 25 phase encodes total (2r + 1)^2
    Measure{r:i32, fov_y:f64, fov_z:f64, n_dummies:usize, g_vectors: Vec<[FieldGrad;3]>},
    /// used to acquire the data set
    Acq{ n_dummies:usize, g_vectors: Vec<[FieldGrad;3]>},
    /// used to tune the sequence parameters (Rf power and crushers/balancers) where n is the number
    /// of reps
    Tune{n:usize},
}

impl EventControllers {
    fn build(params:&DTIFse, w:&Waveforms) -> EventControllers {

        let dwell_time = Freq::khz(params.bandwidth_khz).inv();
        let acq_time = dwell_time.scale(params.n_samples);
        let gro = FieldGrad::from_fov(Length::mm(params.fov_x_mm), dwell_time, Nuc1H);
        let readout = EventControl::<FieldGrad>::new().with_constant_grad(gro).to_shared();

        // build the diffusion event controllers based on the pulse sequence mode
        let (diffusion_x,diffusion_y,diffusion_z) = match &params.mode {
            // "tune" offers full adjustable control of each diffusion gradient
            Mode::Tune{..} => {
                (
                    EventControl::<FieldGrad>::new().with_adj("diff_x").to_shared(),
                    EventControl::<FieldGrad>::new().with_adj("diff_y").to_shared(),
                    EventControl::<FieldGrad>::new().with_adj("diff_z").to_shared()
                )
            }
            // "measure" and "acq" use a pre-determined gradient table to control gradient strengths
            Mode::Measure{ g_vectors, .. } | Mode::Acq { g_vectors, .. } => {

                // determine g_max from the table
                let g_max = g_vectors.iter()
                    .map(|g| (g[0].si().powi(2) + g[1].si().powi(2) + g[2].si().powi(2)).sqrt())
                    .max_by(|a,b|a.partial_cmp(&b).unwrap()).unwrap();

                // discretize the diffusion gradient steps based on g_max
                let diff_grad_resolution = FieldGrad::tesla_per_meter(g_max / i16::MAX as f64); // grad strength per step

                // build lookup tables for diffusion gradient scaling (similar to spatial encoding)
                let mut gx_lut = vec![];
                let mut gy_lut = vec![];
                let mut gz_lut = vec![];
                for g_vec in g_vectors {
                    let step_x = (g_vec[0].si() / diff_grad_resolution.si()).floor() as i32;
                    let step_y = (g_vec[1].si() / diff_grad_resolution.si()).floor() as i32;
                    let step_z = (g_vec[2].si() / diff_grad_resolution.si()).floor() as i32;
                    gx_lut.push(step_x);
                    gy_lut.push(step_y);
                    gz_lut.push(step_z);
                }
                let gx_lut = LUT::new("diff_gx",&gx_lut).to_shared();
                let gy_lut = LUT::new("diff_gy",&gy_lut).to_shared();
                let gz_lut = LUT::new("diff_gz",&gz_lut).to_shared();
                (
                    EventControl::<FieldGrad>::new().with_source_loop(EXPERIMENT)
                        .with_lut(&gx_lut).with_grad_scale(diff_grad_resolution).to_shared(),
                    EventControl::<FieldGrad>::new().with_source_loop(EXPERIMENT)
                        .with_lut(&gy_lut).with_grad_scale(diff_grad_resolution).to_shared(),
                    EventControl::<FieldGrad>::new().with_source_loop(EXPERIMENT)
                        .with_lut(&gz_lut).with_grad_scale(diff_grad_resolution).to_shared(),
                )
            }
        };

        let pe_time = Time::try_from(w.imaging_ramp_time + w.pe_dur).unwrap();
        let ro_time = Time::try_from(w.imaging_ramp_time + acq_time).unwrap();

        let (phase_enc_y,phase_enc_z,phase_rewind_y,phase_rewind_z) = match params.mode {
            Mode::Measure{r,fov_y,fov_z,n_dummies,..} => {

                let phase_step_y = FieldGrad::from_fov(Length::mm(fov_y), pe_time, Nuc1H);
                let phase_step_z = FieldGrad::from_fov(Length::mm(fov_z), pe_time, Nuc1H);

                // populate with dummy scans
                let mut y_steps = vec![0;n_dummies];
                let mut z_steps = vec![0;n_dummies];

                // populate with steps from -r to r
                for y in -r..=r {
                    for z in -r..=r {
                        y_steps.push(y);
                        z_steps.push(z);
                    }
                }
                let lut_y = LUT::new("lut_y",&y_steps).to_shared();
                let lut_z = LUT::new("lut_z",&z_steps).to_shared();
                let phase_enc_y = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_y).with_grad_scale(phase_step_y).to_shared();
                let phase_enc_z = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_z).with_grad_scale(phase_step_z).to_shared();
                // inverse of the phase encodes
                let phase_rewind_y = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_y).with_grad_scale(phase_step_y.scale(-1)).to_shared();
                let phase_rewind_z = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_z).with_grad_scale(phase_step_z.scale(-1)).to_shared();
                (phase_enc_y,phase_enc_z,phase_rewind_y,phase_rewind_z)
            }
            Mode::Acq{n_dummies, ..} => {

                let phase_step_y = FieldGrad::from_fov(Length::mm(params.fov_y_mm), pe_time, Nuc1H);
                let phase_step_z = FieldGrad::from_fov(Length::mm(params.fov_z_mm), pe_time, Nuc1H);

                // populate with dummy scans
                let mut y_steps = vec![0;n_dummies];
                let mut z_steps = vec![0;n_dummies];

                // load steps from CS table
                let mut f = File::open(&params.cs_table).unwrap();
                let mut entries = String::new();
                f.read_to_string(&mut entries).unwrap();
                let entries:Vec<i32> = entries.lines().map(|s| s.parse::<i32>().unwrap()).collect();

                let mut pairs:Vec<_> = entries.chunks_exact(2).map(|coord| {
                    let r:i32 = coord.iter().map(|x|x * x).sum();
                    (coord,r)
                } ).collect();

                // coordinate pairs sorted from least to greatest in terms of |k|
                pairs.sort_by_key(|(_,r)|*r);
                let pairs:Vec<_> = pairs.into_iter().map(|(c,_)| c).collect();
                pairs.iter().for_each(|coords| {
                    y_steps.push(coords[0]);
                    z_steps.push(coords[1]);
                });
                let lut_y = LUT::new("lut_y",&y_steps).to_shared();
                let lut_z = LUT::new("lut_z",&z_steps).to_shared();
                let phase_enc_y = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_y).with_grad_scale(phase_step_y).to_shared();
                let phase_enc_z = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_z).with_grad_scale(phase_step_z).to_shared();
                // inverse of the phase encodes
                let phase_rewind_y = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_y).with_grad_scale(phase_step_y.scale(-1)).to_shared();
                let phase_rewind_z = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_z).with_grad_scale(phase_step_z.scale(-1)).to_shared();
                (phase_enc_y,phase_enc_z,phase_rewind_y,phase_rewind_z)
            },
            Mode::Tune{..} => {
                let phase_enc_y = EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0)).to_shared();
                let phase_enc_z = phase_enc_y.clone();
                let phase_rewind_y = phase_enc_y.clone();
                let phase_rewind_z = phase_enc_y.clone();
                (phase_enc_y,phase_enc_z,phase_rewind_y,phase_rewind_z)
            }
        };

        let pe_ro_ratio = (ro_time / pe_time).si();
        let gp = gro.scale( - pe_ro_ratio / 2.);
        let prephase_x = EventControl::<FieldGrad>::new().with_constant_grad(gp).with_adj("prephase_x").to_shared();
        let crush_left_x = EventControl::<FieldGrad>::new().with_adj("crush_left").to_shared();
        let crush_right_x = EventControl::<FieldGrad>::new().with_adj("crush_right").to_shared();
        let ex_pow = EventControl::<f64>::new().with_adj("rf90").to_shared();
        let ref_pow = EventControl::<f64>::new().with_adj("rf180").to_shared();
        let ref_phase = EventControl::<Angle>::new().with_constant(Angle::deg(90)).to_shared();

        EventControllers {
            ex_pow,
            ref_pow,
            ref_phase,
            readout,
            diffusion_x,
            diffusion_y,
            diffusion_z,
            phase_enc_y,
            phase_enc_z,
            phase_rewind_y,
            phase_rewind_z,
            prephase_x,
            crush_left_x,
            crush_right_x,
        }

    }
}

struct Events {
    exc: RfEvent,
    refoc: RfEvent,
    refoct: RfEvent,
    diff1: GradEvent,
    diff2: GradEvent,
    pe: GradEvent,
    re: GradEvent,
    ro_ru: GradEvent,
    ro_rd: GradEvent,
    acq: ACQEvent,
    rot_ru: GradEvent,
    rot_rd: GradEvent,
    acqt: ACQEvent,
    petl: GradEvent,
    petr: GradEvent,
}

impl Events {
    fn build(params:&DTIFse,w:&Waveforms,e:&EventControllers) -> Events {

        let exc = RfEvent::new(EXC,&w.rf90,&e.ex_pow);
        let refoc = RfEvent::new(REF,&w.rf180,&e.ref_pow).with_phase(&e.ref_phase);
        let refoct = RfEvent::new(REFT, &w.rf180,&e.ref_pow).with_phase(&e.ref_phase);

        let diff1 = GradEvent::new(DIFF1)
            .with_x(&w.diffusion).with_y(&w.diffusion).with_z(&w.diffusion)
            .with_strength_x(&e.diffusion_x).with_strength_y(&e.diffusion_y).with_strength_z(&e.diffusion_z);

        let diff2 = GradEvent::new(DIFF2)
            .with_x(&w.diffusion).with_y(&w.diffusion).with_z(&w.diffusion)
            .with_strength_x(&e.diffusion_x).with_strength_y(&e.diffusion_y).with_strength_z(&e.diffusion_z);

        let pe = GradEvent::new(PE).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.prephase_x).with_strength_y(&e.phase_enc_y).with_strength_z(&e.phase_enc_z);

        let re = GradEvent::new(RE).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.crush_right_x).with_strength_y(&e.phase_rewind_y).with_strength_z(&e.phase_rewind_z);

        let petl = GradEvent::new(PETL).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.crush_left_x).with_strength_y(&e.phase_enc_y).with_strength_z(&e.phase_enc_z);

        let petr = GradEvent::new(PETR).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.crush_right_x).with_strength_y(&e.phase_rewind_y).with_strength_z(&e.phase_rewind_z);

        let ro_ru = GradEvent::new(RO_RU).with_x(&w.ramp_up).with_strength_x(&e.readout);
        let ro_rd = GradEvent::new(RO_RD).with_x(&w.ramp_down).with_strength_x(&e.readout);

        let rot_ru = GradEvent::new(ROT_RU).with_x(&w.ramp_up).with_strength_x(&e.readout);
        let rot_rd = GradEvent::new(ROT_RD).with_x(&w.ramp_down).with_strength_x(&e.readout);

        let acq = ACQEvent::new(ACQ,params.n_samples,Freq::khz(params.bandwidth_khz).inv());
        let acqt = ACQEvent::new(ACQT,params.n_samples,Freq::khz(params.bandwidth_khz).inv());

        Events {
            exc,
            refoc,
            refoct,
            diff1,
            diff2,
            pe,
            re,
            ro_ru,
            ro_rd,
            acq,
            rot_ru,
            rot_rd,
            acqt,
            petl,
            petr,
        }
    }
}

impl PulseSequence for DTIFse {
    fn compile(&self) -> SeqLoop {

        let w = Waveforms::build(self);
        let e = EventControllers::build(self,&w);
        let events = Events::build(self,&w,&e);

        let n_views = match self.mode {
            Mode::Measure { .. } | Mode::Acq{..} => {
                let ny = e.phase_enc_y.lut.as_ref().expect("expected phase_enc_y to contain a LUT").len();
                let nz = e.phase_enc_z.lut.as_ref().expect("expected phase_enc_z to contain a LUT").len();
                assert_eq!(ny,nz);
                ny
            }
            Mode::Tune {n} => {
                n
            }
        };

        // view loop events
        let mut vl = SeqLoop::new_main(VIEW,n_views);
        vl.add_event(events.exc).unwrap();
        vl.add_event(events.diff1).unwrap();
        vl.add_event(events.refoc).unwrap();
        vl.add_event(events.diff2).unwrap();
        vl.add_event(events.pe).unwrap();
        vl.add_event(events.ro_ru).unwrap();
        vl.add_event(events.acq).unwrap();
        vl.add_event(events.ro_rd).unwrap();
        vl.add_event(events.re).unwrap();

        vl.set_time_span(EXC,DIFF1,100,0,Time::us(100)).unwrap();
        vl.set_time_span(REF,DIFF2,100,0,Time::us(100)).unwrap();
        vl.set_time_span(DIFF2,PE,100,0,Time::us(500)).unwrap();
        vl.set_time_span(PE,RO_RU,100,0,Time::us(100)).unwrap();
        vl.set_min_time_span(RO_RU,ACQ,100,0,Time::us(100)).unwrap();
        vl.set_min_time_span(ACQ,RO_RD,100,0,Time::us(100)).unwrap();
        vl.set_time_span(RO_RD,RE,100,0,Time::us(100)).unwrap();

        let tau = vl.get_time_span(REF,ACQ,50,50).unwrap();
        vl.set_time_span(EXC,REF,100,0,tau).expect("failed to set delay");

        // echo loop events
        let mut el = SeqLoop::new(ECHO,self.n_echoes - 1);
        el.add_event(events.refoct).unwrap();
        el.add_event(events.petl).unwrap();
        el.add_event(events.rot_ru).unwrap();
        el.add_event(events.acqt).unwrap();
        el.add_event(events.rot_rd).unwrap();
        el.add_event(events.petr).unwrap();

        let echo_adj_us = 1500; // this needs to be balanced with the el pre-calc time
        let el_pre_calc_time_us = 2000;

        el.set_time_span(REFT,PETL,100,0,Time::us(echo_adj_us)).unwrap();
        el.set_time_span(PETL,ROT_RU,100,0,Time::us(200)).unwrap();
        el.set_min_time_span(ROT_RU,ACQT,100,0,Time::us(200)).unwrap();
        el.set_min_time_span(ACQT,ROT_RD,100,0,Time::us(100)).unwrap();
        el.set_time_span(ROT_RD,PETR,100,0,Time::us(200)).unwrap();

        let tau2 = el.get_time_span(REFT,ACQT,50,50).unwrap();
        let echo_spacing = tau2.scale(2);

        el.set_pre_calc(Time::us(el_pre_calc_time_us));
        el.set_rep_time(echo_spacing).expect("failed to set echo spacing");
        el.set_averages(0);

        vl.add_loop(el).unwrap();
        vl.set_time_span(RE,REFT,100,0,Time::us(echo_adj_us)).unwrap();
        vl.set_pre_calc(Time::ms(3));
        vl.set_rep_time(Time::ms(100)).unwrap();

        match self.mode {
            Mode::Tune { .. } => vl, // return just the view loop
            Mode::Acq { .. } | Mode::Measure { .. } => { // return the view loop inside the experiment loop for diffusion encoding
                let n_dx = e.diffusion_x.lut().unwrap().len();
                let n_dy = e.diffusion_y.lut().unwrap().len();
                let n_dz = e.diffusion_z.lut().unwrap().len();
                assert_eq!(n_dx, n_dy);
                assert_eq!(n_dx, n_dz);
                let mut el = SeqLoop::new(EXPERIMENT,n_dx);
                el.add_loop(vl).unwrap();
                el.set_pre_calc(Time::ms(1));
                el
            },
        }
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("crush_left".to_string(), 0.05);
        state.insert("crush_right".to_string(), 0.07);
        state.insert("prephase_x".to_string(), 0.);
        state.insert("rf90".to_string(), 1.);
        state.insert("rf180".to_string(), 2.);
        match self.mode {
            Mode::Measure { .. } => {}
            Mode::Acq { .. } => {}
            Mode::Tune { .. } => {
                state.insert("diff_x".to_string(), 0.);
                state.insert("diff_y".to_string(), 0.);
                state.insert("diff_z".to_string(), 0.);
            }
        }
        state
    }
}


fn main() {


    //find_phase_shifts("/Users/Wyatt/measure.MRD");

    exit(0);

    let out_dir = r"/Users/wyatt/seq-lib/test_out";

    // compile with default settings
    let mut params = DTIFse::default();

    // mode for single rep through k0
    params.mode = Mode::Tune {n:1};
    let s = params.compile();
    let mut adj = params.adjustment_state();

    // build list of echo times to evaluate each b-matrix
    let mut t_echoes = vec![];
    // get the center of the first echo from the center of ACQ
    t_echoes.extend(s.find_occurrences(ACQ,50));
    t_echoes.extend(s.find_occurrences(ACQT,50));

    // print echo times
    for (i,echo) in t_echoes.iter().enumerate() {
        println!("echo {}: {} ms",i+1, echo.as_ms());
    }

    let grad_soltns:Vec<_> = params.target_bvalues.iter().map(|&target_bval|{
        grad_solve(
            &params,
            "diff_x",
            target_bval,
            FieldGrad::tesla_per_meter(0), // lower limit
            FieldGrad::tesla_per_meter(params.g_limit_tpm), // upper limit
            &[REF,REFT],
            t_echoes[0]
        )
    }).collect();

    for (soltn,target_bval) in grad_soltns.iter().zip(&params.target_bvalues) {
        println!("solved for gradient strength of {} mT/m for target b-value {target_bval} s/mm^2", 1000. * soltn.si());
    }

    // load b-vector table and scale the diffusion gradient strengths by each vector component
    let (shell_idx, b_vectors) = load_bvecs(&params.bvec_table);
    let n_shells = *shell_idx.iter().max().unwrap() + 1;
    assert_eq!(n_shells,params.target_bvalues.len(),"expect {} target b-vals based on max shell index",n_shells);
    let g_vectors:Vec<_> = b_vectors.iter().zip(shell_idx).map(|(bv,s_idx)|{
        [
            grad_soltns[s_idx].scale(bv[0]),
            grad_soltns[s_idx].scale(bv[1]),
            grad_soltns[s_idx].scale(bv[2]),
        ]
    }).collect();

    // calculate the b-matrix for each b-vector and write to a file
    use std::fmt::Write;
    let mut b_info = String::new();
    writeln!(&mut b_info,"idx\tgmax(T/m)\ttrace(s/mm^2)\tbxx\tbyy\tbzz\tbxy\tbxz\tbyz").unwrap();
    let echo_idx = 0;
    for (v, g_vector) in g_vectors.iter().enumerate() {
        // set diffusion parameters to simulate b-values
        *adj.get_mut("diff_x").unwrap() = g_vector[0].si();
        *adj.get_mut("diff_y").unwrap() = g_vector[1].si();
        *adj.get_mut("diff_z").unwrap() = g_vector[2].si();
        let g_max = g_vector[0].si().abs().max(g_vector[1].si().abs()).max(g_vector[2].si().abs());
        //evaluate b-matrix for each g-vector and write to string
        let b_mat = calc_b_matrix(&params, &adj, &[REF,REFT], t_echoes[echo_idx], Nuc1H);
        writeln!(&mut b_info,
                 "{v}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                 g_max.si(), b_mat.trace(), b_mat.bxx, b_mat.byy, b_mat.bzz, b_mat.bxy, b_mat.bxz, b_mat.byz
        ).unwrap();
    }
    File::create(Path::new(out_dir).join("b-info.txt")).unwrap().write_all(b_info.as_bytes()).unwrap();

    //params.mode = Mode::Acq { grad_table: grad_tab};
    params.mode = Mode::Measure {r:3,n_dummies:5,fov_y:params.fov_y_mm,fov_z:params.fov_z_mm, g_vectors: g_vectors.clone()};

    compile_seq(&params.compile(),out_dir,"seq",false);

    params.mode = Mode::Tune {n:1};
    let mut adj = params.adjustment_state();
    *adj.get_mut("diff_x").unwrap() = 1.;
    params.render_to_file(&adj,Path::new(out_dir).join("dti_fse"));

}