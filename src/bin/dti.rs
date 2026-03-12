use std::collections::{HashMap, HashSet};
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use clap::Parser;
use headfile::Headfile;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::compile::Seq;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use serde::{Deserialize, Serialize};
use seq_lib::defs::{RFPhase, EXPERIMENT, GS, GW, RFP, VIEW};
use seq_lib::grad_pulses::{ramp_down, ramp_up, trapezoid};
use seq_lib::{Args, PulseSequence, ToHeadfile, TOML};
use seq_lib::q_calc::{calc_b_matrix, grad_solve, load_bvecs};
use seq_lib::rf_pulses::{hardpulse, hardpulse_composite};

fn main() {
    let args = Args::parse();
    args.run::<DTI>();
}

const SEQ_NAME: &str = "dti";
#[derive(Debug,Clone,Serialize,Deserialize)]
struct DTI {
    /// intended to compile the sequence with no phase encoding and user-adjustable diffusion gradients
    setup_mode: bool,
    /// compiles the sequence for use with display mode and with the MRS sequence simulator
    sim_mode: bool,
    /// builds the sequence with a single repetition and adjustable diffusion gradients to calculate gradient strengths for target b-values
    solve_mode: bool,
    /// field of view in millimeters
    fov_mm: [f64;3],
    /// matrix size for sampling
    matrix_size: [usize;3],
    /// spectral width of the ADC
    spectral_width_khz: f64,
    /// duration of rf excitation pulse in microseconds
    rf_duration_us: usize,
    /// ramp time for trapezoid diffusion gradients in microseconds
    diffusion_ramp_time_us: usize,
    /// ramp time for imaging gradients (phase encoding and readout) in microseconds
    imaging_ramp_time_us: usize,
    /// duration of phase encoding (not including ramp time) in milliseconds
    phase_enc_dur_ms: f64,
    /// diffusion gradient duration in milliseconds
    little_delta_ms: f64,
    /// diffusion gradient separation in milliseconds
    big_delta_ms: Option<f64>,
    /// repetition time in milliseconds
    rep_time_ms: f64,
    /// user-defined upper limit for diffusion gradients in tesla per meter
    g_limit_tpm: f64,
    /// delay after the second diffusion pulse, before phase encoding gradient in milliseconds
    post_diff2_delay_ms: f64,
    /// path to cs table (ascii file with one integer per row in the order py1, pz1, py2, pz2, ... etc.)
    cs_table: PathBuf,
    /// path to b-vector table (ascii file with 4 entries per row: shell_idx, ux, uy, uz)
    bvec_table: PathBuf,
    /// list of target b-values for each shell index specified by the bvec table
    target_bvalues: Vec<f64>,
    /// calculated echo time based on gradient parameters
    echo_time_ms: Option<f64>,
    /// list of shell indices determined by loaded b-vector table
    shell_idx: Option<Vec<usize>>,
    /// number of phase encodings (later determined by the loaded pe table)
    n_phase: Option<usize>,
    /// max calculated diffusion gradient strength
    g_max_tpm: Option<Vec<f64>>,
    /// loaded b-vectors
    b_vecs: Option<Vec<[f64;3]>>,
    /// calculated diffusion gradient vectors in tesla per meter
    g_vecs_tpm: Option<Vec<[f64;3]>>,
    /// calculated b-matrix trace in seconds per square meter for each diffusion encoding
    b_trace:Option<Vec<f64>>,
    /// calculate b-x from b-matrix in seconds per square meter
    bxx:Option<Vec<f64>>,
    /// calculate b-y from b-matrix in seconds per square meter
    byy:Option<Vec<f64>>,
    /// calculate b-z from b-matrix in seconds per square meter
    bzz:Option<Vec<f64>>,
    /// calculate b-x-y cross term from b-matrix in seconds per square meter
    bxy:Option<Vec<f64>>,
    /// calculate b-x-z cross term from b-matrix in seconds per square meter
    bxz:Option<Vec<f64>>,
    /// calculate b-y-z cross term from b-matrix in seconds per square meter
    byz:Option<Vec<f64>>,
}

impl Default for DTI {
    fn default() -> Self {
        DTI {
            setup_mode: false,
            sim_mode: false,
            solve_mode: false,
            spectral_width_khz: 100.,
            fov_mm: [25.6, 12.8,12.8],
            matrix_size: [512, 256, 256],
            n_phase: None,
            rf_duration_us: 100,
            diffusion_ramp_time_us: 500,
            imaging_ramp_time_us: 200,
            phase_enc_dur_ms: 0.3,
            little_delta_ms: 1.7,
            big_delta_ms: None,
            rep_time_ms: 100.,
            g_limit_tpm: 2.3,
            post_diff2_delay_ms: 1.,
            cs_table: PathBuf::from("/Users/Wyatt/26.wang.06/stream_CS256_8x_pa18_pb54"),
            bvec_table: PathBuf::from("/Users/Wyatt/26.wang.06/bvecs.txt"),
            target_bvalues: vec![1_000.,3_000.,5_000.,8_000.,10_000.,12_000.],
            g_max_tpm: None,
            b_vecs: None,
            g_vecs_tpm: None,
            shell_idx: None,
            echo_time_ms: None,
            b_trace: None,
            bxx: None,
            byy: None,
            bzz: None,
            bxy: None,
            bxz: None,
            byz: None,
        }
    }
}

impl DTI {

    /// reads the b-vec table, checks validity, and populates b_vec and shell_idx parameters
    fn read_bvecs(&mut self) {
        let (shell_idx,b_vecs) = load_bvecs(&self.bvec_table);
        let u:HashSet<usize> = shell_idx.iter().cloned().collect();
        assert_eq!(u.len(), self.target_bvalues.len(),"number of target b-values does not match b-vector length");
        self.b_vecs = Some(b_vecs);
        self.shell_idx = Some(shell_idx);
    }

    /// reads phase encoding table, returning coordinates and populating the n_phase parameter
    fn read_pe_table(&mut self) -> Vec<[i32;2]> {
        let s = read_to_string(&self.cs_table).expect("invalid cs table file");
        let cs_samples:Vec<i32> = s.lines().map(|idx| idx.parse::<i32>()
            .expect("failed to parse cs table")).collect();
        let coords:Vec<_> = cs_samples.chunks_exact(2).map(|pair| [pair[0],pair[1]]).collect();
        self.n_phase = Some(coords.len());
        coords
    }

    /// build the view loop in one of two modes. Solve mode renders a single view with a variable
    /// x diffusion gradient to solve for gradient strengths
    fn view_loop(&mut self) -> SeqLoop {

        // single view for solve mode and sim mode
        let n_views = if self.solve_mode {
            1
        }else if self.sim_mode {
            1
        }else if self.setup_mode {
            10000 // arbitrary number of views for setup mode (no phase encoding)
        } else {
            self.read_pe_table(); // determines the number of views from the cs table
            self.n_phase.unwrap()
        };

        // build the sequence events for current state
        let events = Events::build(self);

        // build the view loop
        let mut view_loop = SeqLoop::new_main(VIEW, n_views);
        // add events
        view_loop.add_event(events.exc).unwrap();
        view_loop.add_event(events.diff1).unwrap();
        view_loop.add_event(events.refoc).unwrap();
        view_loop.add_event(events.diff2).unwrap();
        view_loop.add_event(events.pe).unwrap();
        view_loop.add_event(events.ro_ru).unwrap();
        view_loop.add_event(events.acq).unwrap();
        view_loop.add_event(events.ro_rd).unwrap();
        view_loop.add_event(events.spoil).unwrap();

        // pre-calc time to calculate view-dependent parameters
        view_loop.set_pre_calc(Time::ms(4));

        // delay after rf pulses, before diffusion gradient start
        let post_rf_del = Time::us(50);

        // place rf pulses relative to diffusion gradients
        view_loop.set_min_time_span(Events::exc(), Events::diff1(), 100, 0, post_rf_del).unwrap();
        view_loop.set_min_time_span(Events::refoc(), Events::diff2(), 100, 0, post_rf_del).unwrap();

        // // set diffusion pulse separation (big delta)
        // let big_delta = Time::ms(self.big_delta_ms);
        // if let Ok(delta) = view_loop.set_min_time_span(Events::diff1(), Events::diff2(), 0, 0, big_delta) {
        //     println!("set big delta to {} ms",delta.as_ms());
        //     self.big_delta_ms = delta.as_ms();
        // }else {
        //     panic!("failed to set big delta ms");
        // }

        // set delay between second diffusion pulse and phase encoding
        let post_diff2_del = Time::ms(self.post_diff2_delay_ms);
        view_loop.set_time_span(Events::diff2(), Events::pe(), 100, 0, post_diff2_del).unwrap();

        // set small delay after phase encoding and readout
        let post_pe_del = Time::us(50);
        view_loop.set_time_span(Events::pe(), Events::ro_ru(), 100, 0, post_pe_del).unwrap();

        // set up sample acquisition
        view_loop.set_min_time_span(Events::ro_ru(), Events::acq(), 100, 0, Time::us(50)).unwrap();
        view_loop.set_min_time_span(Events::acq(), Events::ro_rd(), 100, 0, Time::us(50)).unwrap();
        view_loop.set_time_span(Events::ro_rd(), Events::spoil(), 100, 0, Time::us(50)).unwrap();




        // determine tau based on refocusing pulse and center of acq
        let tau = view_loop.get_time_span(Events::refoc(),Events::acq(),50,50).expect("failed to get tau");
        view_loop.set_time_span(Events::exc(),Events::refoc(),50,50, tau).expect("failed to set tau");

        let big_delta = view_loop.get_time_span(Events::diff1(),Events::diff2(),0,0).expect("failed to get big delta");
        self.big_delta_ms = Some(big_delta.as_ms());

        // determine the echo time based on event locations
        let te = view_loop.get_time_span(Events::exc(), Events::acq(), 50, 50).expect("failed to get echo time");
        println!("min echo time: {} ms",te.as_ms());
        self.echo_time_ms = Some(te.as_ms());

        // set repetition time for view loop
        view_loop.set_rep_time(Time::ms(self.rep_time_ms)).expect("failed to set rep time");

        // return the view loop
        view_loop
    }

    /// calculates max gradient strength for each b-vector and stores it as an internal parameter
    fn calc_gmax(&mut self,view_loop:&SeqLoop, adj:&mut HashMap<String,f64>) {
        let g_max_tpm:Vec<_> = self.target_bvalues.iter().map(|&target_bval|{
            grad_solve(
                view_loop,
                adj.clone(),
                EventControllers::diff_x(),
                target_bval,
                FieldGrad::tesla_per_meter(0), // lower limit
                FieldGrad::tesla_per_meter(self.g_limit_tpm), // upper limit
                &[Events::refoc()],
                Time::ms(self.echo_time_ms.unwrap()),
            ).si()
        }).collect();
        self.g_max_tpm = Some(g_max_tpm);
    }

    /// calculates the gradient vectors based on the b-vec table and g_max parameter
    fn calc_g_vectors(&mut self) {
        let mut g_vecs_tpm = vec![];
        let shell_idx = self.shell_idx.as_ref().unwrap();
        self.b_vecs.as_ref().unwrap().iter().enumerate().for_each(|(i,b)|{
            let ux = b[0];
            let uy = b[1];
            let uz = b[2];
            let g_tpm = self.g_max_tpm.as_ref().unwrap()[shell_idx[i]];
            // scale by b-vector
            let gx = FieldGrad::tesla_per_meter(g_tpm).scale(ux);
            let gy = FieldGrad::tesla_per_meter(g_tpm).scale(uy);
            let gz = FieldGrad::tesla_per_meter(g_tpm).scale(uz);
            g_vecs_tpm.push([gx.si(),gy.si(),gz.si()]);
        });
        self.g_vecs_tpm = Some(g_vecs_tpm);
    }

    /// calculates the b-matrix entries based on g_max and b-vectors and stores the parameters. B values
    /// are stored in si units (s/m^2)
    fn calc_b_matrix(&mut self, view_loop:&SeqLoop, adj:&mut HashMap<String,f64>) {

        // b-matrix entries
        let mut b_trace = vec![];
        let mut bxx = vec![];
        let mut byy = vec![];
        let mut bzz = vec![];
        let mut bxy = vec![];
        let mut bxz = vec![];
        let mut byz = vec![];

        let te = Time::ms(self.echo_time_ms.unwrap());
        for g_vec_tpm in self.g_vecs_tpm.as_ref().unwrap().iter() {
            // set diffusion gradient strengths to calculate the b-matrix for each vector
            *adj.get_mut(EventControllers::diff_x()).unwrap() = g_vec_tpm[0];
            *adj.get_mut(EventControllers::diff_y()).unwrap() = g_vec_tpm[1];
            *adj.get_mut(EventControllers::diff_z()).unwrap() = g_vec_tpm[2];
            //let g_max = g_vec_tpm[0].abs().max(g_vec_tpm[1].abs()).max(g_vec_tpm[2].abs());
            let b_mat = calc_b_matrix(view_loop, &adj, &[Events::refoc()], te, Nuc1H);
            b_trace.push(b_mat.trace());
            bxx.push(b_mat.bxx);
            byy.push(b_mat.byy);
            bzz.push(b_mat.bzz);
            bxy.push(b_mat.bxy);
            bxz.push(b_mat.bxz);
            byz.push(b_mat.byz);
        }

        // store b-matrix
        self.b_trace = Some(b_trace);
        self.bxx = Some(bxx);
        self.byy = Some(byy);
        self.bzz = Some(bzz);
        self.bxy = Some(bxy);
        self.bxz = Some(bxz);
        self.byz = Some(byz);

    }

}

impl ToHeadfile for DTI {
    fn headfile(&self) -> Headfile {


        let te = self.echo_time_ms.expect("echo time must be solved at this point");
        let tr_us = self.rep_time_ms * 1e3;
        let n_bvecs = self.shell_idx.as_ref().unwrap().len();
        // flatten all b-vec data into a single list for headfile storage
        let b_vecs:Vec<f64> = self.b_vecs.as_ref().expect("b_vecs must be solved at this point")
            .iter().map(|&b_vec|b_vec).flatten().collect();
        let g_vecs:Vec<f64> = self.g_vecs_tpm.as_ref().expect("g_vecs must be solved at this point")
            .iter().map(|&b_vec|b_vec).flatten().collect();
        let bmat_trace = self.b_trace.as_ref().expect("b_trace must be solved at this point");

        let mut h = Headfile::new();

        h.insert_list_2d("bvecs", 3, n_bvecs, &b_vecs, false);
        h.insert_list_2d("gvecs", 3, n_bvecs, &g_vecs, false);
        h.insert_list_1d("target_bvals", &self.target_bvalues, false);
        h.insert_list_1d("bmat_trace", bmat_trace, false);

        h.fov_x(self.fov_mm[0]);
        h.fov_y(self.fov_mm[0]);
        h.fov_z(self.fov_mm[0]);
        h.dim_x(self.matrix_size[0]);
        h.dim_y(self.matrix_size[1]);
        h.dim_z(self.matrix_size[1]);
        h.bw(self.spectral_width_khz * 1e3 / 2.);
        h.te(te);
        h.ne(1);
        h.tr(tr_us as usize);
        h.n_volumes(n_bvecs);
        h.psd_name(SEQ_NAME);
        h
    }

    fn seq_name() -> &'static str {
        SEQ_NAME
    }
}

impl TOML for DTI {}

impl PulseSequence for DTI {
    fn build_sequence(&mut self) -> SeqLoop {

        // load the b-vector table
        self.read_bvecs();

        // set solve mode to true to trigger correct event configuration
        self.solve_mode = true;
        // render the view loop in 'solve mode'
        let view_loop = self.view_loop();

        // instantiate tunable parameters
        let mut adj = self.adjustment_state();

        // calculate the max gradien strengths for target b-values
        self.calc_gmax(&view_loop, &mut adj);
        // calculate and store the gradient vector table
        self.calc_g_vectors();
        // calculate b-matrix and cross terms for reconstruction
        self.calc_b_matrix(&view_loop, &mut adj);

        // turn off solve mode and re-build the view loop with new gradient info
        self.solve_mode = false;
        let view_loop = self.view_loop();

        // set the number of experiments depending on simulation or setup mode
        let n_experiments = if self.sim_mode {
            2
        }else if self.setup_mode {
            1
        } else {
            self.shell_idx.as_ref().unwrap().len()
        };

        // define the parent 'Experiment' loop
        let mut el = SeqLoop::new(EXPERIMENT, n_experiments);
        // give experiment loop 1 ms to lookup values for diffusion encoding
        el.set_pre_calc(Time::ms(1));
        // insert view loop into parent 'Experiment' loop for diffusion encoding
        el.add_loop(view_loop).unwrap();
        el

    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut h = HashMap::new();
        h.insert(EventControllers::diff_x().to_string(),0.);
        h.insert(EventControllers::diff_y().to_string(),0.);
        h.insert(EventControllers::diff_z().to_string(),0.);
        h.insert(EventControllers::prephase_adj().to_string(),0.);
        h.insert(EventControllers::exp_pow().to_string(),0.5);
        h.insert(EventControllers::ref_pow().to_string(),1.);
        h
    }

    fn finish_acquisition(&mut self, acq_dir: impl AsRef<Path>) {
        todo!()
    }

    fn gop_mode(&mut self) {
        self.setup_mode = true;
    }

    fn acq_mode(&mut self) {
        self.setup_mode = false;
        self.sim_mode = false;
        self.solve_mode = false;
    }

    fn display_mode(&mut self) {
        self.sim_mode = true;
    }

    fn sim_mode(&mut self) {
        self.sim_mode = true;
    }
}

struct EventControllers {
    exc_pow:RFP,
    ref_pow:RFP,
    ref_phase:RFPhase,
    readout: GS,
    diffusion_x: GS,
    diffusion_y: GS,
    diffusion_z: GS,
    phase_enc_y: GS,
    phase_enc_z: GS,
    prephase_x: GS,
    spoiler: GS,
}

impl EventControllers {
    fn build(dti:&mut DTI) -> EventControllers {

        let exc_pow = EventControl::<f64>::new().with_adj(Self::exp_pow()).to_shared();
        let ref_pow = EventControl::<f64>::new().with_adj(Self::ref_pow()).to_shared();
        // base phase is 90 deg from excitation phase with a 180 deg chop every view
        let ref_phase = EventControl::<Angle>::new().with_constant(Angle::deg(0.0)).to_shared();
        let dt = Freq::khz(dti.spectral_width_khz).inv();
        let gro = FieldGrad::from_fov(Length::mm(dti.fov_mm[0]),dt,Nuc1H);
        let readout = EventControl::<FieldGrad>::new().with_constant_grad(gro).to_shared();

        let pe_time = Time::ms(dti.phase_enc_dur_ms + dti.imaging_ramp_time_us as f64 * 1e-3);
        let ro_time = dt.scale(dti.matrix_size[0]);

        let pe_step_y = FieldGrad::from_fov(Length::mm(dti.fov_mm[1]),pe_time,Nuc1H);
        let pe_step_z = FieldGrad::from_fov(Length::mm(dti.fov_mm[2]),pe_time,Nuc1H);

        // calculate prephase grad strength
        let ro_moment = ro_time * gro;
        let pe_moment = - ro_moment * 0.5;
        let g_pre:FieldGrad = (pe_moment / pe_time).try_into().unwrap();
        let prephase_x = EventControl::<FieldGrad>::new().with_constant_grad(g_pre)
            .with_adj(Self::prephase_adj()).to_shared();

        let spoiler = EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(200)).to_shared();

        let (diffusion_x,diffusion_y,diffusion_z) = if dti.solve_mode | dti.setup_mode {
            // only turn on the x-diffusion gradient to solve for g_max
            (
                EventControl::<FieldGrad>::new().with_adj(Self::diff_x()).to_shared(),
                EventControl::<FieldGrad>::new().with_adj(Self::diff_y()).to_shared(),
                EventControl::<FieldGrad>::new().with_adj(Self::diff_z()).to_shared(),
            )
        }else {
            // build LUTs for diffusion encoding
            // determine g_max from the table
            let g_max_tpm = *dti.g_max_tpm.as_ref().unwrap().iter().max_by(|a, b|a.partial_cmp(&b).unwrap()).unwrap();

            // discretize the diffusion gradient steps based on g_max
            let diff_grad_resolution = FieldGrad::tesla_per_meter(g_max_tpm / i16::MAX as f64); // grad strength per step

            // build lookup tables for diffusion gradient scaling (similar to spatial encoding)
            let mut gx_lut = vec![];
            let mut gy_lut = vec![];
            let mut gz_lut = vec![];

            dti.g_vecs_tpm.as_ref().expect("g_vecs must be solved at this stage").iter().for_each(|b|{
                let gx = b[0];
                let gy = b[1];
                let gz = b[2];
                // determine step size
                let step_x = (gx / diff_grad_resolution.si()).floor() as i32;
                let step_y = (gy / diff_grad_resolution.si()).floor() as i32;
                let step_z = (gz / diff_grad_resolution.si()).floor() as i32;
                // push to LUT
                gx_lut.push(step_x);
                gy_lut.push(step_y);
                gz_lut.push(step_z);
            });

            let gx_lut = LUT::new("diff_gx",&gx_lut).to_shared();
            let gy_lut = LUT::new("diff_gy",&gy_lut).to_shared();
            let gz_lut = LUT::new("diff_gz",&gz_lut).to_shared();

            (
                EventControl::<FieldGrad>::new().with_source_loop(EXPERIMENT).with_lut(&gx_lut).with_grad_scale(diff_grad_resolution).to_shared(),
                EventControl::<FieldGrad>::new().with_source_loop(EXPERIMENT).with_lut(&gy_lut).with_grad_scale(diff_grad_resolution).to_shared(),
                EventControl::<FieldGrad>::new().with_source_loop(EXPERIMENT).with_lut(&gz_lut).with_grad_scale(diff_grad_resolution).to_shared(),
            )

        };


        let (phase_enc_y,phase_enc_z) = if dti.solve_mode {
            (
                EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0)).to_shared(),
                EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0)).to_shared(),
            )
        }else if dti.setup_mode {
            let coords = dti.read_pe_table();
            let mut phase_y = vec![];
            let mut phase_z = vec![];
            coords.iter().for_each(|_|{
                phase_y.push(0);
                phase_z.push(0);
            });
            let pey_lut = LUT::new("pe_y",&phase_y).to_shared();
            let pez_lut = LUT::new("pe_z",&phase_z).to_shared();
            (
                EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&pey_lut).with_grad_scale(pe_step_y).to_shared(),
                EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&pez_lut).with_grad_scale(pe_step_z).to_shared()
            )
        }else {
            let coords = dti.read_pe_table();
            let mut phase_y = vec![];
            let mut phase_z = vec![];
            coords.iter().for_each(|coord|{
                phase_y.push(coord[0]);
                phase_z.push(coord[1]);
            });
            let pey_lut = LUT::new("pe_y",&phase_y).to_shared();
            let pez_lut = LUT::new("pe_z",&phase_z).to_shared();
            (
                EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&pey_lut).with_grad_scale(pe_step_y).to_shared(),
                EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&pez_lut).with_grad_scale(pe_step_z).to_shared()
            )
        };

        EventControllers {
            exc_pow,
            ref_pow,
            ref_phase,
            readout,
            diffusion_x,
            diffusion_y,
            diffusion_z,
            phase_enc_y,
            phase_enc_z,
            prephase_x,
            spoiler,
        }
    }

    fn exp_pow() -> &'static str {
        "exc_pow"
    }
    fn ref_pow() -> &'static str {
        "ref_pow"
    }

    fn diff_x() -> &'static str {
        "diff_x"
    }

    fn diff_y() -> &'static str {
        "diff_y"
    }

    fn diff_z() -> &'static str {
        "diff_z"
    }

    fn prephase_adj() -> &'static str {
        "prephase_adj"
    }

}


struct Events {
    exc: RfEvent,
    diff1: GradEvent,
    refoc: RfEvent,
    diff2: GradEvent,
    pe: GradEvent,
    ro_ru: GradEvent,
    acq: ACQEvent,
    ro_rd: GradEvent,
    spoil: GradEvent,
}

impl Events {

    fn build(dti:&mut DTI) -> Events {
        let w = Waveforms::build(dti);
        let ec = EventControllers::build(dti);
        let exc = RfEvent::new(Self::exc(),&w.exc,&ec.exc_pow);
        let refoc = RfEvent::new(Self::refoc(),&w.refoc,&ec.ref_pow).with_phase(&ec.ref_phase);
        let diff1 = GradEvent::new(Self::diff1()).with_x(&w.diff).with_y(&w.diff).with_z(&w.diff)
            .with_strength_x(&ec.diffusion_x).with_strength_y(&ec.diffusion_y).with_strength_z(&ec.diffusion_z);
        let diff2 = diff1.clone_with_label(Self::diff2());
        let pe = GradEvent::new(Self::pe()).with_x(&w.pe).with_y(&w.pe).with_z(&w.pe)
            .with_strength_x(&ec.prephase_x).with_strength_y(&ec.phase_enc_y).with_strength_z(&ec.phase_enc_z);
        let ro_ru = GradEvent::new(Self::ro_ru()).with_x(&w.ru).with_strength_x(&ec.readout);
        let ro_rd = GradEvent::new(Self::ro_rd()).with_x(&w.rd).with_strength_x(&ec.readout);
        let spoil = GradEvent::new(Self::spoil()).with_x(&w.pe).with_y(&w.pe).with_z(&w.pe)
            .with_strength_x(&ec.spoiler).with_strength_y(&ec.spoiler).with_strength_z(&ec.spoiler);
        let acq = ACQEvent::new(Self::acq(),dti.matrix_size[0],Freq::khz(dti.spectral_width_khz).inv());
        Events {
            exc,
            diff1,
            refoc,
            diff2,
            pe,
            ro_ru,
            acq,
            ro_rd,
            spoil,
        }
    }

    fn exc() -> &'static str {
        "exc_pulse"
    }
    fn diff1() -> &'static str {
        "diff1"
    }
    fn refoc() -> &'static str {
        "refoc"
    }
    fn diff2() -> &'static str {
        "diff2"
    }

    fn pe() -> &'static str {
        "pe"
    }

    fn ro_ru() -> &'static str {
        "ro_ru"
    }

    fn ro_rd() -> &'static str {
        "ro_rd"
    }

    fn spoil() -> &'static str {
        "spoil"
    }

    fn acq() -> &'static str {
        "acq"
    }

}

struct Waveforms {
    exc: Rc<RfPulse>,
    refoc: Rc<RfPulse>,
    diff: GW,
    ru: GW,
    rd: GW,
    pe: GW,
}

impl Waveforms {
    fn build(dti:&DTI) -> Waveforms {

        let dt = Time::us(2);

        let exc = hardpulse(
            Time::us(dti.rf_duration_us),
            dt,
            Nuc1H
        ).to_shared();

        let refoc = hardpulse_composite(
            Time::us(2 * dti.rf_duration_us),
            dt,
            Nuc1H
        ).to_shared();

        let diff = trapezoid(
            Time::us(dti.diffusion_ramp_time_us),
            Time::us((dti.little_delta_ms * 1e3) as usize - dti.diffusion_ramp_time_us),
            dt
        ).to_shared();

        let ru = ramp_up(Time::us(dti.imaging_ramp_time_us),dt).to_shared();
        let rd = ramp_down(Time::us(dti.imaging_ramp_time_us),dt).to_shared();

        let pe = trapezoid(
            Time::us(dti.imaging_ramp_time_us),
            Time::ms(dti.phase_enc_dur_ms),
            dt
        ).to_shared();

        Waveforms {
            exc,
            refoc,
            diff,
            pe,
            ru,
            rd,
        }

    }
}