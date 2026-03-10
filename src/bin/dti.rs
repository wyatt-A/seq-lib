use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use clap::Parser;
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
use serde::{Deserialize, Serialize};
use seq_lib::defs::{RFPhase, EXPERIMENT, GS, GW, RFP, VIEW};
use seq_lib::grad_pulses::{ramp_up, trapezoid};
use seq_lib::{Args, PulseSequence, ToHeadfile, TOML};
use seq_lib::q_calc::{grad_solve, load_bvecs};
use seq_lib::rf_pulses::{hardpulse, hardpulse_composite};

fn main() {
    let args = Args::parse();
    args.run::<DTI>();
}

const SEQ_NAME: &str = "dti";
#[derive(Debug,Clone,Serialize,Deserialize)]
struct DTI {
    /// determines the mode to compile the sequence in
    setup_mode: bool,
    sim_mode: bool,
    /// solve mode calculates the required g_max for the desired diffusion weighting
    solve_mode: bool,
    bandwidth_khz: f64,
    fov_mm: [f64;3],
    matrix_size: [usize;3],
    n_phase: Option<usize>,
    rf_duration_us: usize,
    diffusion_ramp_time_us: usize,
    imaging_ramp_time_us: usize,
    phase_enc_dur_ms: f64,
    little_delta_ms: f64,
    big_delta_ms: f64,
    rep_time_ms: f64,
    g_limit_tpm: f64,
    cs_table: PathBuf,
    bvec_table: PathBuf,
    target_bvalues: Vec<f64>,
    /// max diffusion gradient strength
    g_max_mtpm: Option<Vec<f64>>,
    dg_x_dir: Option<Vec<f64>>,
    dg_y_dir: Option<Vec<f64>>,
    dg_z_dir: Option<Vec<f64>>,
    shell_idx: Option<Vec<usize>>,
    echo_time_ms: Option<f64>,
}

impl Default for DTI {
    fn default() -> Self {
        DTI {
            setup_mode: false,
            sim_mode: false,
            solve_mode: false,
            bandwidth_khz: 100.,
            fov_mm: [25.6, 12.8,12.8],
            matrix_size: [512, 256, 256],
            n_phase: None,
            rf_duration_us: 100,
            diffusion_ramp_time_us: 500,
            imaging_ramp_time_us: 200,
            phase_enc_dur_ms: 0.5,
            little_delta_ms: 1.7,
            big_delta_ms: 5.,
            rep_time_ms: 100.,
            g_limit_tpm: 2.3,
            cs_table: PathBuf::from(r"C:\workstation\data\petableCS_stream\stream_CS256_8x_pa18_pb54"),
            bvec_table: PathBuf::from(r"C:\workstation\data\diffusion_table\26.wang.06\bvecs.txt"),
            target_bvalues: vec![1_000.,3_000.,5_000.,8_000.,10_000.,12_000.],
            g_max_mtpm: None,
            dg_x_dir: None,
            dg_y_dir: None,
            dg_z_dir: None,
            shell_idx: None,
            echo_time_ms: None,
        }
    }
}

impl DTI {

    fn read_bvecs(&mut self) {
        let (shell_idx,b_vecs) = load_bvecs(&self.bvec_table);
        assert_eq!(shell_idx.len(), self.target_bvalues.len(),"number of target b-values does not match b-vector length");
        let dg_x = b_vecs.iter().map(|x| x[0]).collect::<Vec<f64>>();
        let dg_y = b_vecs.iter().map(|x| x[1]).collect::<Vec<f64>>();
        let dg_z = b_vecs.iter().map(|x| x[2]).collect::<Vec<f64>>();
        self.dg_x_dir = Some(dg_x);
        self.dg_y_dir = Some(dg_y);
        self.dg_z_dir = Some(dg_z);
        self.shell_idx = Some(shell_idx);
    }

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

        let n_views = if self.solve_mode {
            1
        }else if self.sim_mode {
            4
        }else {
            self.read_pe_table();
            self.n_phase.unwrap()
        };

        let events = Events::build(self);

        let mut vl = SeqLoop::new_main(VIEW,n_views);

        vl.add_event(events.exc).unwrap();
        vl.add_event(events.diff1).unwrap();
        vl.add_event(events.refoc).unwrap();
        vl.add_event(events.diff2).unwrap();
        vl.add_event(events.pe).unwrap();
        vl.add_event(events.ro_ru).unwrap();
        vl.add_event(events.acq).unwrap();
        vl.add_event(events.ro_rd).unwrap();
        vl.add_event(events.spoil).unwrap();

        vl.set_pre_calc(Time::ms(4));

        // delay after rf pulses, before diffusion gradient start
        let post_rf_del = Time::us(50);

        vl.set_time_span(Events::exc(),Events::diff1(),100,0,post_rf_del).unwrap();
        vl.set_time_span(Events::refoc(),Events::diff2(),100,0,post_rf_del).unwrap();

        let big_delta = Time::ms(self.big_delta_ms);
        if let Ok(delta) = vl.set_min_time_span(Events::diff1(),Events::diff2(),0,0,big_delta) {
            println!("set big delta to {} ms",delta.as_ms());
            self.big_delta_ms = delta.as_ms();
        }else {
            panic!("failed to set big delta ms");
        }

        let post_diff2_del = Time::us(50);
        vl.set_time_span(Events::diff2(),Events::pe(),100,0,post_diff2_del).unwrap();

        let post_pe_del = Time::us(50);
        vl.set_time_span(Events::pe(),Events::ro_ru(),100,0,post_pe_del).unwrap();

        vl.set_min_time_span(Events::ro_ru(),Events::acq(),100,0,Time::us(50)).unwrap();
        vl.set_min_time_span(Events::acq(),Events::ro_rd(),100,0,Time::us(50)).unwrap();
        vl.set_time_span(Events::ro_rd(),Events::spoil(),100,0,Time::us(50)).unwrap();

        vl.set_averages(1);
        vl.set_rep_time(Time::ms(self.rep_time_ms)).expect("failed to set rep time");

        let te = vl.get_time_span(Events::exc(),Events::acq(),50,50).expect("failed to get echo time");
        println!("min echo time: {} ms",te.as_ms());
        self.echo_time_ms = Some(te.as_ms());

        vl
    }
}

impl ToHeadfile for DTI {
    fn seq_name() -> &'static str {
        SEQ_NAME
    }
}

impl TOML for DTI {}

impl PulseSequence for DTI {
    fn build_sequence(&mut self) -> SeqLoop {

        self.read_bvecs();

        self.solve_mode = true;
        let vl = self.view_loop();

        let adj = self.adjustment_state();

        // figure out gradient strengths
        let grad_soltns:Vec<_> = self.target_bvalues.iter().map(|&target_bval|{
            grad_solve(
                &vl,
                adj.clone(),
                EventControllers::diff_x(),
                target_bval,
                FieldGrad::tesla_per_meter(0), // lower limit
                FieldGrad::tesla_per_meter(self.g_limit_tpm), // upper limit
                &[Events::refoc()],
                Time::ms(self.echo_time_ms.unwrap()),
            )
        }).collect();

        let gmax:Vec<_> = grad_soltns.iter().map(|g| g.si() * 1e3).collect();
        self.g_max_mtpm = Some(gmax);

        self.solve_mode = false;
        let vl = self.view_loop();

        let n_exp = self.shell_idx.as_ref().unwrap().len();

        let mut el = SeqLoop::new(EXPERIMENT,n_exp);
        el.no_overhead();
        el.add_loop(vl).unwrap();
        el

    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut h = HashMap::new();
        if self.solve_mode {
            h.insert(EventControllers::diff_x().to_string(),0.);
        }
        h
    }

    fn finish_acquisition(&mut self, acq_dir: impl AsRef<Path>) {
        todo!()
    }

    fn gop_mode(&mut self) {
        todo!()
    }

    fn acq_mode(&mut self) {
        todo!()
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
        let ref_phase = EventControl::<Angle>::new().with_source_loop(VIEW)
            .with_mod(2).with_scale(Angle::deg(180)).with_constant(Angle::deg(90.0)).to_shared();
        let dt = Freq::khz(dti.bandwidth_khz).inv();
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

        let (diffusion_x,diffusion_y,diffusion_z) = if dti.solve_mode {
            // only turn on the x-diffusion gradient to solve for g_max
            (
                EventControl::<FieldGrad>::new().with_adj(Self::diff_x()).to_shared(),
                EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0)).to_shared(),
                EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0)).to_shared(),
            )
        }else {
            // build LUTs for diffusion encoding
            // determine g_max from the table
            let g_max_mtpm = *dti.g_max_mtpm.as_ref().unwrap().iter().max_by(|a,b|a.partial_cmp(&b).unwrap()).unwrap();

            // discretize the diffusion gradient steps based on g_max
            let diff_grad_resolution = FieldGrad::mt_per_meter(g_max_mtpm / i16::MAX as f64); // grad strength per step

            // build lookup tables for diffusion gradient scaling (similar to spatial encoding)
            let mut gx_lut = vec![];
            let mut gy_lut = vec![];
            let mut gz_lut = vec![];

            let shell_idx = dti.shell_idx.as_ref().unwrap();
            dti.dg_x_dir.as_ref().unwrap().iter().zip(dti.dg_y_dir.as_ref().unwrap().iter()).zip(dti.dg_z_dir.as_ref().unwrap().iter()).enumerate().for_each(|(i,((&ux,&uy),&uz))| {
                // get gradient strength for this shell
                let g_mtpm = dti.g_max_mtpm.as_ref().unwrap()[shell_idx[i]];
                // scale by b-vector
                let gx = FieldGrad::mt_per_meter(g_mtpm).scale(ux);
                let gy = FieldGrad::mt_per_meter(g_mtpm).scale(uy);
                let gz = FieldGrad::mt_per_meter(g_mtpm).scale(uz);
                // determine step size
                let step_x = (gx.si() / diff_grad_resolution.si()).floor() as i32;
                let step_y = (gy.si() / diff_grad_resolution.si()).floor() as i32;
                let step_z = (gz.si() / diff_grad_resolution.si()).floor() as i32;
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
        let acq = ACQEvent::new(Self::acq(),dti.matrix_size[0],Freq::khz(dti.bandwidth_khz).inv());
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
        "diff1"
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
            Time::ms(dti.little_delta_ms),
            dt
        ).to_shared();

        let ru = ramp_up(Time::us(dti.imaging_ramp_time_us),dt).to_shared();
        let rd = ramp_up(Time::us(dti.imaging_ramp_time_us),dt).to_shared();

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