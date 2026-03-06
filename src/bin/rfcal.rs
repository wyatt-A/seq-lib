use std::collections::HashMap;
use std::fs::create_dir_all;
use std::path::Path;
use array_lib::io_cfl::write_cfl;
use array_lib::io_mrd::read_mrd;
use clap::Parser;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::rs_fft::{rs_fftn, rs_fftn_batched};
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use mrs_ppl::compile::{build_ppl, compile_ppl};
use pulse_seq_view::run_viewer;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use serde::{Serialize, Deserialize};
use seq_lib::defs::{RFPhase, GS, GW, RF, RFP, VIEW};
use seq_lib::grad_pulses::{ramp_down, ramp_up};
use seq_lib::{Args, PulseSequence, ToHeadfile, TOML};
use seq_lib::rf_pulses::hardpulse;

const SEQ_NAME: &str = "rfcal";

fn main() {


    let args = Args::parse();

    let setup_dir = args.setup_dir();
    let acq_dir = args.acq_dir();
    let sim_dir = args.sim_dir();

    if args.init {
        let rf_cal = RFCal::default();
        rf_cal.to_file(args.param_file);
        return
    }

    if args.display {
        let mut rf_cal = RFCal::from_file(args.param_file);
        let seq_loop = rf_cal.build_sequence();
        let user_state = rf_cal.adjustment_state();
        let ps_data = seq_loop.render_timeline(&user_state).to_raw();
        run_viewer(ps_data).unwrap();
        return
    }

    if args.setup {
        let mut rf_cal = RFCal::from_file(args.param_file);
        rf_cal.setup_mode = true;
        let seq_loop = rf_cal.build_sequence();
        create_dir_all(&setup_dir).unwrap();
        build_ppl(&seq_loop, &setup_dir, SEQ_NAME, false);
        rf_cal.to_file(setup_dir.join(SEQ_NAME));
        let hf = rf_cal.headfile();
        hf.to_file(&setup_dir.join(format!("{SEQ_NAME}_setup"))).unwrap();
        if args.skip_ppl_compile {
            return
        }
        compile_ppl(&setup_dir);
        return
    }

    if args.acquire {
        let mut rf_cal = RFCal::from_file(args.param_file);
        let seq_loop = rf_cal.build_sequence();
        create_dir_all(&acq_dir).unwrap();
        build_ppl(&seq_loop, &acq_dir, SEQ_NAME, false);
        rf_cal.to_file(acq_dir.join(SEQ_NAME));
        if args.skip_ppl_compile {
            return
        }
        compile_ppl(&acq_dir);
        // copy ppl params
        // run acquisition
        return
    }

    if args.finish {
        let mut rf_cal = RFCal::from_file(acq_dir.join(SEQ_NAME));
        finish_acquisition(&mut rf_cal, &acq_dir);
        rf_cal.to_file(acq_dir.join(SEQ_NAME));
        let mut hf = rf_cal.headfile();
        hf.write_timestamp();
        hf.to_file(&acq_dir.join(SEQ_NAME)).unwrap();
        return
    }

    if args.sim {
        let mut rf_cal = RFCal::from_file(args.param_file);
        let seq_loop = rf_cal.build_sequence();
        create_dir_all(&sim_dir).unwrap();
        rf_cal.to_file(sim_dir.join(SEQ_NAME));
        build_ppl(&seq_loop, &sim_dir, SEQ_NAME, true);
        compile_ppl(&sim_dir);
        return
    }

}

fn finish_acquisition(rf_cal:&mut RFCal, acq_dir:impl AsRef<Path>) {

    let (mut calib_data,dims,..) = read_mrd(&acq_dir.as_ref().join(format!("{SEQ_NAME}.mrd")));
    assert_eq!(dims.shape_squeeze(),vec![rf_cal.n_samples,rf_cal.n_steps,2], "unexpected dimensions");

    println!("{:?}",dims.shape_squeeze());

    rs_fftn_batched(&mut calib_data,&[rf_cal.n_samples],rf_cal.n_steps*2, FftDirection::Inverse,NormalizationType::Unitary);

    let avg_sig:Vec<f32> = calib_data.chunks_exact(rf_cal.n_samples).map(|readout|{
        // this is the average slice signal (K0)
        readout[0].norm()
    }).collect();

    let sig_pairs:Vec<&[f32]> = avg_sig.chunks_exact(rf_cal.n_steps).collect();

    let mut max = 0.;
    let mut m_idx = 0;
    let mut ratios = vec![];
    for i in 0..rf_cal.n_steps {
        let sig_ratio = sig_pairs[0][i] / sig_pairs[1][i];
        ratios.push(sig_ratio);
        if sig_ratio > max {
            max = sig_ratio;
            m_idx = i;
        }
    }

    println!("sig ratios: {:?}",ratios);
    println!("{}",max);
    println!("idx = {}",m_idx);
    println!("pow_frac = {}",rf_cal.power_steps_frac.as_ref().unwrap().get(m_idx).unwrap());


    write_cfl(acq_dir.as_ref().join(format!("{SEQ_NAME}.cfl")),&calib_data,dims);

}




#[derive(Clone,Debug,Serialize,Deserialize)]
struct RFCal {
    setup_mode: bool,
    spec_width_khz: f64,
    n_samples: usize,
    slice_thickness_mm: f64,
    echo_time_ms: f64,
    mixing_time_ms: f64,
    rep_time_ms: f64,
    hard_pulse_dur_ms: f64,
    init_power_db: f64,
    final_power_db: f64,
    n_steps: usize,
    ramp_time_ms: f64,
    stabilize_time_ms: f64,
    spoil_time_ms: f64,
    power_steps_frac: Option<Vec<f64>>,
}

impl TOML for RFCal {}
impl ToHeadfile for RFCal {}

impl Default for RFCal {
    fn default() -> RFCal {
        RFCal {
            setup_mode: false,
            spec_width_khz: 25.,
            n_samples: 256,
            slice_thickness_mm: 5.,
            echo_time_ms: 15.,
            mixing_time_ms: 45.,
            rep_time_ms: 1_000.,
            hard_pulse_dur_ms: 0.5,
            init_power_db: -20.,
            final_power_db: -3.,
            n_steps: 10,
            ramp_time_ms: 0.5,
            stabilize_time_ms: 5.,
            spoil_time_ms: 5.,
            power_steps_frac: None,
        }
    }
}

impl PulseSequence for RFCal {
    fn build_sequence(&mut self) -> SeqLoop {

        let events = Events::build(self);

        if self.setup_mode {
            self.n_steps = 1_000;
        }
        
        let mut vl = SeqLoop::new_main(VIEW,self.n_steps);

        vl.add_event(events.e_ramp_up).unwrap();
        vl.add_event(events.e_exc).unwrap();
        vl.add_event(events.e_ref).unwrap();
        vl.add_event(events.e_echo1).unwrap();
        vl.add_event(events.e_stim).unwrap();
        vl.add_event(events.e_echo2).unwrap();
        vl.add_event(events.e_ramp_down).unwrap();

        vl.set_pre_calc(Time::ms(4));


        let mut tau = Time::ms(self.echo_time_ms / 2.);
        let mut t_mix = Time::ms(self.mixing_time_ms);

        if let Ok(stab_time) =vl.set_min_time_span(Events::ramp_up(),Events::excite(),100,0,Time::ms(self.stabilize_time_ms)) {
            self.stabilize_time_ms = stab_time.as_ms();
        }else {
            panic!("Failed to set stabilize time");
        }

        // attempt to set the echo time. If it's not possible, then the closest possible is used
        if let Ok(t) = vl.set_min_time_span(Events::refoc(),Events::echo1(),50,50,tau) {
            tau = t;
            self.echo_time_ms = t.as_ms() * 2.;
        }else {
            panic!("Failed to set echo time");
        }

        // this should always be possible
        vl.set_time_span(Events::stim(),Events::echo2(),50,50,tau).expect("Failed to set tau");

        // this should always be possible
        vl.set_time_span(Events::excite(),Events::refoc(),50,50,tau).expect("Failed to set tau");

        // attempt to set mixing time
        if let Ok(t) = vl.set_min_time_span(Events::refoc(),Events::stim(),50,50,t_mix) {
            t_mix = t;
            self.mixing_time_ms = t_mix.as_ms();
        }else {
            panic!("Failed to set mixing time");
        }

        if let Ok(t) = vl.set_min_time_span(Events::echo2(),Events::ramp_down(),100,0,Time::ms(self.spoil_time_ms)) {
            self.spoil_time_ms = t.as_ms();
        }else {
            panic!("Failed to set spoiler time");
        }

        vl.set_rep_time(Time::ms(self.rep_time_ms)).expect("Failed to set rep time");

        vl

    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut h = HashMap::new();
        if self.setup_mode {
            h.insert("rf_power".to_string(),0.0);
        }
        h
    }
}


struct Waveforms {
    ramp_up: GW,
    ramp_down: GW,
    p1: RF,
    p2: RF,
}

impl Waveforms {
    pub fn build(rfcal: &RFCal) -> Waveforms {

        let dt = Time::us(2);
        let tramp = Time::ms(rfcal.ramp_time_ms);

        let ramp_up = ramp_up(tramp,dt).to_shared();
        let ramp_down = ramp_down(tramp,dt).to_shared();

        let p1 = hardpulse(Time::ms(rfcal.hard_pulse_dur_ms),dt,Nuc1H).to_shared();
        let p2 = hardpulse(Time::ms(2. * rfcal.hard_pulse_dur_ms),dt,Nuc1H).to_shared();

        Waveforms {
            ramp_up,
            ramp_down,
            p1,
            p2,
        }

    }
}

struct EventControllers {
    c_gro: GS,
    c_rfp: RFP,
    c_phase_exc: RFPhase,
    c_phase_ref: RFPhase,
}

impl EventControllers {
    pub fn build(rf_cal:&mut RFCal) -> EventControllers {

        let w = Waveforms::build(rf_cal);
        let gro = w.p1.grad_strength(Length::mm(rf_cal.slice_thickness_mm));

        let c_gro = EventControl::<FieldGrad>::new().with_constant_grad(gro).to_shared();

        let init_frac = db_to_frac(rf_cal.init_power_db);
        let final_frac = db_to_frac(rf_cal.final_power_db);
        let frac_step = (final_frac - init_frac) / (rf_cal.n_steps as f64 - 1.0);

        let frac_steps = (0..rf_cal.n_steps).map(|i| i as f64 * frac_step + init_frac).collect();

        let c_rfp = if rf_cal.setup_mode {
            EventControl::<f64>::new().with_adj("rf_power").to_shared()
        }else {
            rf_cal.power_steps_frac = Some(frac_steps);
            EventControl::<f64>::new().with_source_loop(VIEW).with_scale(frac_step).with_constant(init_frac).to_shared()
        };

        let c_phase_exc = EventControl::<Angle>::new().with_constant(Angle::deg(45)).to_shared();
        let c_phase_ref = EventControl::<Angle>::new().with_constant(Angle::deg(315)).to_shared();

        EventControllers {
            c_gro,
            c_rfp,
            c_phase_exc,
            c_phase_ref,
        }

    }
}

struct Events {
    e_ramp_up: GradEvent,
    e_exc: RfEvent,
    e_ref: RfEvent,
    e_echo1: ACQEvent,
    e_stim: RfEvent,
    e_echo2: ACQEvent,
    e_ramp_down: GradEvent,
}

impl Events {
    fn build(rf_cal:&mut RFCal) -> Events {

        let w = Waveforms::build(rf_cal);
        let ec = EventControllers::build(rf_cal);

        let e_ramp_up = GradEvent::new(Events::ramp_up()).with_z(&w.ramp_up).with_strength_z(&ec.c_gro);
        let e_ramp_down = GradEvent::new(Events::ramp_down()).with_z(&w.ramp_down).with_strength_z(&ec.c_gro);

        let e_exc = RfEvent::new(Events::excite(),&w.p1,&ec.c_rfp).with_phase(&ec.c_phase_exc);
        let e_ref = RfEvent::new(Events::refoc(),&w.p2,&ec.c_rfp).with_phase(&ec.c_phase_ref);
        let e_stim = RfEvent::new(Events::stim(),&w.p1,&ec.c_rfp).with_phase(&ec.c_phase_ref);

        let e_echo1 = ACQEvent::new(Events::echo1(),rf_cal.n_samples,Freq::khz(rf_cal.spec_width_khz).inv());
        let e_echo2 = e_echo1.clone_with_label(Events::echo2());


        Events {
            e_ramp_up,
            e_exc,
            e_ref,
            e_echo1,
            e_stim,
            e_echo2,
            e_ramp_down,
        }

    }

    fn ramp_up() -> &'static str {
        "ru"
    }

    fn ramp_down() -> &'static str {
        "rd"
    }

    fn excite() -> &'static str {
        "exc"
    }

    fn refoc() -> &'static str {
        "ref"
    }

    fn stim() -> &'static str {
        "stim"
    }

    fn echo1() -> &'static str {
        "echo1"
    }

    fn echo2() -> &'static str {
        "echo2"
    }

}




fn db_to_frac(db:f64) -> f64 {
    10f64.powf(db/ 10.)
}