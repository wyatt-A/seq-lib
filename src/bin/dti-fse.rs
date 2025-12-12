use std::collections::HashMap;
use std::rc::Rc;
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
use mrs_ppl::compile::compile_seq;
use seq_lib::q_calc::{binary_solve, calc_b_matrix};

// shorthand types
type GW = Rc<Waveform>;
type RF = Rc<RfPulse>;
type GS = Rc<EventControl<FieldGrad>>;
type RFP = Rc<EventControl<f64>>;
type RFPhase = Rc<EventControl<Angle>>;

// loop names
const VIEW:&str = "view";
const ECHO:&str = "echo";


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
    bandwidth_khz: f64,
    n_samples: usize,
    fov_x_mm: f64,
    fov_y_mm: f64,
    fov_z_mm: f64,
}


struct Waveforms {
    ramp_up: GW,
    ramp_down: GW,
    diffusion: GW,
    phase_enc: GW,
    rf90: RF,
    rf180: RF,

    imaging_ramp_time:Time,
    diffusion_ramp_time:Time,
    diffusion_pulse_plat_dur:Time,
    pe_dur:Time,
    rf_dur:Time,
}

impl Waveforms {
    fn build(params:&DTIFse) -> Waveforms {

        let imaging_ramp_time = Time::us(200);
        let diffusion_ramp_time = Time::ms(1);
        let diffusion_pulse_plat_dur = Time::ms(4);
        let pe_dur = Time::ms(1);
        let rf_dur = Time::us(100);

        // waveform time-base
        let grad_dt = Time::us(2);
        let rf_dt = grad_dt;

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
            diffusion_ramp_time,
            diffusion_pulse_plat_dur,
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
    prephase_x: GS,
    crush_left_x: GS,
    crush_right_x: GS,
}

impl EventControllers {
    fn build(params:&DTIFse, w:&Waveforms) -> EventControllers {

        let dwell_time = Freq::khz(params.bandwidth_khz).inv();
        let acq_time = dwell_time.scale(params.n_samples);

        let gro = FieldGrad::from_fov(Length::mm(params.fov_x_mm), dwell_time, Nuc1H);

        let readout = EventControl::<FieldGrad>::new().with_constant_grad(gro).to_shared();

        let diffusion_x = EventControl::<FieldGrad>::new().with_adj("diff_x").to_shared();
        let diffusion_y = EventControl::<FieldGrad>::new().with_adj("diff_y").to_shared();
        let diffusion_z = EventControl::<FieldGrad>::new().with_adj("diff_z").to_shared();

        let pe_time = Time::try_from(w.imaging_ramp_time + w.pe_dur).unwrap();
        let ro_time = Time::try_from(w.imaging_ramp_time + acq_time).unwrap();

        let mut y_steps = vec![];
        let mut z_steps = vec![];
        for y in -2..=2 {
            for z in -2..=2 {
                y_steps.push(y);
                z_steps.push(z);
            }
        }

        let phase_step_y = FieldGrad::from_fov(Length::mm(params.fov_y_mm), pe_time, Nuc1H);
        let phase_step_z = FieldGrad::from_fov(Length::mm(params.fov_z_mm), pe_time, Nuc1H);

        let lut_y = LUT::new("lut_y",&y_steps).to_shared();
        let lut_z = LUT::new("lut_z",&z_steps).to_shared();

        let phase_enc_y = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_y).with_grad_scale(phase_step_y).to_shared();
        let phase_enc_z = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_z).with_grad_scale(phase_step_z).to_shared();

        // inverse of the phase encodes
        let phase_rewind_y = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_y).with_grad_scale(phase_step_y.scale(-1)).to_shared();
        let phase_rewind_z = EventControl::<FieldGrad>::new().with_source_loop(VIEW).with_lut(&lut_z).with_grad_scale(phase_step_z.scale(-1)).to_shared();

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
        let refoc = RfEvent::new(REF,&w.rf180,&e.ref_pow);
        let refoct = RfEvent::new(REFT, &w.rf180,&e.ref_pow);

        let diff1 = GradEvent::new(DIFF1)
            .with_x(&w.diffusion).with_y(&w.diffusion).with_z(&w.diffusion)
            .with_strength_x(&e.diffusion_x).with_strength_y(&e.diffusion_y).with_strength_z(&e.diffusion_z);

        let diff2 = GradEvent::new(DIFF2)
            .with_x(&w.diffusion).with_y(&w.diffusion).with_z(&w.diffusion)
            .with_strength_x(&e.diffusion_x).with_strength_y(&e.diffusion_y).with_strength_z(&e.diffusion_z);

        let pe = GradEvent::new(PE).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.prephase_x).with_strength_y(&e.phase_enc_y).with_strength_z(&e.phase_enc_z);

        let re = GradEvent::new(RE).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.crush_left_x).with_strength_y(&e.phase_rewind_y).with_strength_z(&e.phase_rewind_z);

        let petl = GradEvent::new(PETL).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.crush_left_x).with_strength_y(&e.phase_enc_y).with_strength_z(&e.phase_enc_z);

        let petr = GradEvent::new(PETR).with_x(&w.phase_enc).with_y(&w.phase_enc).with_z(&w.phase_enc)
            .with_strength_x(&e.crush_left_x).with_strength_y(&e.phase_rewind_y).with_strength_z(&e.phase_rewind_z);

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

impl Default for DTIFse {
    fn default() -> Self {
        DTIFse {
            bandwidth_khz: 100.,
            n_samples: 512,
            fov_x_mm: 20.,
            fov_y_mm: 12.,
            fov_z_mm: 12.,
        }
    }
}

impl PulseSequence for DTIFse {
    fn compile(&self) -> SeqLoop {

        let w = Waveforms::build(self);
        let e = EventControllers::build(self,&w);
        let events = Events::build(self,&w,&e);

        // view loop events
        let mut vl = SeqLoop::new_main(VIEW,1);
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
        let mut el = SeqLoop::new(ECHO,5);
        el.add_event(events.refoct).unwrap();
        el.add_event(events.petl).unwrap();
        el.add_event(events.rot_ru).unwrap();
        el.add_event(events.acqt).unwrap();
        el.add_event(events.rot_rd).unwrap();
        el.add_event(events.petr).unwrap();

        el.set_time_span(REFT,PETL,100,0,Time::us(400)).unwrap();
        el.set_time_span(PETL,ROT_RU,100,0,Time::us(100)).unwrap();
        el.set_min_time_span(ROT_RU,ACQT,100,0,Time::us(100)).unwrap();
        el.set_min_time_span(ACQT,ROT_RD,100,0,Time::us(100)).unwrap();
        el.set_time_span(ROT_RD,PETR,100,0,Time::us(100)).unwrap();

        let tau2 = el.get_time_span(REFT,ACQT,50,50).unwrap();
        let echo_spacing = tau2.scale(2);
        println!("echo spacing: {} ms",echo_spacing.as_ms());

        el.set_pre_calc(Time::ms(1));
        el.set_rep_time(echo_spacing).expect("failed to set echo spacing");
        el.set_averages(0);

        vl.add_loop(el).unwrap();

        vl.set_time_span(RE,REFT,100,0,Time::us(400)).unwrap();
        vl.set_pre_calc(Time::ms(3));
        vl.set_rep_time(Time::ms(100)).unwrap();
        vl
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("crush_left".to_string(),0.05);
        state.insert("crush_right".to_string(), 0.07);
        state.insert("prephase_x".to_string(),0.);
        state.insert("rf90".to_string(), 1.);
        state.insert("rf180".to_string(), 2.);
        state.insert("diff_x".to_string(), 0.);
        state.insert("diff_y".to_string(), 0.);
        state.insert("diff_z".to_string(), 0.);
        state
    }
}

fn main() {
    let s = DTIFse::default();
    let p = s.compile();
    let mut a = s.adjustment_state();
    // *a.get_mut("crush_left").unwrap() = 0.;
    // *a.get_mut("crush_right").unwrap() = 0.;

    let mut t_inv = p.find_occurrences(REF,50);
    t_inv.extend(
        p.find_occurrences(REFT,50)
    );
    let t_inv:Vec<_> = t_inv.into_iter().map(|t|t.as_sec()).collect();

    // build list of echo times to evaluate each b-matrix
    let mut t_echoes = vec![];

    // get the center of the first echo from the center of ACQ
    t_echoes.push(
        p.find_occurrences(ACQ,50).get(0).unwrap().as_sec()
    );

    // get the remaining echoes from the echo train
    let echo_train = p.find_occurrences(ACQT,50);
    for i in 0..5 { // number of acqT
        t_echoes.push(
            echo_train[i].as_sec()
        )
    }

    println!("echo times to evaluate: {:?}", t_echoes);

    let f = |g| {
        *a.get_mut("diff_x").unwrap() = g;
        let w = p.render_timeline(&a).render();
        calc_b_matrix(&w,&t_inv,t_echoes[0],Nuc1H).trace()
    };
    let soltn = binary_solve(0.,2.,13_000.,1e-6,100,f);
    println!("{:?}", soltn);

    *a.get_mut("diff_x").unwrap() = soltn;
    let w = p.render_timeline(&a).render();
    //evaluate b-matrix for each echo time and report the trace
    for t in t_echoes {
        let bmat = calc_b_matrix(&w,&t_inv,t,Nuc1H);
        let b = bmat.trace();
        println!("{:?}",b)
    }

    s.render_to_file(&a,"fse_dti");
    //compile_seq(&s.compile(),"ppl","seq",false);

}