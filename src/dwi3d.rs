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
use crate::grad_pulses::{ramp_down, ramp_up, trapezoid};
use crate::PulseSequence;
use crate::rf_pulses::{hardpulse, hardpulse_composite, sinc3};

pub struct Dwi3DParams {
    diffusion_ramp_time_us: u32,
    diffusion_const_time_ms: f64,
    ramp_time_us: u32,
    pe_time_ms: f64,
    fov_x_mm: f64,
    fov_y_mm: f64,
    fov_z_mm: f64,
    n_x: usize,
    n_y: usize,
    n_z: usize,
    n_views: usize,
    bw_hz:f64,
}

impl Default for Dwi3DParams {
    fn default() -> Self {
        Dwi3DParams {
            diffusion_ramp_time_us: 400,
            diffusion_const_time_ms: 5.,
            ramp_time_us: 200,
            pe_time_ms: 1.,
            fov_x_mm: 19.2,
            fov_y_mm: 12.0,
            fov_z_mm: 12.0,
            n_x: 384,
            n_y: 240,
            n_z: 240,
            n_views: 1,
            bw_hz: 100_000.,
        }
    }
}

impl PulseSequence for Dwi3DParams {
    fn compile(&self) -> SeqLoop {

        // time between end of rf90 and start of first diffusion pulse
        let del1_us = 100;
        // time between end of rf180 and start of second diffusion pulse
        let del2_us = 100;

        // separation between diffusion gradients
        let big_delta_ms = 10;
        let te_ms = 50;
        let tr_ms = 100;

        let w = Waveforms::new(&self);
        let ec = EventControllers::new(&self,&w);
        let events = Events::new(&self,&w,&ec);

        let mut vl = SeqLoop::new_main("view",self.n_views);

        vl.add_event(events.rf90).unwrap();
        vl.add_event(events.diff1).unwrap();
        vl.add_event(events.rf180).unwrap();
        vl.add_event(events.diff2).unwrap();
        vl.add_event(events.phase_enc).unwrap();
        vl.add_event(events.read_ru).unwrap();
        vl.add_event(events.acq).unwrap();
        vl.add_event(events.read_rd).unwrap();

        vl.set_time_span("rf90","diff1",100,0,Time::us(del1_us)).unwrap();
        vl.set_time_span("rf180","diff2",100,0,Time::us(del2_us)).unwrap();
        vl.set_time_span("diff1","diff2",0,0,Time::ms(big_delta_ms)).unwrap();

        vl.set_min_time_span("ru","acq",100,0,Time::us(0)).unwrap();
        vl.set_min_time_span("acq","rd",100,0,Time::us(0)).unwrap();
        vl.set_time_span("pe","ru",100,0,Time::us(100)).unwrap();

        vl.set_time_span("rf90","acq",50,50,Time::ms(te_ms)).unwrap();

        vl.set_pre_calc(Time::ms(2));

        vl.set_rep_time(Time::ms(tr_ms)).unwrap();

        vl
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        HashMap::from_iter(
            [
                ("read_prephase".to_string(), 0.),
                ("diff_x".to_string(), 1.0),
                ("diff_y".to_string(), 1.0),
                ("diff_z".to_string(), 1.0),
                ("rf90_pow".to_string(), 1.0),
                ("rf180_pow".to_string(), 2.0),
                ("rf180_phase".to_string(), 0.0),
            ]
        )
    }
}

struct Events {
    rf90: RfEvent,
    diff1: GradEvent,
    rf180: RfEvent,
    diff2: GradEvent,
    phase_enc: GradEvent,
    read_ru: GradEvent,
    acq: ACQEvent,
    read_rd: GradEvent,
}

impl Events {
    pub fn new(params:&Dwi3DParams, w:&Waveforms, ec: &EventControllers) -> Events {

        let rf90 = RfEvent::new("rf90",&w.rf_90,&ec.c_rf90);
        let rf180 = RfEvent::new("rf180",&w.rf_180,&ec.c_rf180).with_phase(&ec.c_rf180_phase);

        let diff1 = GradEvent::new("diff1")
            .with_x(&w.diffusion)
            .with_y(&w.diffusion)
            .with_z(&w.diffusion)
            .with_strength_x(&ec.c_diff_x)
            .with_strength_y(&ec.c_diff_y)
            .with_strength_z(&ec.c_diff_z);

        let diff2 = diff1.clone_with_label("diff2");

        let phase_enc = GradEvent::new("pe")
            .with_x(&w.phase_encode)
            .with_y(&w.phase_encode)
            .with_z(&w.phase_encode)
            .with_strength_x(&ec.c_pe_x)
            .with_strength_y(&ec.c_pe_y)
            .with_strength_z(&ec.c_pe_z);

        let read_ru = GradEvent::new("ru").with_x(&w.ramp_up).with_strength_x(&ec.c_ro);
        let read_rd = GradEvent::new("rd").with_x(&w.ramp_down).with_strength_x(&ec.c_ro);
        let acq = ACQEvent::new("acq",params.n_x,Freq::hz(params.bw_hz).inv());

        Events {
            rf90,
            diff1,
            rf180,
            diff2,
            phase_enc,
            read_ru,
            acq,
            read_rd,
        }
    }
}


struct Waveforms {
    diffusion: Rc<Waveform>,
    ramp_up: Rc<Waveform>,
    ramp_down: Rc<Waveform>,
    phase_encode: Rc<Waveform>,
    rf_90: Rc<RfPulse>,
    rf_180: Rc<RfPulse>,
}

impl Waveforms {
    pub fn new(params:&Dwi3DParams) -> Waveforms {

        let grad_dt = Time::us(2);
        let rf_dt = Time::us(2);

        // non-selective excitation and refocusing rf pulses
        let rf_90 = hardpulse(Time::us(100),rf_dt,Nuc1H).to_shared();
        //let rf_90 = sinc3(Time::us(100),rf_dt,Nuc1H).to_shared();
        let rf_180 = hardpulse_composite(Time::us(200),rf_dt,Nuc1H).to_shared();
        //let rf_180 = sinc3(Time::us(200),rf_dt,Nuc1H).to_shared();

        // diffusion encoding pulse
        let diffusion = trapezoid(Time::us(params.diffusion_ramp_time_us),Time::ms(params.diffusion_const_time_ms),grad_dt).to_shared();

        // ramp up / ramp down for readout gradient
        let ramp_up = ramp_up(Time::us(params.ramp_time_us),grad_dt).to_shared();
        let ramp_down = ramp_down(Time::us(params.ramp_time_us),grad_dt).to_shared();

        // phase encoding pulse
        let phase_encode = trapezoid(Time::us(params.ramp_time_us),Time::ms(params.pe_time_ms),grad_dt).to_shared();

        Waveforms {
            diffusion,
            ramp_up,
            ramp_down,
            phase_encode,
            rf_90,
            rf_180,
        }

    }
}

struct EventControllers {
    c_ro: Rc<EventControl<FieldGrad>>,
    c_diff_x: Rc<EventControl<FieldGrad>>,
    c_diff_y: Rc<EventControl<FieldGrad>>,
    c_diff_z: Rc<EventControl<FieldGrad>>,
    c_pe_x: Rc<EventControl<FieldGrad>>,
    c_pe_y: Rc<EventControl<FieldGrad>>,
    c_pe_z: Rc<EventControl<FieldGrad>>,
    c_rf90: Rc<EventControl<f64>>,
    c_rf180: Rc<EventControl<f64>>,
    c_rf180_phase: Rc<EventControl<Angle>>,
}

impl EventControllers {
    pub fn new(params:&Dwi3DParams,waveforms:&Waveforms) -> EventControllers {

        let g_ro = FieldGrad::from_fov(
            Length::mm(params.fov_x_mm),
            Freq::hz(params.bw_hz).inv(),
            Nuc1H,
        );

        let c_ro = EventControl::<FieldGrad>::new().with_constant_grad(g_ro).to_shared();

        let lut_pe_y = LUT::new("pe_y",&vec![0;params.n_views]).to_shared();
        let lut_pe_z = LUT::new("pe_z",&vec![0;params.n_views]).to_shared();

        // characteristic time for the readout gradient
        let acq_time:Time = (
            Freq::hz(params.bw_hz).inv().scale(params.n_x).quantity() +
                Time::us(params.ramp_time_us).quantity())
            .try_into().unwrap();

        let pe_time = waveforms.phase_encode.integrate_real();
        let ratio = (acq_time / pe_time).si();
        let c_pe_x = EventControl::<FieldGrad>::new().with_constant_grad(
            g_ro.scale( - ratio / 2.),
        )
            .with_adj("read_prephase")
            .to_shared();

        let c_pe_y = EventControl::<FieldGrad>::new().with_grad_scale(
            FieldGrad::from_fov(
                Length::mm(params.fov_y_mm),
                waveforms.phase_encode.integrate_real(),
                Nuc1H,
            )
        )
            .with_source_loop("view")
            .with_lut(&lut_pe_y)
            .to_shared();

        let c_pe_z = EventControl::<FieldGrad>::new().with_grad_scale(
            FieldGrad::from_fov(
                Length::mm(params.fov_z_mm),
                waveforms.phase_encode.integrate_real(),
                Nuc1H,
            )
        )
            .with_source_loop("view")
            .with_lut(&lut_pe_z)
            .to_shared();

        let c_diff_x = EventControl::<FieldGrad>::new().with_adj("diff_x").to_shared();
        let c_diff_y = EventControl::<FieldGrad>::new().with_adj("diff_y").to_shared();
        let c_diff_z = EventControl::<FieldGrad>::new().with_adj("diff_z").to_shared();

        let c_rf90 = EventControl::<f64>::new().with_adj("rf90_pow").to_shared();
        let c_rf180 = EventControl::<f64>::new().with_adj("rf180_pow").to_shared();
        let c_rf180_phase = EventControl::<Angle>::new().with_adj("rf180_phase").to_shared();

        EventControllers {
            c_ro,
            c_diff_x,
            c_diff_y,
            c_diff_z,
            c_pe_x,
            c_pe_y,
            c_pe_z,
            c_rf90,
            c_rf180,
            c_rf180_phase,
        }

    }
}







