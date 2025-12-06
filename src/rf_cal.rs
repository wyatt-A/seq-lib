use std::collections::HashMap;
use std::rc::Rc;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::waveform::Waveform;
use crate::grad_pulses::{ramp_down, ramp_up};
use crate::PulseSequence;
use crate::rf_pulses::hardpulse;

pub struct RfCal {
    slice_thickness_mm: f64,
    rf_dir_us: usize,
    tau1_ms: f64,
    t_fill_ms: f64,
    n_steps: usize,
    ramp_time_us: usize,
    n_samples: usize,
    bandwidth_khz: f64,
    rep_time_ms: f64,
}

type GW = Rc<Waveform>;
type GC = Rc<EventControl<FieldGrad>>;
type PC = Rc<EventControl<f64>>;
type RF = Rc<RfPulse>;


struct Waveforms {
    rf_pulse: RF,
    ramp_up: GW,
    ramp_down: GW,
}

impl Waveforms {
    fn new(params:&RfCal) -> Waveforms {

        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);
        let rf_dur = Time::us(params.rf_dir_us);
        let rf_pulse = hardpulse(rf_dur,rf_dt,Nuc1H).to_shared();
        let grad_ramp_time = Time::us(params.ramp_time_us);

        let ramp_up = ramp_up(grad_ramp_time,grad_dt).to_shared();
        let ramp_down = ramp_down(grad_ramp_time,grad_dt).to_shared();

        Waveforms {
            rf_pulse,
            ramp_up,
            ramp_down
        }

    }
}

struct EventControllers {
    rf_power: PC,
    grad_strength: GC
}

impl EventControllers {

    fn new(params:&RfCal,w:&Waveforms) -> EventControllers {

        let gs = w.rf_pulse.grad_strength(
            Length::mm(params.slice_thickness_mm)
        );

        let rf_power = EventControl::<f64>::new().with_adj("rf_pow").to_shared();
        let grad_strength = EventControl::<FieldGrad>::new().with_constant_grad(gs).to_shared();

        EventControllers {
            rf_power,
            grad_strength,
        }

    }

}

struct EventLabels {
    ramp_up: &'static str,
    ramp_down: &'static str,
    alpha_1: &'static str,
    alpha_2: &'static str,
    alpha_3: &'static str,
    acq_1: &'static str,
    acq_2: &'static str,
}
impl EventLabels {
    fn new() -> EventLabels {
        EventLabels {
            ramp_up: "ru",
            ramp_down: "rd",
            alpha_1: "alpha_1",
            alpha_2: "alpha_2",
            alpha_3: "alpha_3",
            acq_1: "acq_1",
            acq_2: "acq_2",
        }
    }
}

struct Events {
    alpha_1: RfEvent,
    alpha_2: RfEvent,
    alpha_3: RfEvent,
    ramp_up: GradEvent,
    ramp_down: GradEvent,
    acq_1: ACQEvent,
    acq_2: ACQEvent,
}

impl Events {
    fn new(params: &RfCal,el:&EventLabels, w:&Waveforms,ec:&EventControllers) -> Events {

        let alpha_1 = RfEvent::new(el.alpha_1,&w.rf_pulse,&ec.rf_power);
        let alpha_2 = RfEvent::new(el.alpha_2,&w.rf_pulse,&ec.rf_power);
        let alpha_3 = RfEvent::new(el.alpha_3,&w.rf_pulse,&ec.rf_power);

        let ramp_up = GradEvent::new(el.ramp_up).with_x(&w.ramp_up).with_strength_x(&ec.grad_strength);
        let ramp_down = GradEvent::new(el.ramp_down).with_x(&w.ramp_down).with_strength_x(&ec.grad_strength);

        let t_dwell = Freq::khz(params.bandwidth_khz).inv();

        let acq_1 = ACQEvent::new(el.acq_1,params.n_samples,t_dwell);
        let acq_2 = ACQEvent::new(el.acq_2,params.n_samples,t_dwell);

        Events {
            alpha_1,
            alpha_2,
            alpha_3,
            ramp_up,
            ramp_down,
            acq_1,
            acq_2,
        }

    }
}

impl Default for RfCal {
    fn default() -> Self {
        RfCal {
            slice_thickness_mm: 1.0,
            rf_dir_us: 200,
            tau1_ms: 5.0,
            t_fill_ms: 5.0,
            n_steps: 3,
            ramp_time_us: 500,
            n_samples: 256,
            bandwidth_khz: 30.,
            rep_time_ms: 1000.,
        }
    }
}

impl PulseSequence for RfCal {
    fn compile(&self) -> SeqLoop {

        let el = EventLabels::new();
        let w = Waveforms::new(self);
        let ec = EventControllers::new(self, &w);
        let events = Events::new(self,&el,&w,&ec);

        let mut vl = SeqLoop::new_main("view",self.n_steps);

        vl.add_event(events.ramp_up).unwrap();
        vl.add_event(events.alpha_1).unwrap();
        vl.add_event(events.alpha_2).unwrap();
        vl.add_event(events.acq_1).unwrap();
        vl.add_event(events.alpha_3).unwrap();
        vl.add_event(events.acq_2).unwrap();
        vl.add_event(events.ramp_down).unwrap();

        vl.set_time_span(el.ramp_up,el.alpha_1,100,0,Time::ms(1)).unwrap();
        vl.set_time_span(el.alpha_1,el.alpha_2,50,50,Time::ms(self.tau1_ms)).unwrap();
        vl.set_time_span(el.alpha_2,el.acq_1,50,50,Time::ms(self.tau1_ms)).unwrap();

        vl.set_min_time_span(el.acq_1,el.alpha_3,100,0,Time::ms(self.t_fill_ms)).unwrap();

        vl.set_time_span(el.alpha_3,el.acq_2,50,50,Time::ms(self.tau1_ms)).unwrap();
        vl.set_time_span(el.acq_2,el.ramp_down,100,0,Time::ms(1)).unwrap();

        vl.set_pre_calc(Time::ms(2));
        vl.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();

        vl
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("rf_pow".to_string(),1.);
        state
    }
}