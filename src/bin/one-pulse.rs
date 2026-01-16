use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::rc::Rc;
use clap::Parser;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Freq, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::SeqLoop;
use seq_lib::{InputArgs, PulseSequence};
use seq_lib::rf_pulses::hardpulse;
use serde::{Deserialize, Serialize};

type Power = Rc<EventControl<f64>>;
type RF = Rc<RfPulse>;


const RF: &'static str = "rf";
const ACQ: &'static str = "acq";

#[derive(Serialize,Deserialize)]
struct OnePulse {
    bandwidth_khz: f64,
    n_samples: usize,
    rf_dur_us: usize,
    delay_us: usize,
    n_reps: usize,
    rep_time_ms: f64,
}

impl Default for OnePulse {
    fn default() -> Self {
        OnePulse {
            bandwidth_khz: 100.,
            n_samples: 2048,
            rf_dur_us: 100,
            delay_us: 500,
            n_reps: 5,
            rep_time_ms: 100.
        }
    }
}





impl PulseSequence for OnePulse {
    fn compile(&self) -> SeqLoop {
        let mut vl = SeqLoop::new_main("view",self.n_reps);
        let events = Events::new(self);
        vl.add_event(events.rf_pulse).unwrap();
        vl.add_event(events.acq).unwrap();
        vl.set_time_span(RF,ACQ,50,0,Time::us(self.delay_us)).unwrap();
        vl.set_pre_calc(Time::ms(2));
        vl.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();
        vl
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}

struct Waveforms {
    hard_pulse: RF,
}

impl Waveforms {
    fn new(params:&OnePulse) -> Waveforms {
        let rf_dt = Time::us(2);
        let duration = Time::us(params.rf_dur_us);
        let hard_pulse = hardpulse(duration,rf_dt,Nuc1H).to_shared();
        Waveforms {
            hard_pulse
        }
    }
}

struct EventControllers {
    rf_power: Power
}

impl EventControllers {
    pub fn new() -> EventControllers {
        let rf_power = EventControl::<f64>::new().with_constant(1.0).to_shared();
        EventControllers {
            rf_power
        }
    }
}

struct Events {
    rf_pulse: RfEvent,
    acq: ACQEvent,
}

impl Events {
    fn new(params:&OnePulse) -> Events {
        let w = Waveforms::new(params);
        let ec = EventControllers::new();
        let rf_pulse = RfEvent::new(RF,&w.hard_pulse,&ec.rf_power);
        let dwell = Freq::khz(params.bandwidth_khz).inv();
        let acq = ACQEvent::new(ACQ,params.n_samples,dwell);
        Events {
            rf_pulse,
            acq
        }
    }
}

fn main() {
    let args = InputArgs::parse();
    if args.default {
        let p = OnePulse::default();
        let config = toml::to_string_pretty(&p).unwrap();;
        let mut file = File::create(&args.input.with_extension("toml")).unwrap();
        file.write_all(config.as_bytes()).unwrap();
    }
    let mut file = File::open(&args.input.with_extension("toml")).unwrap();
    let mut toml_str = String::new();
    file.read_to_string(&mut toml_str).unwrap();
    let one_pulse:OnePulse = toml::from_str(&toml_str).unwrap();
    let state = one_pulse.adjustment_state();
    one_pulse.render_to_file(&state,&args.output);
}
