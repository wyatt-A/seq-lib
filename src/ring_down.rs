use std::collections::HashMap;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{FieldGrad, Freq, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use crate::grad_pulses::trapezoid;
use crate::PulseSequence;
use crate::rf_pulses::hardpulse;

pub struct RingDown {
    n_reps: usize,
    spectral_width_hz: f64,
    n_samples: usize,
    ramp_time_us: usize,
    grad_pulse_dur_ms: f64,
    rf_pulse_dur_us: usize,
    pub ring_delay_us: usize,
    acq_delay_us: usize,
    rep_time_ms: f64,
}

impl Default for RingDown {
    fn default() -> RingDown {
        RingDown {
            n_reps: 20,
            spectral_width_hz: 100_000.,
            n_samples: 1024,
            ramp_time_us: 250,
            grad_pulse_dur_ms: 5.,
            rf_pulse_dur_us: 140,
            ring_delay_us: 500,
            acq_delay_us: 200,
            rep_time_ms: 500.,
        }
    }
}


impl PulseSequence for RingDown {
    fn compile(&self) -> SeqLoop {

        let specwidth = Freq::hz(self.spectral_width_hz);
        let t_dwell = specwidth.inv();

        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);

        let c_rf_pow = EventControl::<f64>::new()
            .with_adj("rf_pow")
            .to_shared();

        let c_grad_x = EventControl::<FieldGrad>::new()
            .with_adj("grad_x").to_shared();

        let c_grad_y = EventControl::<FieldGrad>::new()
            .with_adj("grad_y").to_shared();

        let c_grad_z = EventControl::<FieldGrad>::new()
            .with_adj("grad_z").to_shared();

        let h90 = hardpulse(Time::us(self.rf_pulse_dur_us),rf_dt,Nuc1H).to_shared();

        let w_t = trapezoid(Time::us(self.ramp_time_us),Time::ms(self.grad_pulse_dur_ms),grad_dt)
            .to_shared();

        let e_grad = GradEvent::new("grad")
            .with_x(&w_t).with_y(&w_t).with_z(&w_t)
            .with_strength_x(&c_grad_x).with_strength_y(&c_grad_y).with_strength_z(&c_grad_z);
        let e_exc = RfEvent::new("exc", &h90, &c_rf_pow);
        let e_acq = ACQEvent::new("acq", self.n_samples, t_dwell);

        let mut vl = SeqLoop::new_main("view",self.n_reps);

        vl.add_event(e_grad).unwrap();
        vl.add_event(e_exc).unwrap();
        vl.add_event(e_acq).unwrap();

        vl.set_pre_calc(Time::ms(3));
        vl.set_time_span("grad","exc",100,0,Time::us(self.ring_delay_us)).unwrap();
        vl.set_time_span("exc","acq",100,0,Time::us(self.acq_delay_us)).unwrap();
        vl.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();
        vl
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        HashMap::from([
            (String::from("grad_x"), 0.0),
            (String::from("grad_y"),0.0),
            (String::from("grad_z"),0.0),
        ])
    }
}