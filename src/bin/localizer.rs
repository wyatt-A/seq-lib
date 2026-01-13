use std::rc::Rc;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Time};
use seq_struct::grad_strength::EventControl;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::waveform::Waveform;
use seq_lib::grad_pulses::{ramp_down, ramp_up, trapezoid};
use seq_lib::rf_pulses;
use seq_lib::defs::{VIEW, SLICE, RF, GW};

enum Plane {
    XY,
    XZ,
    YZ,
}

enum Mode {
    ThreePlane,
    //OnePlane{plane:Plane},
}

struct Localizer {
    /// determines the mode to compile the sequence in
    mode: Mode,
    bandwidth_khz: f64,
    n_samples: usize,
    fov: f64,
    slice_thickness_mm: f64,
    rf_duration_us: usize,
    grad_ramp_time_us: usize,
    phase_enc_dur_ms: f64,
}

impl Default for Localizer {
    fn default() -> Self {
        Self {
            mode: Mode::ThreePlane,
            bandwidth_khz: 100.0,
            n_samples: 256,
            fov: 25.6,
            slice_thickness_mm: 1.,
            rf_duration_us: 1000,
            grad_ramp_time_us: 100,
            phase_enc_dur_ms: 0.5,
        }
    }
}

struct Waveforms {
    rf_pulse: RF,
    ru: GW,
    rd: GW,
    pe: GW,
}

impl Waveforms {
    fn build(params:&Localizer) -> Waveforms {

        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);

        let rf_pulse = rf_pulses::sinc3(
            Time::us(params.rf_duration_us),
            rf_dt,
            Nuc1H
        ).to_shared();

        let ru = ramp_up(
            Time::us(params.grad_ramp_time_us),
            grad_dt
        ).to_shared();

        let rd = ramp_down(
            Time::us(params.grad_ramp_time_us),
            grad_dt
        ).to_shared();

        let pe = trapezoid(
            Time::us(params.grad_ramp_time_us),
            Time::ms(params.phase_enc_dur_ms),
            grad_dt
        ).to_shared();

        Waveforms {
            rf_pulse,
            ru,
            rd,
            pe,
        }

    }
}




fn main() {
    let loop_name = VIEW;
}