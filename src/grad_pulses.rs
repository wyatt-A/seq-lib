use mr_units::primitive::Time;
use seq_struct::waveform::Waveform;


pub mod scale_factors {
    use std::f64::consts::PI;

    /// scale relative to a hard pulse with same duration and magnitude
    pub const HALF_SIN_SCALE:f64 = 2./PI;

    /// scale relative to a linear ramp of the same duration and magnitude
    pub const QUARTER_SIN_SCALE:f64 = 2./PI;

}


pub fn trapezoid(ramp_time:Time, const_time:Time, dt:Time) -> Waveform {
    let n_ramp = (ramp_time / dt).si().round() as usize;
    let n_const = (const_time / dt).si().round() as usize;
    Waveform::new().ramp(0,1,n_ramp,dt).constant(1,n_const,dt).ramp(1,0,n_ramp,dt)
}

pub fn half_sin(duration:Time, dt:Time) -> Waveform {
    let n_samples = (duration / dt).si().round() as usize;
    Waveform::new().add_half_sin(0,n_samples,dt)
}

pub fn ramp_up(duration:Time, dt:Time) -> Waveform {
    let n_samples = (duration / dt).si().round() as usize;
    Waveform::new().ramp(0, 1, n_samples, dt)
}

pub fn ramp_down(duration:Time, dt:Time) -> Waveform {
    let n_samples = (duration / dt).si().round() as usize;
    Waveform::new().ramp(1, 0, n_samples, dt)
}

/// quarter-sin ramp-up. This is a smooth version of ramp_up
pub fn quarter_sin_ru(duration:Time, dt:Time) -> Waveform {
    let n_samples = (duration / dt).si().round() as usize;
    let quadrant = 0;
    Waveform::new().add_quarter_sin(quadrant,n_samples,dt)
}

/// quarter-sin ramp-down. This is a smooth version of ramp_down
pub fn quarter_sin_rd(duration:Time, dt:Time) -> Waveform {
    let n_samples = (duration / dt).si().round() as usize;
    let quadrant = 1;
    Waveform::new().add_quarter_sin(quadrant,n_samples,dt)
}