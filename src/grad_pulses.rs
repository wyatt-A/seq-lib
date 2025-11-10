use mr_units::primitive::Time;
use seq_struct::waveform::Waveform;

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