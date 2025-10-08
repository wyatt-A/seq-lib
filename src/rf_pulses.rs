use mr_units::constants::Nucleus;
use mr_units::primitive::Time;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::waveform::Waveform;

pub fn hardpulse(duration:Time, dt:Time, nucleus: Nucleus) -> RfPulse {
    let n_samples = (duration / dt).si().round() as usize;
    let w = Waveform::new().constant(1,n_samples,dt).to_shared();
    RfPulse::new(&w,2.,nucleus)
}