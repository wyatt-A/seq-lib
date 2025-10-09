use mr_units::constants::Nucleus;
use mr_units::primitive::Time;
use num_complex::{Complex64};
use seq_struct::rf_pulse::RfPulse;
use seq_struct::waveform::Waveform;

pub fn hardpulse(duration:Time, dt:Time, nucleus: Nucleus) -> RfPulse {
    let n_samples = (duration / dt).si().round() as usize;
    let w = Waveform::new().constant(1,n_samples,dt).to_shared();
    RfPulse::new(&w,2.,nucleus)
}

pub fn hardpulse_composite(duration:Time, dt:Time, nucleus: Nucleus) -> RfPulse {
    let n_samples = (duration / dt).si().round() as usize;
    let n_0 = n_samples / 4;
    let n_90 = n_0 * 2;
    let mut s = vec![Complex64::ONE;n_0];
    s.extend(vec![Complex64::i();n_90]);
    s.extend(vec![Complex64::ONE;n_0]);
    let w = Waveform::new().add_list_c(&s,dt).to_shared();
    RfPulse::new(&w,2.,nucleus)
}