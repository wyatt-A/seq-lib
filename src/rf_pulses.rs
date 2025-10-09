use std::f64::consts::PI;
use mr_units::constants::Nucleus;
use mr_units::primitive::Time;
use num_complex::{Complex64};
use seq_struct::rf_pulse::RfPulse;
use seq_struct::waveform::{linspace, Waveform};

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



pub fn sinc1(duration:Time, dt:Time, nuc:Nucleus) -> RfPulse {
    let n_samples = (duration / dt).si().round() as usize;
    let x = linspace(-PI,PI,n_samples);
    let h_x = linspace(-PI,PI,n_samples);
    let h = h_x.into_iter().map(|x| (x.cos() + 1.) / 2. );
    let y:Vec<_> = x.into_iter().zip(h).map(|(x,h)| {
        if x == 0. {
            1.
        }else {
            h * (x.sin() / x)
        }
    } ).collect();
    let w = Waveform::new().add_list_r(&y,dt).to_shared();
    RfPulse::new(&w,1.,nuc)
}

pub fn sinc3(duration:Time, dt:Time, nuc:Nucleus) -> RfPulse {
    let n_samples = (duration / dt).si().round() as usize;
    let x = linspace(-2. * PI,2. * PI,n_samples);
    let h_x = linspace(-PI,PI,n_samples);
    let h = h_x.into_iter().map(|x| (x.cos() + 1.) / 2. );
    let y:Vec<_> = x.into_iter().zip(h).map(|(x,h)| {
        if x == 0. {
            1.
        }else {
            h * (x.sin() / x)
        }
    } ).collect();
    let w = Waveform::new().add_list_r(&y,dt).to_shared();
    RfPulse::new(&w,3.,nuc)
}

pub fn sinc5(duration:Time, dt:Time, nuc:Nucleus) -> RfPulse {
    let n_samples = (duration / dt).si().round() as usize;
    let x = linspace(-3. * PI,3. * PI,n_samples);
    let h_x = linspace(-PI,PI,n_samples);
    let h = h_x.into_iter().map(|x| (x.cos() + 1.) / 2. );
    let y:Vec<_> = x.into_iter().zip(h).map(|(x,h)| {
        if x == 0. {
            1.
        }else {
            h * (x.sin() / x)
        }
    } ).collect();
    let w = Waveform::new().add_list_r(&y,dt).to_shared();
    RfPulse::new(&w,5.,nuc)
}