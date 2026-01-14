use std::rc::Rc;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::grad_strength::EventControl;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::variable::LUT;
use seq_struct::waveform::Waveform;
use seq_lib::grad_pulses::{ramp_down, ramp_up, trapezoid};
use seq_lib::rf_pulses;
use seq_lib::defs::{VIEW, SLICE, RF, GW, GS};

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

struct EventControllers {
    gro: GS,
    gsl: GS,
    gslr: GS,
    gpe: GS,
    gpre: GS,
}

impl EventControllers {
    fn build(params:&Localizer, waveforms: &Waveforms) -> EventControllers {

        let t_dwell = Freq::khz(params.bandwidth_khz).inv();
        let acq_t = t_dwell.scale(params.n_samples);

        let gread = FieldGrad::from_fov(Length::mm(params.fov),t_dwell,Nuc1H);
        let gro = EventControl::<FieldGrad>::new().with_constant_grad(gread).to_shared();

        let rt = Time::us(params.grad_ramp_time_us);

        // calculate pre-phase gradient strength
        let m1:Time = (acq_t + rt).try_into().unwrap();
        let m2:Time = (Time::ms(params.phase_enc_dur_ms) + rt).try_into().unwrap();
        let f = m1 / m2;
        let g_pre = gread.scale(-f.si() / 2.);
        let gpre = EventControl::<FieldGrad>::new().with_constant_grad(g_pre).to_shared();

        let g_sl = waveforms.rf_pulse.grad_strength(
            Length::mm(params.slice_thickness_mm)
        );

        let gsl = EventControl::<FieldGrad>::new().with_constant_grad(g_sl).to_shared();

        let m1:Time = (waveforms.rf_pulse.duration() + rt).try_into().unwrap();
        let m2:Time = (Time::ms(params.phase_enc_dur_ms) + rt).try_into().unwrap();
        let f = m1 / m2;
        let g_slr = gread.scale(-f.si() / 2.);
        let gslr = EventControl::<FieldGrad>::new().with_constant_grad(g_slr).to_shared();

        let tpe:Time = (Time::ms(params.phase_enc_dur_ms) + rt).try_into().unwrap();
        let gpe_step = FieldGrad::from_fov(Length::mm(params.fov),tpe,Nuc1H);

        let mut steps = vec![];
        for p in 0..params.n_samples as i32 {
            steps.push(p - p/2)
        }

        let pe_lut = LUT::new("pelut",&steps).to_shared();
        let gpe = EventControl::<FieldGrad>::new().with_lut(&pe_lut).with_grad_scale(gpe_step).to_shared();

        EventControllers {
            gro,
            gsl,
            gslr,
            gpe,
            gpre,
        }

    }
}




fn main() {
    let loop_name = VIEW;
}