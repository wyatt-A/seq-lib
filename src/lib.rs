mod rf_pulses;
pub use seq_struct;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::waveform::Waveform;
use crate::rf_pulses::hardpulse;

#[cfg(test)]
mod tests {
    use crate::{PulseSequence, SliceProfileSE};

    #[test]
    pub fn slice_profile_se() {

        let s = SliceProfileSE {
            exc_dur_us: 140,
            ref_dur_us: 280,
            ramp_time_us: 200,
            crush_dur_us: 300,
            bw_hz: 100_000.,
            n_reps: 10,
            fov_mm: 20.,
            n_samples: 256,
            te_ms: 10.,
        };

        let seq = s.compile();

    }

}

/// Specifies a data structure that compiles to a pulse sequence
pub trait PulseSequence: Default {
    /// main pulse sequence generation routine. This is where all the pulse sequence logic is
    /// implemented
    fn compile(&self) -> SeqLoop;
}

pub struct SliceProfileSE {
    exc_dur_us: usize,
    ref_dur_us: usize,
    ramp_time_us: usize,
    crush_dur_us: usize,
    bw_hz: f64,
    n_reps: usize,
    fov_mm: f64,
    n_samples: usize,
    te_ms: f64,
}

impl Default for SliceProfileSE {
    fn default() -> SliceProfileSE {
        SliceProfileSE {
            exc_dur_us: 140,
            ref_dur_us: 280,
            ramp_time_us: 200,
            crush_dur_us: 300,
            bw_hz: 100_000.,
            n_reps: 10,
            fov_mm: 20.,
            n_samples: 256,
            te_ms: 10.,
        }
    }
}

impl PulseSequence for SliceProfileSE {
    fn compile(&self) -> SeqLoop {

        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);

        let bw = Freq::hz(self.bw_hz);
        let t_dwell = bw.inv();

        let acq_time = t_dwell.scale(self.n_samples).scale(1.1);

        // single main loop structure
        let mut vl = SeqLoop::new_main("view",self.n_reps);

        let p_exc = hardpulse(
            Time::us(self.exc_dur_us),
            rf_dt,
            Nuc1H
        ).to_shared();

        let p_ref = hardpulse(
            Time::us(self.ref_dur_us),
            rf_dt,
            Nuc1H
        ).to_shared();

        let n_ramp_samp = (Time::us(self.ramp_time_us) / grad_dt).si().round() as usize;
        // let w_ru = Waveform::new().ramp(0,1,n_ramp_samp,grad_dt);
        // let w_rd = Waveform::new().ramp(1,0,n_ramp_samp,grad_dt);

        let n_ss_samp = (Time::us(self.exc_dur_us * 3) / grad_dt).si().round() as usize;
        let w_ss = Waveform::new().ramp(0,1,n_ramp_samp,grad_dt)
            .constant(1,n_ss_samp,grad_dt)
            .ramp(1,0,n_ramp_samp,grad_dt)
            .to_shared();

        let w_ss_ref = Waveform::new().ramp(0,1,n_ramp_samp,grad_dt)
            .constant(1,n_ss_samp,grad_dt)
            .ramp(1,0,n_ramp_samp,grad_dt)
            .to_shared();

        let n_crush_samp = (Time::us(self.crush_dur_us) / grad_dt).si().round() as usize;
        let w_crush = Waveform::new()
            .ramp(0,1,n_ramp_samp,grad_dt)
            .constant(1,n_crush_samp,grad_dt)
            .ramp(1,0,n_ramp_samp,grad_dt)
            .to_shared();

        let n_ro_samp = ( acq_time / grad_dt ).si().ceil() as usize;
        let w_ro = Waveform::new().ramp(0,1,n_ramp_samp,grad_dt)
            .constant(1,n_ro_samp,grad_dt)
            .ramp(1,0,n_ramp_samp,grad_dt)
            .to_shared();

        let c_exc_pow = EventControl::<f64>::new().with_adj("exc_pow").to_shared();
        let c_ref_pow = EventControl::<f64>::new().with_adj("ref_pow").to_shared();

        let c_ss_grad = EventControl::<FieldGrad>::new().with_adj("ss").to_shared();
        let c_ss_ref_grad = EventControl::<FieldGrad>::new().with_adj("ss_ref").to_shared();

        let c_crush = EventControl::<FieldGrad>::new().with_adj("crush").to_shared();
        let c_prephase = EventControl::<FieldGrad>::new().with_adj("prephase").to_shared();
        let c_ro = EventControl::<FieldGrad>::new().with_constant_grad(
            FieldGrad::from_fov(Length::mm(self.fov_mm),t_dwell,Nuc1H)
        ).to_shared();
        let c_180_phase = EventControl::<Angle>::new().with_constant(Angle::deg(90)).to_shared();

        let e_exc = RfEvent::new("exc",&p_exc,&c_exc_pow);
        let e_ref = RfEvent::new("ref",&p_ref,&c_ref_pow).with_phase(&c_180_phase);

        let e_ss = GradEvent::new("ss").with_z(&w_ss).with_strength_z(&c_ss_grad);
        let e_sr = GradEvent::new("ss_ref").with_z(&w_ss_ref).with_strength_z(&c_ss_ref_grad).with_x(&w_ss_ref).with_strength_x(&c_prephase);
        let e_crush_left = GradEvent::new("crush_l").with_x(&w_crush).with_strength_x(&c_crush);
        let e_crush_right = GradEvent::new("crush_r").with_x(&w_crush).with_strength_x(&c_crush);
        let e_ro = GradEvent::new("ro").with_x(&w_ro).with_strength_x(&c_ro);
        let e_acq = ACQEvent::new("acq",self.n_samples,t_dwell);

        vl.add_event(e_ss).unwrap();
        vl.add_event(e_exc).unwrap();
        vl.add_event(e_sr).unwrap();
        vl.add_event(e_crush_left).unwrap();
        vl.add_event(e_ref).unwrap();
        vl.add_event(e_crush_right).unwrap();
        vl.add_event(e_ro).unwrap();
        vl.add_event(e_acq).unwrap();

        let tau = Time::ms(self.te_ms).scale(0.5);

        vl.set_pre_calc(Time::ms(2));
        // center the slice select gradient with the exc pulse
        vl.set_time_span("ss","exc",50,50,Time::us(0)).unwrap();
        // end of ss to start of ss_ref
        vl.set_time_span("ss","ss_ref",100,0,Time::ms(1)).unwrap();

        vl.set_time_span("crush_l","ref",100,0,Time::us(200)).unwrap();
        vl.set_time_span("ref","crush_r",100,0,Time::us(200)).unwrap();
        vl.set_time_span("exc","ref",50,50,tau).unwrap();
        vl.set_time_span("ref","ro",50,50,tau).unwrap();
        vl.set_time_span("ref","acq",50,50,tau).unwrap();

        vl
    }
}