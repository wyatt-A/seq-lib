mod rf_pulses;
mod grad_pulses;
pub mod ring_down;
pub mod se3d;
pub mod ssme;
pub mod dwi3d;
pub mod q_calc;

use std::collections::HashMap;
use std::path::Path;
pub use seq_struct;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::compile::{Seq, Timeline};
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::waveform::Waveform;
use crate::rf_pulses::{hardpulse, hardpulse_composite, sinc5};

#[cfg(test)]
mod tests {
    use crate::{PulseSequence, SliceProfileSE};

    #[test]
    pub fn slice_profile_se() {

        let s = SliceProfileSE {
            rf_dur_us: 140,
            ramp_time_us: 200,
            crush_dur_us: 300,
            bw_hz: 100_000.,
            n_reps: 10,
            fov_mm: 20.,
            n_samples: 256,
            te_ms: 10.,
            rep_time_ms: 500.0,
        };

        let seq = s.compile();
        let adj_state = s.adjustment_state();
        let t = seq.render_timeline(&adj_state);
        t.write_to_file("se_out.txt")
    }

}

/// Specifies a data structure that compiles to a pulse sequence
pub trait PulseSequence: Default {
    /// main pulse sequence generation routine. This is where all the pulse sequence logic is
    /// implemented
    fn compile(&self) -> SeqLoop;
    fn adjustment_state(&self) -> HashMap<String,f64>;

    /// render sequence timeline `[t,Gx,Gy,Gz,Bx,By,rec]` where t (sec), G (T/m), B (T), rec (rad)
    fn render(&self,state:&HashMap<String,f64>) -> Seq {
        self.compile().render_timeline(state).render()
    }

    fn render_to_file(&self, state:&HashMap<String,f64>, filename:impl AsRef<Path>) {
        self.compile().render_timeline(state).write_to_file(filename)
    }

}

pub struct SliceProfileSE {
    rf_dur_us: usize,
    ramp_time_us: usize,
    crush_dur_us: usize,
    bw_hz: f64,
    n_reps: usize,
    fov_mm: f64,
    n_samples: usize,
    te_ms: f64,
    rep_time_ms: f64,
}

impl Default for SliceProfileSE {
    fn default() -> SliceProfileSE {
        SliceProfileSE {
            rf_dur_us: 500,
            ramp_time_us: 200,
            crush_dur_us: 300,
            bw_hz: 100_000.,
            n_reps: 10,
            fov_mm: 20.,
            n_samples: 256,
            te_ms: 20.,
            rep_time_ms: 500.
        }
    }
}

impl PulseSequence for SliceProfileSE {
    // define all adjustment states
    fn adjustment_state(&self) -> HashMap<String,f64> {
        let mut state = HashMap::new();
        state.insert("exc_pow".to_string(),1.);
        state.insert("ref_pow".to_string(),2.);
        state.insert("ss".to_string(),0.5);
        state.insert("ss_ref".to_string(),0.1);
        state.insert("crush".to_string(),0.5);
        state.insert("prephase".to_string(),0.5);
        state
    }

    fn compile(&self) -> SeqLoop {

        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);

        let bw = Freq::hz(self.bw_hz);
        let t_dwell = bw.inv();

        let acq_time = t_dwell.scale(self.n_samples).scale(1.1);

        // single main loop structure
        let mut vl = SeqLoop::new_main("view",self.n_reps);

        let p_exc = sinc5(
            Time::us(self.rf_dur_us),
            rf_dt,
            Nuc1H
        ).to_shared();

        // let p_ref = hardpulse(
        //     Time::us(self.exc_dur_us),
        //     rf_dt,
        //     Nuc1H
        // ).to_shared();

        let p_ref = sinc5(
            Time::us(self.rf_dur_us),
            rf_dt,
            Nuc1H
        ).to_shared();

        let n_ramp_samp = (Time::us(self.ramp_time_us) / grad_dt).si().round() as usize;
        // let w_ru = Waveform::new().ramp(0,1,n_ramp_samp,grad_dt);
        // let w_rd = Waveform::new().ramp(1,0,n_ramp_samp,grad_dt);

        let n_ss_samp = (Time::us(self.rf_dur_us * 3) / grad_dt).si().round() as usize;
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

        let e_ss = GradEvent::new("ss").with_x(&w_ss).with_strength_x(&c_ss_grad);
        let e_sr = GradEvent::new("ss_ref").with_x(&w_ss_ref).with_strength_x(&c_prephase);
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

        vl.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();

        vl
    }
}



pub struct RfCal {

    n_reps: usize,
    n_samples: usize,
    bandwidth_hz: f64,
    tau_1_ms: f64,
    tau_2_ms: f64,
    rf_dur_us: usize,
    grad_stab_time_ms: f64,
    rep_time_ms: f64,
}


impl Default for RfCal {
    fn default() -> Self {
        RfCal {
            n_reps: 10,
            n_samples: 256,
            bandwidth_hz: 50_000.,
            tau_1_ms: 10.0,
            tau_2_ms: 2.,
            rf_dur_us: 500,
            grad_stab_time_ms: 2.,
            rep_time_ms: 500.,
        }
    }
}

impl PulseSequence for RfCal {
    fn compile(&self) -> SeqLoop {

        let bw = Freq::hz(self.bandwidth_hz);
        let grad_stab_time = Time::ms(self.grad_stab_time_ms);
        let tau = Time::ms(self.tau_1_ms);
        let t_fill = Time::ms(self.tau_2_ms);
        let rep_time = Time::ms(self.rep_time_ms);

        let rf_dur = Time::us(self.rf_dur_us);
        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);

        // event controllers
        let c_gx = EventControl::<FieldGrad>::new()
            .with_adj("slice_x")
            .to_shared();

        let c_gy = EventControl::<FieldGrad>::new()
            .with_adj("slice_y")
            .to_shared();

        let c_gz = EventControl::<FieldGrad>::new()
            .with_adj("slice_z")
            .to_shared();

        let c_rf = EventControl::<f64>::new()
            .with_adj("rf_pow")
            .to_shared();

        let c_rf_phase = EventControl::<Angle>::new()
            .with_adj("rf_phase")
            .to_shared();

        // RF pulses
        //let n_alpha_samples = (rf_dur / rf_dt).si() as usize;

        //let alpha_pulse = sinc5(rf_dur,rf_dt,Nuc1H).to_shared();
        let alpha_pulse = hardpulse(rf_dur,rf_dt,Nuc1H).to_shared();

        // let alpha_pulse = RfPulse::new(
        //     &Waveform::new().add_list_r(
        //         &vec![1.;n_alpha_samples],
        //         rf_dt
        //     ).to_shared(),
        //     3.,
        //     Nuc1H
        // ).to_shared();

        // Grad waveforms
        let ramp_up = Waveform::new()
            .ramp(0,1,100,grad_dt).to_shared();
        let ramp_down = Waveform::new()
            .ramp(1,0,100,grad_dt).to_shared();

        // Grad events
        let ru = GradEvent::new("ru")
            .with_x(&ramp_up)
            .with_y(&ramp_up)
            .with_z(&ramp_up)
            .with_strength_x(&c_gx)
            .with_strength_y(&c_gy)
            .with_strength_z(&c_gz);

        let rd = GradEvent::new("rd")
            .with_x(&ramp_down)
            .with_y(&ramp_down)
            .with_z(&ramp_down)
            .with_strength_x(&c_gx)
            .with_strength_y(&c_gy)
            .with_strength_z(&c_gz);

        // RF events
        let alpha_1 = RfEvent::new("alpha_1",&alpha_pulse,&c_rf);
        let alpha_2 = RfEvent::new("alpha_2",&alpha_pulse,&c_rf).with_phase(&c_rf_phase);
        let alpha_3 = RfEvent::new("alpha_3",&alpha_pulse,&c_rf);

        // ACQ events
        let acq_1 = ACQEvent::new("acq_1",self.n_samples,bw.inv());
        let acq_2 = ACQEvent::new("acq_2",self.n_samples,bw.inv());

        // event timing
        let mut vl = SeqLoop::new_main("view",self.n_reps);
        vl.add_event(ru).unwrap();
        vl.add_event(alpha_1).unwrap();
        vl.add_event(alpha_2).unwrap();
        vl.add_event(acq_1).unwrap();
        vl.add_event(alpha_3).unwrap();
        vl.add_event(acq_2).unwrap();
        vl.add_event(rd).unwrap();

        // end uf ramp up to start of first rf pulse is 200 us
        vl.set_time_span("ru","alpha_1",100,0,grad_stab_time).unwrap();

        // time difference between center of first rf to second rf pulse is tau
        vl.set_time_span("alpha_1","alpha_2",50,50,tau).unwrap();

        // time difference between second rf pulse and spin echo is tau
        vl.set_time_span("alpha_2","acq_1",50,50,tau).unwrap();

        // time from spin echo to third rf pulse is t_fill
        vl.set_time_span("acq_1","alpha_3",100,0,t_fill).unwrap();

        // time from third pulse to stim echo is tau
        vl.set_time_span("alpha_3","acq_2",50,50,tau).unwrap();

        vl.set_min_time_span("acq_2","rd",100,0,Time::us(200)).unwrap();

        vl.set_pre_calc(Time::ms(3));

        vl.set_rep_time(rep_time).unwrap();

        vl


    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut state = HashMap::new();
        state.insert("slice_x".to_string(), 0.2);
        state.insert("slice_y".to_string(), 0.0);
        state.insert("slice_z".to_string(), 0.0);
        state.insert("rf_pow".to_string(), 1.0);
        state.insert("rf_phase".to_string(), 0.0);
        state
    }
}
