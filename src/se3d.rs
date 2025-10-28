use std::collections::HashMap;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, SpatFreq, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use seq_struct::waveform::Waveform;
use crate::{PulseSequence, SliceProfileSE};
use crate::rf_pulses::{hardpulse, sinc3, sinc5};

pub struct SpinEcho2D {
    rf_dur_us: usize,
    ramp_time_us: usize,
    crush_dur_us: usize,
    bw_hz: f64,
    n_phase: usize,
    fov_mm: f64,
    n_samples: usize,
    te_ms: f64,
    rep_time_ms: f64,
    slice_thickness_mm: f64,
}

impl Default for SpinEcho2D {
    fn default() -> SpinEcho2D {
        SpinEcho2D {
            rf_dur_us: 3000,
            ramp_time_us: 200,
            crush_dur_us: 300,
            bw_hz: 100_000.,
            fov_mm: 20.,
            n_samples: 256,
            n_phase: 256,
            te_ms: 30.,
            rep_time_ms: 500.,
            slice_thickness_mm: 1.,
        }
    }
}

impl PulseSequence for SpinEcho2D {
    fn compile(&self) -> SeqLoop {

        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);

        let nuc = Nuc1H;

        let p_rf = sinc3(
            Time::us(self.rf_dur_us),
            rf_dt,
            nuc
        ).to_shared();

        let bw = Freq::hz(self.bw_hz);
        let t_dwell = bw.inv();
        let acq_time = t_dwell.scale(self.n_samples).scale(1.1);

        let n_ramp_samp = (Time::us(self.ramp_time_us) / grad_dt).si().round() as usize;
        let n_ss_samp = (Time::us(self.rf_dur_us * 2) / grad_dt).si().round() as usize;
        let n_ro_samp = ( acq_time / grad_dt ).si().ceil() as usize;
        let n_crush_samp = (Time::us(self.crush_dur_us) / grad_dt).si().round() as usize;

        let gs_ro = FieldGrad::from_fov(Length::mm(self.fov_mm),t_dwell,nuc);
        let ro_ratio = grad_dt.scale(n_ramp_samp + n_ro_samp) / grad_dt.scale(n_ramp_samp + n_ss_samp);
        assert!(ro_ratio.is_unitless());
        let gs_prephase = gs_ro.scale( ro_ratio.si() / 2.);

        let gs_slice = p_rf.grad_strength(Length::mm(self.slice_thickness_mm));
        let gs_slice_refocus = gs_slice.scale( - 0.5);

        // phase encoding step size
        let gs_phase1 = FieldGrad::from_fov(Length::mm(self.fov_mm),grad_dt.scale(n_ramp_samp + n_ss_samp),nuc);

        let w_ss = Waveform::new().ramp(0,1,n_ramp_samp,grad_dt)
            .constant(1,n_ss_samp,grad_dt)
            .ramp(1,0,n_ramp_samp,grad_dt)
            .to_shared();

        let w_crush = Waveform::new()
            .ramp(0,1,n_ramp_samp,grad_dt)
            .constant(1,n_crush_samp,grad_dt)
            .ramp(1,0,n_ramp_samp,grad_dt)
            .to_shared();

        let w_ro = Waveform::new().ramp(0,1,n_ramp_samp,grad_dt)
            .constant(1,n_ro_samp,grad_dt)
            .ramp(1,0,n_ramp_samp,grad_dt)
            .to_shared();

        let c_exc_pow = EventControl::<f64>::new().with_adj("exc_pow").to_shared();
        let c_ref_pow = EventControl::<f64>::new().with_adj("ref_pow").to_shared();

        let c_ss_grad = EventControl::<FieldGrad>::new().with_constant_grad(gs_slice).to_shared();
        let c_ss_ref_grad = EventControl::<FieldGrad>::new().with_constant_grad(gs_slice_refocus).with_adj("ss_ref").to_shared();

        let c_crush = EventControl::<FieldGrad>::new().with_adj("crush").to_shared();
        let c_prephase = EventControl::<FieldGrad>::new().with_constant_grad(gs_prephase).with_adj("prephase").to_shared();
        let c_ro = EventControl::<FieldGrad>::new().with_constant_grad(
            gs_ro
        ).to_shared();

        let k_min = - (self.n_phase as i32 / 2);
        let lut = (k_min..k_min+self.n_phase as i32).collect::<Vec<i32>>();
        //let lut = vec![0;self.n_phase];
        let lut_pe = LUT::new("pe",&lut).to_shared();

        let c_pe1 = EventControl::<FieldGrad>::new().with_lut(&lut_pe).with_source_loop("view").with_grad_scale(gs_phase1).to_shared();

        let c_180_phase = EventControl::<Angle>::new().with_constant(Angle::deg(90)).to_shared();
        let e_exc = RfEvent::new("exc",&p_rf,&c_exc_pow);
        let e_ref = RfEvent::new("ref",&p_rf,&c_ref_pow).with_phase(&c_180_phase);

        let e_ss = GradEvent::new("ss").with_z(&w_ss).with_strength_z(&c_ss_grad);
        let e_sr = GradEvent::new("ss_ref").with_x(&w_ss).with_z(&w_ss)
            .with_strength_x(&c_prephase).with_strength_z(&c_ss_ref_grad);
        let e_crush_left = GradEvent::new("crush_l").with_x(&w_crush).with_strength_x(&c_crush);
        let e_crush_right = GradEvent::new("crush_r").with_x(&w_crush).with_strength_x(&c_crush);

        let e_pe = GradEvent::new("pe").with_y(&w_ss).with_strength_y(&c_pe1);

        let e_ro = GradEvent::new("ro").with_x(&w_ro).with_strength_x(&c_ro);
        let e_acq = ACQEvent::new("acq",self.n_samples,t_dwell);

        // single main loop structure
        let mut vl = SeqLoop::new_main("view",self.n_phase);

        vl.add_event(e_ss).unwrap();
        vl.add_event(e_exc).unwrap();
        vl.add_event(e_sr).unwrap();
        vl.add_event(e_crush_left).unwrap();
        vl.add_event(e_ref).unwrap();
        vl.add_event(e_crush_right).unwrap();
        vl.add_event(e_pe).unwrap();
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
        vl.set_min_time_span("pe","ro",100,0,Time::us(200)).unwrap();
        vl.set_time_span("exc","ref",50,50,tau).unwrap();
        vl.set_time_span("ref","ro",50,50,tau).unwrap();
        vl.set_time_span("ref","acq",50,50,tau).unwrap();

        vl.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();

        vl
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        todo!()
    }
}