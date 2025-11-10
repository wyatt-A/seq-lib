use std::collections::HashMap;
use std::f64::consts::PI;
use std::rc::Rc;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use seq_struct::waveform::{linspace, Waveform};
use crate::grad_pulses::{half_sin, ramp_up, trapezoid};
use crate::PulseSequence;
use crate::rf_pulses::sinc3;

pub struct SSMEParams {

    n_samples: usize,
    n_phase: usize,
    n_echoes: usize,
    bw_hz: f64,

    fov_read_mm: f64,
    fov_phase_mm: f64,
    slice_thickness_mm: f64,

    te_ms: f64,
    rep_time_ms: f64,

    rf_dur_us: usize,
    ramp_time_us: usize,
    crush_dur_us: usize,
    pe_time_us: usize,
    ss_ref_time_us: usize,

}

struct EventControllers {
    c_ro: Rc<EventControl<FieldGrad>>,
    c_gr_crush: Rc<EventControl<FieldGrad>>,
    c_gp_crush: Rc<EventControl<FieldGrad>>,
    c_gs_crush: Rc<EventControl<FieldGrad>>,
    c_ss: Rc<EventControl<FieldGrad>>,
    c_pe_pos: Rc<EventControl<FieldGrad>>,
    c_pe_neg: Rc<EventControl<FieldGrad>>,
    c_pre_ro: Rc<EventControl<FieldGrad>>,
    c_ss_ref: Rc<EventControl<FieldGrad>>,
    c_rf90: Rc<EventControl<f64>>,
    c_rf180: Rc<EventControl<f64>>,
    c_rf180_phase: Rc<EventControl<Angle>>,
}

impl EventControllers {
    pub fn new(params: &SSMEParams,w:&Waveforms) -> Self {

        // readout gradient strength
        let g_ro = FieldGrad::from_fov(
            Length::mm(params.fov_read_mm),
            Freq::hz(params.bw_hz).inv(),
            Nuc1H
        );

        // calculate the characteristic phase encode time
        let pe_time = w.phase_encode.integrate().0;

        // phase encode gradient step size
        let g_pe = FieldGrad::from_fov(
            Length::mm(params.fov_phase_mm),
            pe_time,
            Nuc1H
        );

        let gr_crush = g_ro;
        let gp_crush = g_ro;
        let gs_crush = g_ro;

        // slice selection gradient from rf pulse
        let g_ss = w.rf_pulse.grad_strength(
            Length::mm(params.slice_thickness_mm),
        );

        // slice selection refocus
        let t = Time::us(params.ramp_time_us + params.rf_dur_us);
        let ss_ref_time = w.slice_refocus.integrate().0;
        let frac = t.si() / ss_ref_time.si();
        let g_ss_ref = g_ss.scale(- 0.5 * frac);

        // phase encoding
        let pe_start = - (params.n_phase as i32)/2;
        let pe_tab = (pe_start..(pe_start + params.n_phase as i32)).collect::<Vec<i32>>();
        println!("pe_tab = {:?}",pe_tab);
        let lut_pe = LUT::new("pe",&pe_tab).to_shared();

        // readout pre-phase
        let ro_moment = Freq::hz(params.bw_hz).inv().scale(params.n_samples as f64);
        let frac = ro_moment.si() / w.phase_encode.integrate().0.si();
        let g_pre_ro = g_ro.scale( 0.5 * frac);

        let c_ro = EventControl::<FieldGrad>::new().with_constant_grad(g_ro).to_shared();
        let c_ss = EventControl::<FieldGrad>::new().with_constant_grad(g_ss).to_shared();
        let c_gr_crush = EventControl::<FieldGrad>::new().with_constant_grad(gr_crush).to_shared();
        let c_gp_crush = EventControl::<FieldGrad>::new().with_constant_grad(gp_crush).to_shared();
        let c_gs_crush = EventControl::<FieldGrad>::new().with_constant_grad(gs_crush).to_shared();
        let c_ss_ref = EventControl::<FieldGrad>::new().with_constant_grad(g_ss_ref).to_shared();
        let c_pe_pos = EventControl::<FieldGrad>::new().with_source_loop("view").with_lut(&lut_pe).with_grad_scale(g_pe).to_shared();
        let c_pe_neg = EventControl::<FieldGrad>::new().with_source_loop("view").with_lut(&lut_pe).with_grad_scale(g_pe.scale(-1)).to_shared();
        let c_pre_ro = EventControl::<FieldGrad>::new().with_constant_grad(g_pre_ro).to_shared();

        let c_rf90 = EventControl::<f64>::new().with_adj("rf90").to_shared();
        let c_rf180 = EventControl::<f64>::new().with_adj("rf180").to_shared();
        let c_rf180_phase = EventControl::<Angle>::new().with_constant(Angle::deg(90)).to_shared();

        EventControllers {
            c_ro,
            c_gr_crush,
            c_gp_crush,
            c_gs_crush,
            c_ss,
            c_ss_ref,
            c_pe_pos,
            c_pe_neg,
            c_pre_ro,
            c_rf90,
            c_rf180,
            c_rf180_phase
        }
    }
}


struct Waveforms {
    crusher: Rc<Waveform>,
    ramp_up: Rc<Waveform>,
    ramp_down: Rc<Waveform>,
    phase_encode: Rc<Waveform>,
    slice_refocus: Rc<Waveform>,
    rf_pulse: Rc<RfPulse>,
}

impl Waveforms {
    pub fn new(params:&SSMEParams) -> Waveforms {

        let grad_dt = Time::us(2);
        let rf_dt = Time::us(2);

        let ramp_up = ramp_up(
            Time::us(params.ramp_time_us),
            grad_dt
        ).to_shared();

        let ramp_down = crate::grad_pulses::ramp_down(
            Time::us(params.ramp_time_us),
            grad_dt
        ).to_shared();

        let crusher = half_sin(
            Time::us(params.crush_dur_us),
            grad_dt
        ).to_shared();

        let phase_encode = trapezoid(
            Time::us(params.ramp_time_us),
            Time::us(params.pe_time_us),
            grad_dt
        ).to_shared();

        let slice_refocus = trapezoid(
            Time::us(params.ramp_time_us),
            Time::us(params.ss_ref_time_us),
            grad_dt
        ).to_shared();

        let rf_pulse = sinc3(
            Time::us(params.rf_dur_us),
            rf_dt,
            Nuc1H
        ).to_shared();

        Waveforms {
            crusher,
            ramp_up,
            ramp_down,
            phase_encode,
            slice_refocus,
            rf_pulse,
        }


    }
}



impl Default for SSMEParams {
    fn default() -> SSMEParams {
        SSMEParams {
            n_samples: 512,
            n_phase: 5,
            n_echoes: 4,
            bw_hz: 100_000.,
            fov_read_mm: 20.0,
            fov_phase_mm: 20.0,
            slice_thickness_mm: 1.0,
            te_ms: 20.0,
            rep_time_ms: 4000.0,
            rf_dur_us: 1000,
            ramp_time_us: 150,
            crush_dur_us: 500,
            pe_time_us: 500,
            ss_ref_time_us: 500,
        }
    }
}

impl PulseSequence for SSMEParams {
    fn compile(&self) -> SeqLoop {

        let waveforms = Waveforms::new(&self);
        let event_controllers = EventControllers::new(&self,&waveforms);

        let e_ss_ru = GradEvent::new("ss_ru")
            .with_z(&waveforms.ramp_up)
            .with_strength_z(&event_controllers.c_ss);

        let e_ss_rd = GradEvent::new("ss_rd")
            .with_z(&waveforms.ramp_down)
            .with_strength_z(&event_controllers.c_ss);

        let e_sr_ru = GradEvent::new("sr_ru")
            .with_z(&waveforms.ramp_up)
            .with_strength_z(&event_controllers.c_ss);

        let e_sr_rd = GradEvent::new("sr_rd")
            .with_z(&waveforms.ramp_down)
            .with_strength_z(&event_controllers.c_ss);

        let e_ss_ref = GradEvent::new("ss_ref")
            .with_z(&waveforms.slice_refocus)
            .with_strength_z(&event_controllers.c_ss_ref);

        let e_pre_ro = GradEvent::new("pre_ro")
            .with_x(&waveforms.phase_encode)
            .with_strength_x(&event_controllers.c_pre_ro);

        let e_crush_left = GradEvent::new("crush_left")
            .with_x(&waveforms.crusher)
            .with_y(&waveforms.crusher)
            .with_z(&waveforms.crusher)
            .with_strength_x(&event_controllers.c_gr_crush)
            .with_strength_y(&event_controllers.c_gp_crush)
            .with_strength_z(&event_controllers.c_gs_crush);

        let e_crush_right = GradEvent::new("crush_right")
            .with_x(&waveforms.crusher)
            .with_y(&waveforms.crusher)
            .with_z(&waveforms.crusher)
            .with_strength_x(&event_controllers.c_gr_crush)
            .with_strength_y(&event_controllers.c_gp_crush)
            .with_strength_z(&event_controllers.c_gs_crush);

        let e_pe_pos = GradEvent::new("pe_pos")
            .with_y(&waveforms.phase_encode)
            .with_strength_y(&event_controllers.c_pe_pos);

        let e_pe_neg = GradEvent::new("pe_neg")
            .with_y(&waveforms.phase_encode)
            .with_strength_y(&event_controllers.c_pe_neg);

        let e_ro_ru = GradEvent::new("ro_ru")
            .with_x(&waveforms.ramp_up)
            .with_strength_x(&event_controllers.c_ro);

        let e_ro_rd = GradEvent::new("ro_rd")
            .with_x(&waveforms.ramp_down)
            .with_strength_x(&event_controllers.c_ro);

        let e_rf90 = RfEvent::new("rf90",&waveforms.rf_pulse,&event_controllers.c_rf90);
        let e_rf180 = RfEvent::new("rf180",&waveforms.rf_pulse,&event_controllers.c_rf180)
            .with_phase(&event_controllers.c_rf180_phase);

        let e_acq = ACQEvent::new("acq",self.n_samples,Freq::hz(self.bw_hz).inv());

        let mut el = SeqLoop::new("echo",self.n_echoes);
        el.set_pre_calc(Time::ms(2));

        el.add_event(e_crush_left).unwrap();
        el.add_event(e_sr_ru).unwrap();
        el.add_event(e_rf180).unwrap();
        el.add_event(e_sr_rd).unwrap();
        el.add_event(e_crush_right).unwrap();
        el.add_event(e_pe_pos).unwrap();
        el.add_event(e_ro_ru).unwrap();
        el.add_event(e_acq).unwrap();
        el.add_event(e_ro_rd).unwrap();
        el.add_event(e_pe_neg).unwrap();

        el.set_time_span("crush_left","sr_ru", 100, 0, Time::us(2)).unwrap();
        el.set_time_span("sr_ru","rf180", 100, 0, Time::us(100)).unwrap();
        el.set_time_span("rf180","sr_rd", 100, 0, Time::us(100)).unwrap();
        el.set_time_span("sr_rd","crush_right", 100, 0, Time::us(2)).unwrap();

        el.set_time_span("pe_pos","ro_ru", 100, 0, Time::us(2)).unwrap();
        el.set_min_time_span("ro_ru","acq", 100, 0, Time::us(300)).unwrap(); // echo balance
        el.set_min_time_span("acq","ro_rd", 100, 0, Time::us(2)).unwrap();
        el.set_min_time_span("ro_rd","pe_neg", 100, 0, Time::us(2)).unwrap();

        el.set_min_time_span("rf180","acq",50,50,Time::ms(self.te_ms).scale(0.5)).unwrap();
        el.set_rep_time(Time::ms(self.te_ms)).unwrap();

        let mut vl = SeqLoop::new_main("view",self.n_phase);
        vl.add_event(e_ss_ru).unwrap();
        vl.add_event(e_rf90).unwrap();
        vl.add_event(e_ss_rd).unwrap();
        vl.add_event(e_ss_ref).unwrap();
        vl.add_event(e_pre_ro).unwrap();

        vl.set_pre_calc(Time::ms(3));
        vl.add_loop(el).unwrap();

        vl.set_time_span("ss_ru","rf90",100, 0, Time::us(2)).unwrap();
        vl.set_time_span("rf90","ss_rd",100, 0, Time::us(100)).unwrap();
        vl.set_time_span("ss_rd","ss_ref",100, 0, Time::us(2)).unwrap();
        vl.set_time_span("ss_ref","pre_ro",100, 0, Time::us(2)).unwrap();

        vl.set_time_span("rf90","rf180",50,50, Time::ms(self.te_ms).scale(0.5)).unwrap();

        vl.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();

        vl

    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        HashMap::from(
            [("rf90".to_string(),0.5),
            ("rf180".to_string(),1.)]
        )
    }
}