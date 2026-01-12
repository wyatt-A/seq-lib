use std::collections::HashMap;
use std::rc::Rc;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use mrs_ppl::compile::compile_seq;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::{Orientation, SeqLoop};
use seq_struct::waveform::Waveform;
use seq_lib::grad_pulses::{ramp_down, ramp_up, trapezoid};
use seq_lib::PulseSequence;
use seq_lib::rf_pulses::sinc3;

type W = Rc<Waveform>;
type Gradient = Rc<EventControl<FieldGrad>>;
type Power = Rc<EventControl<f64>>;
type Phase = Rc<EventControl<Angle>>;
type RF = Rc<RfPulse>;

const EXC:&str = "exc_pulse";
const REF:&str = "ref_pulse";
const EXC_SS_RU:&str = "exc_ss_ru";
const EXC_SS_RD:&str = "exc_ss_rd";
const REF_SS_RU:&str = "ref_ss_ru";
const REF_SS_RD:&str = "ref_ss_rd";
const RO_RU:&str = "ro_ru";
const RO_RD:&str = "ro_rd";
const PE: &str = "pe";
const ACQ: &str = "acq";


fn main() {
    let sp = SliceProfile::default();
    let p = sp.adjustment_state();
    sp.render_to_file(&p,"slice_profile");
}

struct SliceProfile {
    n_samples: usize,
    fov_mm: f64,
    slice_thickness_mm: f64,
    bandwidth_khz: f64,
    pulse_duration_us: usize,
    n_reps: usize,
    rep_time_ms: f64,
    ramp_time_us: usize,
    pe_time_ms: f64,
}

impl Default for SliceProfile {
    fn default() -> Self {
        Self {
            n_samples: 256,
            fov_mm: 10.0,
            slice_thickness_mm: 0.5,
            bandwidth_khz: 100.,
            pulse_duration_us: 200,
            n_reps: 10,
            rep_time_ms: 500.,
            ramp_time_us: 500,
            pe_time_ms: 1.0,
        }
    }
}

struct Waveforms {
    rf_pulse: RF,
    ru: W,
    rd: W,
    pe: W,
}

impl Waveforms {
    pub fn new(params:&SliceProfile) -> Self {
        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);
        let t_ramp = Time::us(params.ramp_time_us);
        let rf_pulse = sinc3(Time::us(params.pulse_duration_us),rf_dt,Nuc1H).to_shared();
        let ru = ramp_up(t_ramp,grad_dt).to_shared();
        let rd = ramp_down(t_ramp,grad_dt).to_shared();
        let pe = trapezoid(t_ramp,Time::ms(params.pe_time_ms),grad_dt).to_shared();
        Self {
            rf_pulse,
            ru,
            rd,
            pe,
        }
    }
}


struct EventControllers {
    exc_pow: Power,
    ref_pow: Power,
    ref_phase: Phase,
    read: Gradient,
    phase_encode: Gradient,
    slice_select: Gradient,
}

impl EventControllers {
    pub fn new(params:&SliceProfile,w:&Waveforms) -> Self {

        let exc_pow = EventControl::<f64>::new().with_constant(1.).to_shared();
        let ref_pow = EventControl::<f64>::new().with_constant(2.).to_shared();
        let ref_phase = EventControl::<Angle>::new().with_constant(Angle::deg(90)).to_shared();

        let t_dwell = Freq::khz(params.bandwidth_khz).inv();

        let gro = FieldGrad::from_fov(
            Length::mm(params.fov_mm),
            t_dwell,
            Nuc1H,
        );

        let read = EventControl::<FieldGrad>::new().with_constant_grad(
            gro
        ).to_shared();

        let acq_time = t_dwell.scale(params.n_samples);

        let scale = (acq_time / Time::ms(params.pe_time_ms)).si();

        let phase_encode = EventControl::<FieldGrad>::new().with_constant_grad(
            gro.scale(scale * 0.5)
        ).to_shared();

        // let slice_select = EventControl::<FieldGrad>::new().with_constant_grad(
        //     w.rf_pulse.grad_strength(
        //         Length::mm(params.slice_thickness_mm)
        //     )
        // ).to_shared();

        let slice_select = EventControl::<FieldGrad>::new().with_constant_grad(
            FieldGrad::default().scale(0)
        ).to_shared();

        Self {
            exc_pow,
            ref_pow,
            ref_phase,
            read,
            phase_encode,
            slice_select,
        }

    }
}

struct Events {

    exc_pulse: RfEvent,
    ref_pulse: RfEvent,

    exc_ss_ru: GradEvent,
    exc_ss_rd: GradEvent,

    ref_ss_ru: GradEvent,
    ref_ss_rd: GradEvent,

    ro_ru: GradEvent,
    ro_rd: GradEvent,

    pe: GradEvent,

    acq: ACQEvent,

}

impl Events {
    pub fn new(params:&SliceProfile,w:&Waveforms,ec:&EventControllers) -> Self {

        let exc_pulse = RfEvent::new(EXC,&w.rf_pulse,&ec.exc_pow);
        let ref_pulse = RfEvent::new(REF,&w.rf_pulse,&ec.ref_pow).with_phase(&ec.ref_phase);

        let exc_ss_ru = GradEvent::new(EXC_SS_RU).with_x(&w.ru).with_strength_x(&ec.slice_select);
        let exc_ss_rd = GradEvent::new(EXC_SS_RD).with_x(&w.rd).with_strength_x(&ec.slice_select);

        let ref_ss_ru = GradEvent::new(REF_SS_RU).with_x(&w.ru).with_strength_x(&ec.slice_select);
        let ref_ss_rd = GradEvent::new(REF_SS_RD).with_x(&w.rd).with_strength_x(&ec.slice_select);

        let ro_ru = GradEvent::new(RO_RU).with_x(&w.ru).with_strength_x(&ec.read);
        let ro_rd = GradEvent::new(RO_RD).with_x(&w.rd).with_strength_x(&ec.read);

        let pe = GradEvent::new(PE).with_x(&w.pe).with_strength_x(&ec.phase_encode);
        let acq = ACQEvent::new(ACQ,params.n_samples,Freq::khz(params.bandwidth_khz).inv());

        Self {
            exc_pulse,
            ref_pulse,
            exc_ss_ru,
            exc_ss_rd,
            ref_ss_ru,
            ref_ss_rd,
            ro_ru,
            ro_rd,
            pe,
            acq,
        }

    }
}


impl PulseSequence for SliceProfile {
    fn compile(&self) -> SeqLoop {

        let tau = Time::ms(4);

        let w = Waveforms::new(self);
        let ec = EventControllers::new(self, &w);
        let events = Events::new(self,&w,&ec);

        let mut vl = SeqLoop::new_main("view",self.n_reps);

        vl.add_event(events.exc_ss_ru).unwrap();
        vl.add_event(events.exc_pulse).unwrap();
        vl.add_event(events.exc_ss_rd).unwrap();
        vl.add_event(events.pe).unwrap();
        vl.add_event(events.ref_ss_ru).unwrap();
        vl.add_event(events.ref_pulse).unwrap();
        vl.add_event(events.ref_ss_rd).unwrap();
        vl.add_event(events.ro_ru).unwrap();
        vl.add_event(events.acq).unwrap();
        vl.add_event(events.ro_rd).unwrap();

        vl.set_time_span(EXC_SS_RU,EXC,100,0,Time::us(100)).unwrap();
        vl.set_time_span(EXC,EXC_SS_RD,100,0,Time::us(100)).unwrap();
        vl.set_time_span(EXC_SS_RD,PE,100,0,Time::us(500)).unwrap();

        vl.set_time_span(REF_SS_RU,REF,100,0,Time::us(100)).unwrap();
        vl.set_time_span(REF,REF_SS_RD,100,0,Time::us(100)).unwrap();

        vl.set_time_span(RO_RU,ACQ,100,0,Time::us(300)).unwrap();
        vl.set_min_time_span(ACQ,RO_RD,100,0,Time::us(300)).unwrap();

        vl.set_time_span(EXC,REF,50,50,tau).unwrap();
        vl.set_time_span(REF,ACQ,50,50,tau).unwrap();

        vl.set_pre_calc(Time::ms(2));
        vl.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();

        let mut el = SeqLoop::new("experiment",1);
        el.orientation = Some(Orientation::new(&[[Angle::deg(90),Angle::deg(0),Angle::deg(0)]]));

        el.add_loop(vl).unwrap();

        el
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}