use std::collections::HashMap;
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
use seq_struct::waveform::Waveform;
use crate::grad_pulses::{ramp_down, ramp_up, trapezoid};
use crate::PulseSequence;
use crate::q_calc::{calc_b_matrix, BMat};
use crate::rf_pulses::{hardpulse, hardpulse_composite, sinc3};

pub struct Dwi3DParams {
    pub diffusion_ramp_time_us: u32,
    pub diffusion_const_time_ms: f64,
    pub ramp_time_us: u32,
    pub pe_time_ms: f64,
    pub fov_x_mm: f64,
    pub fov_y_mm: f64,
    pub fov_z_mm: f64,
    pub n_x: usize,
    pub n_y: usize,
    pub n_z: usize,
    pub rep_time_ms: f64,
    pub bw_hz:f64,
    pub n_views: usize,
}

impl Default for Dwi3DParams {
    fn default() -> Self {
        Dwi3DParams {
            diffusion_ramp_time_us: 400,
            diffusion_const_time_ms: 2.,
            ramp_time_us: 200,
            pe_time_ms: 0.2,
            fov_x_mm: 19.7,
            fov_y_mm: 12.8,
            fov_z_mm: 12.8,
            n_x: 394,
            n_y: 256,
            n_z: 256,
            rep_time_ms: 100.,
            n_views: 1,
            bw_hz: 100_000.,
        }
    }
}

impl Dwi3DParams {

    pub fn calc_b_matrix(&self, adj_state:&HashMap<String,f64>, echo_idx:usize) -> BMat {

        // get view loop
        let mut vl = self.view_loop();
        // set iterations to 1 for a single view
        vl.set_count(1);

        let seq = vl.render_timeline(&adj_state).render();

        // time points where phase is inverted (center of rf180)
        let mut t_inv:Vec<_> = vl.find_occurrences("rf180",50)
            .into_iter().map(|t|t.as_sec()).collect();

        let t_inv_2:Vec<_> = vl.find_occurrences("rf180_2",50)
            .into_iter().map(|t|t.as_sec()).collect();

        t_inv.extend(t_inv_2);

        println!("t_inv = {:?}", t_inv);

        // assumed echo time
        // let t_echo = vl.find_occurrences("acq",50)
        //     .get(0).unwrap().as_sec();

        let t_echo = if echo_idx == 0 {
            vl.find_occurrences("acq",50).get(0).unwrap().as_sec()
        }else {
            vl.find_occurrences("acq_2",50).get(echo_idx - 1).unwrap().as_sec()
        };

        // performs numerical integration to calculate b-matrix
        calc_b_matrix(
                &seq.time_sec,
                &seq.gx_tpm,
                &seq.gy_tpm,
                &seq.gz_tpm,
                &t_inv,
                t_echo,
                Nuc1H,
        )

    }




    fn view_loop(&self) -> SeqLoop {
        let w = Waveforms::new(&self);
        let ec = EventControllers::new(&self);
        let events = Events::new(&self,&w,&ec);

        let mut vl = SeqLoop::new_main("view",self.n_views);

        vl.add_event(events.rf90).unwrap();
        vl.add_event(events.diff1).unwrap();
        vl.add_event(events.rf180).unwrap();
        vl.add_event(events.diff2).unwrap();
        vl.add_event(events.phase_enc).unwrap();
        vl.add_event(events.read_ru).unwrap();
        vl.add_event(events.acq).unwrap();
        vl.add_event(events.read_rd).unwrap();

        // time between end of rf90 and start of first diffusion pulse
        let del1 = Time::us(100);
        // time between end of rf180 and start of second diffusion pulse
        let del2 = Time::us(100);
        // time after second diffusion before phase encoding. This should be chosen minimize effect
        // of eddy currents
        let del3 = Time::ms(1);

        vl.set_time_span("rf90","diff1",100,0,del1).unwrap();
        vl.set_time_span("rf180","diff2",100,0,del2).unwrap();
        vl.set_min_time_span("diff2","pe",100,0,del3).unwrap();

        vl.set_time_span("pe","ru",100,0,Time::us(100)).unwrap();
        vl.set_time_span("ru","acq",100,0,Time::us(100)).unwrap();
        vl.set_min_time_span("acq","rd",100,0,Time::us(0)).unwrap();

        let tau = vl.get_time_span("rf180","acq",50,50).unwrap();

        if let Err(e) = vl.set_time_span("rf90","rf180",50,50,tau) {
            panic!("failed to set tau. You must lengthen the time between the inversion pulse and echo: {:?}",e);
        }




        let mut el = SeqLoop::new("echo",3);
        el.add_event(events.phase_enc_left).unwrap();
        el.add_event(events.rf180_2).unwrap();
        el.add_event(events.phase_enc_right).unwrap();
        el.add_event(events.read_ru_2).unwrap();
        el.add_event(events.acq_2).unwrap();
        el.add_event(events.read_rd_2).unwrap();


        el.set_time_span("pe_left","rf180_2",100,0, Time::us(100)).unwrap();
        el.set_time_span("rf180_2","pe_right",200,0, Time::us(100)).unwrap();
        el.set_time_span("pe_right","ru_2",100,0, Time::us(100)).unwrap();
        el.set_time_span("ru_2","acq_2",100,0,Time::us(100)).unwrap();
        el.set_min_time_span("acq_2","rd_2",100,0,Time::us(0)).unwrap();


        let tau2 = el.get_time_span("rf180_2","acq_2",50,50).unwrap();

        el.set_rep_time(tau2.scale(2)).unwrap();

        vl.set_pre_calc(Time::ms(2));
        vl.add_loop(el).unwrap();
        vl.set_time_span("rd","pe_left",100,0,Time::us(100)).unwrap();

        vl.set_rep_time(Time::ms(200)).unwrap();

        vl
    }

}



impl PulseSequence for Dwi3DParams {
    fn compile(&self) -> SeqLoop {
        self.view_loop()
    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        HashMap::from_iter(
            [
                ("read_prephase".to_string(), 0.),
                ("diff_x".to_string(), 0.0),
                ("diff_y".to_string(), 0.0),
                ("diff_z".to_string(), 0.0),
                ("rf90_pow".to_string(), 0.0),
                ("rf180_pow".to_string(), 0.0),
                ("rf180_phase".to_string(), 0.0),
                ("crush_left".to_string(), 0.0),
                ("crush_right".to_string(), 0.0),
            ]
        )
    }
}

struct Events {
    rf90: RfEvent,
    diff1: GradEvent,
    rf180: RfEvent,
    diff2: GradEvent,
    phase_enc: GradEvent,
    read_ru: GradEvent,
    acq: ACQEvent,
    read_rd: GradEvent,

    phase_enc_left: GradEvent,
    rf180_2: RfEvent,
    phase_enc_right: GradEvent,
    read_ru_2: GradEvent,
    acq_2: ACQEvent,
    read_rd_2: GradEvent,

}

impl Events {
    pub fn new(params:&Dwi3DParams, w:&Waveforms, ec: &EventControllers) -> Events {

        let rf90 = RfEvent::new("rf90",&w.rf_90,&ec.c_rf90);
        let rf180 = RfEvent::new("rf180",&w.rf_180,&ec.c_rf180).with_phase(&ec.c_rf180_phase);
        let rf180_2 = RfEvent::new("rf180_2",&w.rf_180,&ec.c_rf180).with_phase(&ec.c_rf180_phase);

        let diff1 = GradEvent::new("diff1")
            .with_x(&w.diffusion)
            .with_y(&w.diffusion)
            .with_z(&w.diffusion)
            .with_strength_x(&ec.c_diff_x)
            .with_strength_y(&ec.c_diff_y)
            .with_strength_z(&ec.c_diff_z);

        let diff2 = diff1.clone_with_label("diff2");

        let phase_enc = GradEvent::new("pe")
            .with_x(&w.phase_encode)
            .with_y(&w.phase_encode)
            .with_z(&w.phase_encode)
            .with_strength_x(&ec.c_pe_x)
            .with_strength_y(&ec.c_pe_y)
            .with_strength_z(&ec.c_pe_z);

        let read_ru = GradEvent::new("ru").with_x(&w.ramp_up).with_strength_x(&ec.c_ro);
        let read_rd = GradEvent::new("rd").with_x(&w.ramp_down).with_strength_x(&ec.c_ro);
        let acq = ACQEvent::new("acq",params.n_x,Freq::hz(params.bw_hz).inv());


        let read_ru_2 = GradEvent::new("ru_2").with_x(&w.ramp_up).with_strength_x(&ec.c_ro);
        let read_rd_2 = GradEvent::new("rd_2").with_x(&w.ramp_down).with_strength_x(&ec.c_ro);
        let acq_2 = ACQEvent::new("acq_2",params.n_x,Freq::hz(params.bw_hz).inv());

        let phase_enc_left = GradEvent::new("pe_left")
            .with_x(&w.phase_encode)
            .with_y(&w.phase_encode)
            .with_z(&w.phase_encode)
            .with_strength_x(&ec.c_crush_left)
            .with_strength_y(&ec.c_pe_ry)
            .with_strength_z(&ec.c_pe_rz);

        let phase_enc_right = GradEvent::new("pe_right")
            .with_x(&w.phase_encode)
            .with_y(&w.phase_encode)
            .with_z(&w.phase_encode)
            .with_strength_x(&ec.c_crush_right)
            .with_strength_y(&ec.c_pe_y)
            .with_strength_z(&ec.c_pe_z);

        Events {
            rf90,
            diff1,
            rf180,
            diff2,
            phase_enc,
            read_ru,
            acq,
            read_rd,
            phase_enc_left,
            rf180_2,
            phase_enc_right,
            read_ru_2,
            acq_2,
            read_rd_2,
        }
    }
}


struct Waveforms {
    diffusion: Rc<Waveform>,
    ramp_up: Rc<Waveform>,
    ramp_down: Rc<Waveform>,
    phase_encode: Rc<Waveform>,
    rf_90: Rc<RfPulse>,
    rf_180: Rc<RfPulse>,
}

impl Waveforms {
    pub fn new(params:&Dwi3DParams) -> Waveforms {

        let grad_dt = Time::us(2);
        let rf_dt = Time::us(2);

        // non-selective excitation and refocusing rf pulses
        let rf_90 = hardpulse(Time::us(100),rf_dt,Nuc1H).to_shared();
        //let rf_90 = sinc3(Time::us(100),rf_dt,Nuc1H).to_shared();
        let rf_180 = hardpulse_composite(Time::us(200),rf_dt,Nuc1H).to_shared();
        //let rf_180 = sinc3(Time::us(200),rf_dt,Nuc1H).to_shared();

        // diffusion encoding pulse
        let diffusion = trapezoid(Time::us(params.diffusion_ramp_time_us),Time::ms(params.diffusion_const_time_ms),grad_dt).to_shared();

        // ramp up / ramp down for readout gradient
        let ramp_up = ramp_up(Time::us(params.ramp_time_us),grad_dt).to_shared();
        let ramp_down = ramp_down(Time::us(params.ramp_time_us),grad_dt).to_shared();

        // phase encoding pulse
        let phase_encode = trapezoid(Time::us(params.ramp_time_us),Time::ms(params.pe_time_ms),grad_dt).to_shared();

        Waveforms {
            diffusion,
            ramp_up,
            ramp_down,
            phase_encode,
            rf_90,
            rf_180,
        }

    }
}

struct EventControllers {
    c_ro: Rc<EventControl<FieldGrad>>,
    c_diff_x: Rc<EventControl<FieldGrad>>,
    c_diff_y: Rc<EventControl<FieldGrad>>,
    c_diff_z: Rc<EventControl<FieldGrad>>,
    c_pe_x: Rc<EventControl<FieldGrad>>,
    c_pe_y: Rc<EventControl<FieldGrad>>,
    c_pe_z: Rc<EventControl<FieldGrad>>,

    c_pe_ry: Rc<EventControl<FieldGrad>>,
    c_pe_rz: Rc<EventControl<FieldGrad>>,

    c_rf90: Rc<EventControl<f64>>,
    c_rf180: Rc<EventControl<f64>>,
    c_rf180_phase: Rc<EventControl<Angle>>,

    c_crush_left: Rc<EventControl<FieldGrad>>,
    c_crush_right: Rc<EventControl<FieldGrad>>,
}

impl EventControllers {
    pub fn new(params:&Dwi3DParams) -> EventControllers {

        let g_ro = FieldGrad::from_fov(
            Length::mm(params.fov_x_mm),
            Freq::hz(params.bw_hz).inv(),
            Nuc1H,
        );

        let c_ro = EventControl::<FieldGrad>::new().with_constant_grad(g_ro).to_shared();

        //let mut f = File::open(r"C:\workstation\data\petableCS_stream\stream_CS256_8x_pa18_pb54").unwrap();
        // let mut f = File::open(r"stream_CS256_8x_pa18_pb54").unwrap();
        // let mut s = String::new();
        // f.read_to_string(&mut s).unwrap();
        // let values = s.lines().map(|line|line.parse::<i32>().unwrap()).collect::<Vec<i32>>();
        // let mut pe_y_vals = vec![];
        // let mut pe_z_vals = vec![];
        // values.chunks_exact(2).for_each(|x|{
        //     pe_y_vals.push(x[0]);
        //     pe_z_vals.push(x[1]);
        // });

        let pe_y_vals = vec![128;4];
        let pe_z_vals = vec![64;4];

        //let pe_y_vals = vec![0;params.n_views];
        //let pe_z_vals = vec![0;params.n_views];

        let lut_pe_y = LUT::new("pe_y",&pe_y_vals).to_shared();
        let lut_pe_z = LUT::new("pe_z",&pe_z_vals).to_shared();

        // characteristic time for the readout gradient
        let acq_time:Time = (
            Freq::hz(params.bw_hz).inv().scale(params.n_x).quantity() +
                Time::us(params.ramp_time_us).quantity())
            .try_into().unwrap();

        let pe_time = Time::ms(params.pe_time_ms + Time::us(params.ramp_time_us).as_ms());
        let ratio = (acq_time / pe_time).si();
        let c_pe_x = EventControl::<FieldGrad>::new().with_constant_grad(
            g_ro.scale( - ratio / 2.),
        )
            .with_adj("read_prephase")
            .to_shared();


        let pe_y_grad = FieldGrad::from_fov(
            Length::mm(params.fov_y_mm),
            pe_time,
            Nuc1H,
        );

        let c_pe_y = EventControl::<FieldGrad>::new().with_grad_scale(
            pe_y_grad
        )
            .with_source_loop("view")
            .with_lut(&lut_pe_y)
            .to_shared();

        let c_pe_ry = EventControl::<FieldGrad>::new().with_grad_scale(
            pe_y_grad.scale(-1)
        )
            .with_source_loop("view")
            .with_lut(&lut_pe_y)
            .to_shared();

        let pe_z_grad = FieldGrad::from_fov(
            Length::mm(params.fov_z_mm),
            pe_time,
            Nuc1H,
        );

        let c_pe_z = EventControl::<FieldGrad>::new().with_grad_scale(
            pe_z_grad
        )
            .with_source_loop("view")
            .with_lut(&lut_pe_z)
            .to_shared();

        let c_pe_rz = EventControl::<FieldGrad>::new().with_grad_scale(
            pe_z_grad.scale(-1)
        )
            .with_source_loop("view")
            .with_lut(&lut_pe_z)
            .to_shared();

        let c_crush_left = EventControl::<FieldGrad>::new()
            .with_constant_grad(FieldGrad::mt_per_meter(100))
            .with_adj("crush_left").to_shared();

        let c_crush_right = EventControl::<FieldGrad>::new()
            .with_constant_grad(FieldGrad::mt_per_meter(100))
            .with_adj("crush_right").to_shared();

        //let c_diff_x = EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(2360)).with_adj("diff_x").to_shared();
        let c_diff_x = EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0)).with_adj("diff_x").to_shared();
        let c_diff_y = EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0.)).with_adj("diff_y").to_shared();
        let c_diff_z = EventControl::<FieldGrad>::new().with_constant_grad(FieldGrad::mt_per_meter(0.)).with_adj("diff_z").to_shared();

        let c_rf90 = EventControl::<f64>::new().with_adj("rf90_pow").to_shared();
        let c_rf180 = EventControl::<f64>::new().with_adj("rf180_pow").to_shared();
        let c_rf180_phase = EventControl::<Angle>::new().with_adj("rf180_phase").to_shared();

        EventControllers {
            c_ro,
            c_diff_x,
            c_diff_y,
            c_diff_z,
            c_pe_x,
            c_pe_y,
            c_pe_z,
            c_pe_ry,
            c_pe_rz,
            c_rf90,
            c_rf180,
            c_rf180_phase,
            c_crush_left,
            c_crush_right,
        }

    }
}
