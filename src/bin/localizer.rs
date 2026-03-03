use std::collections::HashMap;
use clap::Parser;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use mrs_ppl::compile::{build_seq, compile_seq};
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::rf_pulse::RfPulse;
use seq_struct::seq_loop::{Orientations, SeqLoop};
use seq_struct::variable::LUT;
use seq_struct::waveform::Waveform;
use seq_lib::grad_pulses::{ramp_down, ramp_up, trapezoid};
use seq_lib::{rf_pulses, PulseSequence};
use seq_lib::defs::{VIEW, SLICE, RF, GW, GS, RFP};
use param_editor::param_tree::{ParamTree};
use param_editor::value::Value;
use param_editor::{edit, unit};
use param_editor::parameter::Parameter;
use param_editor::ParameterTree;
use pulse_seq_view::run_viewer;
use seq_lib::Args;


fn main() {
    let out_dir = r"D:\dev\test\260123";

    let args = Args::parse();

    let param_tree = if args.init {
        Localizer::default().param_tree()
    }else {
        ParamTree::from_file(&args.param_file).unwrap()
    };

    let localizer = Localizer::from_params(&param_tree);

    localizer.param_tree().to_json_file(&args.param_file).unwrap();

    if args.edit {
        edit(&args.param_file).unwrap();
    }

    let state = localizer.adjustment_state();

    let timeline_data = localizer.render_timeline(&state).to_raw_loop_range(0,2);
    run_viewer(timeline_data).unwrap();

    let localizer = localizer.compile();
    //compile_seq(&localizer, out_dir, "localizer", false);
    //build_seq(out_dir)
}


impl ParameterTree for Localizer {
    fn param_tree(&self) -> ParamTree {
        ParamTree::new("localizer_default","localizer")
            .with_parameter(
                "Geometry",
                "FOV",
                Parameter::new("fov","fov",Value::Float(self.fov)).with_unit_str("mm").unwrap()
            ).unwrap()
            .with_parameter(
                "Geometry",
                "Slice Selection",
                Parameter::new("t_slice","slice thickness",Value::Float(self.slice_thickness_mm)).with_unit_str("mm").unwrap()
            ).unwrap()
            .with_parameter(
                "Geometry",
                "Slice Selection",
                Parameter::new("d_slice","pulse duration",Value::Count(self.rf_duration_us)).with_unit_str("us").unwrap()
            ).unwrap()
            .with_parameter(
                "Sampling",
                "Matrix",
                Parameter::new("n_read","n read",Value::Count(self.n_samples))
            ).unwrap()
            .with_parameter(
                "Sampling",
                "Bandwidth",
                Parameter::new("bw","bandwidth",Value::Float(self.bandwidth_khz)).with_unit_str("khz").unwrap()
            ).unwrap()
            .with_parameter(
                "Gradient",
                "Timing",
                Parameter::new("t_ramp","ramp time",Value::Count(self.grad_ramp_time_us)).with_unit_str("us").unwrap()
            ).unwrap()
            .with_parameter(
                "Gradient",
                "Timing",
                Parameter::new("t_phase","pe time",Value::Float(self.phase_enc_dur_ms)).with_unit_str("ms").unwrap()
            ).unwrap()
            .with_parameter(
                "Mode",
                "Setup",
                Parameter::new("setup","setup mode",Value::Bool(self.setup_mode))
            ).unwrap()
    }

    fn from_params(param_tree: &ParamTree) -> Self {
        // this dump may not work if there is a name collision. If there is a name collision,
        // we need to query params by tab and group address
        let params = param_tree.flatten_params().unwrap();
        Localizer {
            fov: params.get("fov").unwrap().into(),
            slice_thickness_mm: params.get("t_slice").unwrap().into(),
            n_samples: params.get("n_read").unwrap().into(),
            bandwidth_khz: params.get("bw").unwrap().into(),
            rf_duration_us: params.get("d_slice").unwrap().into(),
            grad_ramp_time_us: params.get("t_ramp").unwrap().into(),
            phase_enc_dur_ms: params.get("t_phase").unwrap().into(),
            setup_mode: params.get("setup").unwrap().into(),
        }
    }
}

#[derive(Clone,Debug)]
struct Localizer {
    fov: f64,
    slice_thickness_mm: f64,

    n_samples: usize,
    bandwidth_khz: f64,

    rf_duration_us: usize,
    grad_ramp_time_us: usize,
    phase_enc_dur_ms: f64,
    setup_mode: bool,
}

impl Default for Localizer {
    fn default() -> Self {
        Self {
            bandwidth_khz: 100.0,
            n_samples: 256,
            fov: 25.6,
            slice_thickness_mm: 0.3,
            rf_duration_us: 1500,
            grad_ramp_time_us: 500,
            phase_enc_dur_ms: 0.5,
            setup_mode: false
        }
    }
}

struct Waveforms {
    rf_pulse: RF,
    ru: GW,
    rd: GW,
    pe: GW,
    sl: GW,
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

        let sl = trapezoid(
            Time::us(params.grad_ramp_time_us),
            Time::us(params.rf_duration_us),
            grad_dt
        ).to_shared();

        Waveforms {
            rf_pulse,
            ru,
            rd,
            pe,
            sl,
        }

    }
}

struct EventControllers {
    gro: GS,
    gsl: GS,
    gslr: GS,
    gpe: GS,
    gpre: GS,
    rf: RFP,
    gsc: GS,
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

        let m1:Time = (Time::us(params.rf_duration_us) + rt).try_into().unwrap();
        let m2:Time = (Time::ms(params.phase_enc_dur_ms) + rt).try_into().unwrap();
        let f = m1 / m2;
        let g_slr = g_sl.scale(-f.si() / 2.);
        let gslr = EventControl::<FieldGrad>::new().with_constant_grad(g_slr).with_adj("ssr").to_shared();

        let tpe:Time = (Time::ms(params.phase_enc_dur_ms) + rt).try_into().unwrap();
        let gpe_step = FieldGrad::from_fov(Length::mm(params.fov),tpe,Nuc1H);

        let steps = if params.setup_mode {
            vec![0;params.n_samples]
        }else {
            let mut steps = vec![];
            for p in 0..params.n_samples as i32 {
                steps.push(p - params.n_samples as i32/2)
            }
            steps
        };


        let pe_lut = LUT::new("pelut",&steps).to_shared();
        let gpe = EventControl::<FieldGrad>::new().with_source_loop("view").with_lut(&pe_lut).with_grad_scale(gpe_step).to_shared();

        let rf = EventControl::<f64>::new().with_adj("rf_pow").to_shared();

        let gsc = EventControl::<FieldGrad>::new().with_constant_grad(g_pre.scale(-1)).with_adj("gsc").to_shared();

        EventControllers {
            gro,
            gsl,
            gslr,
            gpe,
            gpre,
            rf,
            gsc,
        }

    }
}

struct Events {
    sl: GradEvent,
    alpha: RfEvent,
    pe_ssr: GradEvent,
    roru: GradEvent,
    acq: ACQEvent,
    rord: GradEvent,
    gsc: GradEvent,
}

impl Events {
    fn build(params:&Localizer, w: &Waveforms, ec: &EventControllers) -> Events {

        let sl = GradEvent::new("sl").with_z(&w.sl).with_strength_z(&ec.gsl);

        let pe_ssr = GradEvent::new("pe_ssr").with_x(&w.pe).with_strength_x(&ec.gpre).with_y(&w.pe).with_strength_y(&ec.gpe).with_z(&w.pe).with_strength_z(&ec.gslr);
        let alpha = RfEvent::new("alpha",&w.rf_pulse,&ec.rf);

        let roru = GradEvent::new("roru").with_x(&w.ru).with_strength_x(&ec.gro);
        let rord = GradEvent::new("rord").with_x(&w.rd).with_strength_x(&ec.gro);
        let acq = ACQEvent::new("acq",params.n_samples,Freq::khz(params.bandwidth_khz).inv());

        let gsc = GradEvent::new("gsc")
            .with_x(&w.pe).with_strength_x(&ec.gsc)
            .with_y(&w.pe).with_strength_y(&ec.gsc)
            .with_z(&w.pe).with_strength_z(&ec.gsc);



        Events {
            sl,
            alpha,
            pe_ssr,
            roru,
            acq,
            rord,
            gsc,
        }
    }
}

impl PulseSequence for Localizer {
    fn compile(&self) -> (SeqLoop,Self) {

        let waveforms = Waveforms::build(&self);
        let event_controllers = EventControllers::build(self, &waveforms);
        let events = Events::build(&self,&waveforms,&event_controllers);

        let mut sl = SeqLoop::new_main("slice",3);

        if self.setup_mode {
            let o = [
                [Angle::deg(0),Angle::deg(0),Angle::deg(0)],
            ];
            sl.set_orientations(Orientations::new(&o));
        }else {
            let o = [
                [Angle::deg(0),Angle::deg(0),Angle::deg(0)],
                [Angle::deg(0),Angle::deg(0),Angle::deg(90)],
                [Angle::deg(0),Angle::deg(-90),Angle::deg(0)],
            ];
            sl.set_orientations(Orientations::new(&o));
        };

        // 3-plane localizer rotations

        sl.add_event(events.sl).unwrap();
        sl.add_event(events.alpha).unwrap();
        sl.add_event(events.pe_ssr).unwrap();
        sl.add_event(events.roru).unwrap();
        sl.add_event(events.acq).unwrap();
        sl.add_event(events.rord).unwrap();
        sl.add_event(events.gsc).unwrap();


        sl.set_time_span("sl","alpha",50,50,Time::us(0)).unwrap();
        sl.set_time_span("sl","pe_ssr",100,0,Time::us(200)).unwrap();
        sl.set_time_span("pe_ssr","roru",100,0,Time::us(200)).unwrap();
        sl.set_min_time_span("roru","acq",100,0,Time::us(0)).unwrap();
        sl.set_min_time_span("acq","rord",100,0,Time::us(0)).unwrap();
        sl.set_min_time_span("rord","gsc",100,0,Time::us(100)).unwrap();
        sl.set_rep_time(Time::ms(10)).unwrap();
        sl.set_pre_calc(Time::ms(4));
        let mut vl = SeqLoop::new("view", self.n_samples);
        vl.set_pre_calc(Time::ms(2));
        vl.add_loop(sl).unwrap();
        vl.set_rep_time(Time::ms(50)).unwrap();

        (vl,self.clone())

    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut adj = HashMap::new();
        adj.insert("rf_pow".to_string(),1.);
        adj.insert("ssr".to_string(),0.);
        adj.insert("gsc".to_string(),0.);
        adj
    }
}




