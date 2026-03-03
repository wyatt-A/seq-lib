use std::collections::HashMap;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use mrs_ppl::compile::{build_seq, compile_seq};
use pulse_seq_view::run_viewer;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use seq_lib::defs::{GS, GW, RF, RFP, RF_POWER, VIEW};
use seq_lib::grad_pulses::{half_sin, quarter_sin_rd, quarter_sin_ru, ramp_down, ramp_up, trapezoid};
use seq_lib::{rf_pulses, PulseSequence};
use seq_lib::grad_pulses::scale_factors::{HALF_SIN_SCALE, QUARTER_SIN_SCALE};

fn main() {

    let mut mgre = Mgre::default();
    mgre.sim_mode = false;
    mgre.gop_mode = true;
    mgre.n_read = 2_560;
    mgre.n_phase_y = 204_800/3;
    mgre.n_phase_z = 204_800/3;
    mgre.fov_mm = [25.6,12.8,12.8];
    mgre.echo_location = 0.4;

    let (seq_loop,mgre) = mgre.compile();
    let user_state = mgre.adjustment_state();
    println!("{:?}",mgre);


    let out_dir = r"D:\dev\test\260303\mgre";
    compile_seq(&seq_loop,out_dir,"mgre",true);
    build_seq(out_dir)
    //let ps_data = sl.render_timeline(&user_state).to_raw_loop_range(1198,1199);

    //run_viewer(ps_data).unwrap();

}

#[derive(Debug,Clone)]
pub struct Mgre {
    /// field of view in mm
    fov_mm: [f64;3],
    /// receiver bandwidth in kHz
    bandwidth_khz: f64,
    /// number of readout samples
    n_read: usize,
    /// number of phase encoding samples along y
    n_phase_y: usize,
    /// number of phase encoding samples along z
    n_phase_z: usize,
    /// relative location of gradient echo (0.5 is center in ro)
    echo_location: f64,
    /// duration of rf pulse in microseconds
    rf_pulse_dur_us: f64,
    /// gradient ramp time in microseconds
    ramp_time_us: f64,
    /// phase encoding duration in microseconds
    pe_dur_us: f64,
    /// use smooth gradient shapes instead of linear ramps
    smooth_grad:bool,
    /// when gain optimization is true, phase encoding is disabled
    gop_mode:bool,
    /// if sim mode is true, only 4 views will be calculated
    sim_mode:bool,
    /// repetition time in milliseconds
    rep_time_ms: f64,
    /// post-alpha pulse delay in microseconds
    post_alpha_del_us: f64,
    /// post phase encode pulse delay in microseconds
    post_pe_del_us: f64,
    /// delay between readout ramp and start of acq
    post_ru_del_us: f64,
    /// delay after acquisition, before ramp down
    post_acq_del_us: f64,
    /// delay between readout gradients
    inter_readout_del_us: f64,
    /// calculated echo 1 time
    te1_ms: Option<f64>,
    /// calculated echo 2 time
    te2_ms: Option<f64>,
}

impl Default for Mgre {
    fn default() -> Self {
        Mgre {
            fov_mm: [20.,12.8,12.8],
            bandwidth_khz: 400.,
            n_read: 2000,
            n_phase_y: 1200,
            n_phase_z: 1200,
            rf_pulse_dur_us: 100.,
            ramp_time_us: 300.,
            pe_dur_us: 2000.,
            echo_location: 0.5,
            smooth_grad: true,
            sim_mode: true,
            gop_mode: true,
            rep_time_ms: 100.,
            post_alpha_del_us: 0.,
            post_pe_del_us: 13.,
            post_ru_del_us: 50.,
            post_acq_del_us: 50.,
            inter_readout_del_us: 13.,
            te1_ms: None,
            te2_ms: None,
        }
    }
}

impl PulseSequence for Mgre {
    fn compile(&self) -> (SeqLoop,Self) {

        let mut mgre = self.clone();

        let tr = Time::ms(self.rep_time_ms);
        let pre_calc_time = Time::ms(4);
        let post_alpha_del = Time::us(self.post_alpha_del_us);
        let post_pe_del = Time::us(self.post_pe_del_us);
        let post_ro_ramp_del = Time::us(self.post_ru_del_us);
        let post_acq_del = Time::us(self.post_acq_del_us);
        let inter_readout_del = Time::us(self.inter_readout_del_us);

        let e = Events::build(self);

        let n_views = if self.sim_mode {
            4
        }else {
            self.n_phase_y*self.n_phase_z
        };

        let mut vl = SeqLoop::new_main(VIEW,n_views);
        // give 2 ms of calculation time for parameters
        vl.set_pre_calc(pre_calc_time);

        // add events to view loop
        vl.add_event(e.e_alpha).unwrap();
        vl.add_event(e.e_pe).unwrap();
        vl.add_event(e.e_ro_ru).unwrap();
        vl.add_event(e.e_acq).unwrap();
        vl.add_event(e.e_ro_rd).unwrap();
        vl.add_event(e.e_ro_ru2).unwrap();
        vl.add_event(e.e_acq2).unwrap();
        vl.add_event(e.e_ro_rd2).unwrap();

        // end of alpha pulse to start of phase encoding
        if let Ok(time) = vl.set_min_time_span(Events::alpha(),Events::pe(),100,0,post_alpha_del) {
            mgre.post_alpha_del_us = time.as_us();
            println!("set post-alpha delay to {:?} us",mgre.post_alpha_del_us);
        }else {
            panic!("failed to set post-alpha delay");
        }

        if let Ok(time) = vl.set_min_time_span(Events::pe(),Events::ro_ru(),100,0,post_pe_del) {
            mgre.post_pe_del_us = time.as_us();
            println!("set post-phase encode delay to {:?} us",mgre.post_pe_del_us);
        }

        if let Ok(time) = vl.set_min_time_span(Events::ro_ru(),Events::acq(),100,0,post_ro_ramp_del) {
            mgre.post_ru_del_us = time.as_us();
            println!("set post-readout ramp delay to {:?} us",mgre.post_ru_del_us);
        }

        if let Ok(time) = vl.set_min_time_span(Events::acq(),Events::ro_rd(),100,0,post_acq_del) {
            mgre.post_acq_del_us = time.as_us();
            println!("set post-acq delay to {:?} us",mgre.post_acq_del_us);
        }

        if let Ok(time) = vl.set_min_time_span(Events::ro_rd(),Events::ro_ru2(),100,0,inter_readout_del) {
            mgre.inter_readout_del_us = time.as_us();
            println!("set inter-readout delay to {:?} us",mgre.inter_readout_del_us);
        }

        vl.set_min_time_span(Events::ro_ru2(),Events::acq2(),100,0,post_ro_ramp_del).unwrap();
        vl.set_min_time_span(Events::acq2(),Events::ro_rd2(),100,0,post_acq_del).unwrap();

        vl.set_averages(1);
        vl.set_rep_time(tr).unwrap();


        // calculate echo times
        // time from center of alpha pulse to end of readout ramp up
        let t0 = vl.get_time_span(Events::alpha(),Events::ro_ru(),50,100).unwrap();
        let t1 = vl.get_time_span(Events::alpha(),Events::ro_rd(),50,0).unwrap();
        // first echo forms between ru and rd, parameterized by echo location
        let te1 = (t1.as_ms() - t0.as_ms()) * self.echo_location + t0.as_ms();
        mgre.te1_ms = Some(te1);

        let t0 = vl.get_time_span(Events::alpha(),Events::ro_ru2(),50,100).unwrap();
        let t1 = vl.get_time_span(Events::alpha(),Events::ro_rd2(),50,0).unwrap();
        // second echo forms between ru2 and rd2, parameterized by 1 - echo location due to inverted
        // gradient polarity
        let te2 = (t1.as_ms() - t0.as_ms()) * (1. - self.echo_location) + t0.as_ms();
        mgre.te2_ms = Some(te2);

        // return loop and modified parameters
        (vl,mgre)

    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut s = HashMap::new();
        s.insert(RF_POWER.to_string(),1.0);
        s.insert("gpre".to_string(),0.);
        s
    }
}

struct Waveforms {
    /// rf pulse
    rf: RF,
    /// gradient ramp-up
    ru: GW,
    /// gradient ramp-down
    rd: GW,
    /// phase encoding pulse
    pe: GW,
}

impl Waveforms {
    fn build(mgre: &Mgre) -> Waveforms {
        let rf_dt = Time::us(2);
        let grad_dt = Time::us(2);

        let rf = rf_pulses::hardpulse(
            Time::us(mgre.rf_pulse_dur_us),
            rf_dt,
            Nuc1H
        ).to_shared();

        let (ru,rd,pe) = if mgre.smooth_grad {

            let ru = quarter_sin_ru(
                Time::us(mgre.ramp_time_us),
                grad_dt
            ).to_shared();

            let rd = quarter_sin_rd(
                Time::us(mgre.ramp_time_us),
                grad_dt
            ).to_shared();

            let pe = half_sin(
                Time::us(mgre.pe_dur_us),
                grad_dt
            ).to_shared();

            (ru,rd,pe)
        }else {

            let ru = ramp_up(
                Time::us(mgre.ramp_time_us),
                grad_dt
            ).to_shared();

            let rd = ramp_down(
                Time::us(mgre.ramp_time_us),
                grad_dt
            ).to_shared();

            let pe = trapezoid(
                Time::us(mgre.ramp_time_us),
                Time::us(mgre.pe_dur_us),
                grad_dt
            ).to_shared();

            (ru, rd, pe)
        };


        Waveforms {
            rf,
            ru,
            rd,
            pe,
        }
    }
}

struct EventControllers {
    c_gro: GS,
    c_gro2: GS,
    c_gpe_x: GS,
    c_gpe_y: GS,
    c_gpe_z: GS,
    c_rfp: RFP,
}

impl EventControllers {
    fn build(mgre: &Mgre) -> EventControllers {

        let nuc = Nuc1H;
        let fov_x = Length::mm(mgre.fov_mm[0]);
        let fov_y = Length::mm(mgre.fov_mm[1]);
        let fov_z = Length::mm(mgre.fov_mm[2]);
        let s_dt:Time = Freq::khz(mgre.bandwidth_khz).inv().into();

        // effective phase encode time (shorter for half-sin lobe)
        let pe_dt = if mgre.smooth_grad {
            // trapezoidal gradient
            Time::us(mgre.pe_dur_us + mgre.ramp_time_us)
        }else {
            // half-sin gradient
            Time::us(mgre.pe_dur_us / HALF_SIN_SCALE)
        };

        // effective ramp time (shorter for smooth ramps)
        let t_ramp = if mgre.smooth_grad {
            Time::us(mgre.ramp_time_us / QUARTER_SIN_SCALE)
        }else {
            Time::us(mgre.ramp_time_us)
        };

        // calculate imaging gradient strengths
        let gro = FieldGrad::from_fov(fov_x,s_dt,nuc);
        let gpe_y = FieldGrad::from_fov(fov_y,pe_dt,nuc);
        let gpe_z = FieldGrad::from_fov(fov_z,pe_dt,nuc);

        // echo location coefficient (between 0 and 1). This scales the pre-phase gradient lobe
        let echo_location = mgre.echo_location.clamp(0.,1.);

        // readout moment
        let ro_time:Time = (s_dt.scale(mgre.n_read) + t_ramp).try_into().unwrap();
        let ro_moment = ro_time * gro;

        let pe_moment = - ro_moment * echo_location;
        // prephase lobe strength
        let gpe_x:FieldGrad = (pe_moment / pe_dt).try_into().unwrap();

        // first echo readout
        let c_gro = EventControl::<FieldGrad>::new().with_constant_grad(gro).to_shared();
        // second echo readout
        let c_gro2 = EventControl::<FieldGrad>::new().with_constant_grad(gro.scale(-1)).to_shared();
        // pre-phasing
        let c_gpe_x = EventControl::<FieldGrad>::new().with_constant_grad(gpe_x).with_adj("gpre").to_shared();

        // phase encoding steps for y and z axes
        let mut pe_steps_y:Vec<i32> = (0..mgre.n_phase_y).map(|step| step as i32).map(|step| step - mgre.n_phase_y as i32  / 2).collect();
        let mut pe_steps_z:Vec<i32> = (0..mgre.n_phase_z).map(|step| step as i32).map(|step| step - mgre.n_phase_z as i32  / 2).collect();

        if mgre.gop_mode {
            pe_steps_y.fill(0);
            pe_steps_z.fill(0);
        }

        let lut_pe_y = LUT::new("pe_y_lut",&pe_steps_y).to_shared();
        let lut_pe_z = LUT::new("pe_z_lut",&pe_steps_z).to_shared();

        // view index is used for phase y index directly
        let c_gpe_y = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_y)
            .with_source_loop(VIEW)
            .with_mod(mgre.n_phase_y as i32)
            .with_grad_scale(gpe_y)
            .to_shared();

        // phase z index is view_idx % n_phase_y
        let c_gpe_z = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_z)
            .with_source_loop(VIEW)
            .with_div_op(mgre.n_phase_y as i32)
            .with_grad_scale(gpe_z)
            .to_shared();

        // user-adjustable rf power
        let c_rfp = EventControl::<f64>::new().with_adj(RF_POWER).to_shared();

        EventControllers {
            c_gro,
            c_gro2,
            c_gpe_x,
            c_gpe_y,
            c_gpe_z,
            c_rfp,
        }

    }
}

struct Events {
    e_alpha: RfEvent,
    e_pe: GradEvent,
    e_ro_ru: GradEvent,
    e_ro_rd: GradEvent,

    e_ro_ru2: GradEvent,
    e_ro_rd2: GradEvent,

    e_acq: ACQEvent,

    e_acq2: ACQEvent,
}

impl Events {

    fn build(mgre: &Mgre) -> Events {
        let w = Waveforms::build(mgre);
        let ec = EventControllers::build(mgre);
        let e_alpha = RfEvent::new(Events::alpha(),&w.rf,&ec.c_rfp);
        let e_pe = GradEvent::new(Events::pe())
            .with_x(&w.pe).with_y(&w.pe).with_z(&w.pe)
            .with_strength_x(&ec.c_gpe_x).with_strength_y(&ec.c_gpe_y).with_strength_z(&ec.c_gpe_z);

        // readout ramps for echo 1
        let e_ro_ru = GradEvent::new(Events::ro_ru()).with_x(&w.ru).with_strength_x(&ec.c_gro);
        let e_ro_rd = GradEvent::new(Events::ro_rd()).with_x(&w.rd).with_strength_x(&ec.c_gro);

        // readout ramps for echo 2
        let e_ro_ru2 = GradEvent::new(Events::ro_ru2()).with_x(&w.ru).with_strength_x(&ec.c_gro2);
        let e_ro_rd2 = GradEvent::new(Events::ro_rd2()).with_x(&w.rd).with_strength_x(&ec.c_gro2);

        let e_acq = ACQEvent::new(Events::acq(),mgre.n_read,Freq::khz(mgre.bandwidth_khz).inv());
        let e_acq2 = e_acq.clone().set_label(Events::acq2());

        Events {
            e_alpha,
            e_pe,
            e_ro_ru,
            e_ro_rd,
            e_ro_ru2,
            e_ro_rd2,
            e_acq,
            e_acq2,
        }
    }

    fn alpha() -> &'static str {
        "alpha"
    }

    fn pe() -> &'static str {
        "pe"
    }

    fn ro_ru() -> &'static str {
        "ro_ru"
    }

    fn ro_rd() -> &'static str {
        "ro_rd"
    }

    fn ro_ru2() -> &'static str {
        "ro_ru2"
    }

    fn ro_rd2() -> &'static str {
        "ro_rd2"
    }

    fn acq() -> &'static str {
        "acq"
    }

    fn acq2() -> &'static str {
        "acq2"
    }
}























