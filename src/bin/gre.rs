use std::collections::HashMap;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use mrs_ppl::compile::{build_ppl,compile_ppl};
use pulse_seq_view::run_viewer;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use seq_lib::defs::{EXPERIMENT, GS, GW, RF, RFP, RF_POWER, VIEW};
use seq_lib::grad_pulses::{half_sin, quarter_sin_rd, quarter_sin_ru, ramp_down, ramp_up, trapezoid};
use seq_lib::{rf_pulses, PulseSequence};
use seq_lib::grad_pulses::scale_factors::{HALF_SIN_SCALE, QUARTER_SIN_SCALE};

fn main() {

    let mut gre = Gre::default();
    gre.sim_mode = false;
    gre.gop_mode = true;
    gre.smooth_grad = true;
    gre.n_read = 2560;
    gre.n_phase_y = 1;
    gre.n_phase_z = 1;
    gre.fov_mm = [25.6,12.8,12.8];
    gre.echo_location = 0.2;
    gre.pe_dur_us = 1000.;
    gre.n_pspace_samples = 2;
    gre.pspace_step_size_mtpm = 100.;
    gre.rep_time_ms = 15.;

    let (seq_loop,mgre) = gre.compile();
    let user_state = mgre.adjustment_state();
    println!("{:?}",mgre);

    // let out_dir = r"D:\dev\test\260303\mgre";
    // compile_seq(&seq_loop,out_dir,"mgre",true);
    // build_seq(out_dir)
    //let ps_data = seq_loop.render_timeline(&user_state).to_raw_loop_range(0,3);
    let ps_data = seq_loop.render_timeline(&user_state).to_raw();

    run_viewer(ps_data).unwrap();

}

#[derive(Debug,Clone)]
pub struct Gre {
    /// field of view in mm
    fov_mm: [f64;3],
    /// number of p-space samples to acquire multiple echo times
    n_pspace_samples: usize,
    /// step size to modulate the pre-phase gradient in millitesla per meter
    pspace_step_size_mtpm: f64,
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
    /// estimated echo time
    te_ms: Option<f64>,
    /// gradient strength limit
    grad_limit_tpm: f64,
}

impl Default for Gre {
    fn default() -> Self {
        Gre {
            fov_mm: [20.,12.8,12.8],
            n_pspace_samples: 3,
            pspace_step_size_mtpm: 1.,
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
            te_ms: None,
            grad_limit_tpm: 2.0,
        }
    }
}

impl PulseSequence for Gre {
    fn compile(&self) -> (SeqLoop,Self) {

        let mut gre = self.clone();

        let tr = Time::ms(self.rep_time_ms);
        let pre_calc_time = Time::ms(4);
        let post_alpha_del = Time::us(self.post_alpha_del_us);
        let post_pe_del = Time::us(self.post_pe_del_us);
        let post_ro_ramp_del = Time::us(self.post_ru_del_us);
        let post_acq_del = Time::us(self.post_acq_del_us);

        let e = Events::build(self);

        let n_views = if self.sim_mode {
            4
        }else {
            self.n_phase_y*self.n_phase_z
        };

        let mut expl = SeqLoop::new(EXPERIMENT,self.n_pspace_samples);

        let mut vl = SeqLoop::new_main(VIEW,n_views);
        // give 2 ms of calculation time for parameters
        vl.set_pre_calc(pre_calc_time);

        // add events to view loop
        vl.add_event(e.e_alpha).unwrap();
        vl.add_event(e.e_pe).unwrap();
        vl.add_event(e.e_ro_ru).unwrap();
        vl.add_event(e.e_acq).unwrap();
        vl.add_event(e.e_ro_rd).unwrap();

        // end of alpha pulse to start of phase encoding
        if let Ok(time) = vl.set_min_time_span(Events::alpha(),Events::pe(),100,0,post_alpha_del) {
            gre.post_alpha_del_us = time.as_us();
            println!("set post-alpha delay to {:?} us", gre.post_alpha_del_us);
        }else {
            panic!("failed to set post-alpha delay");
        }

        if let Ok(time) = vl.set_min_time_span(Events::pe(),Events::ro_ru(),100,0,post_pe_del) {
            gre.post_pe_del_us = time.as_us();
            println!("set post-phase encode delay to {:?} us", gre.post_pe_del_us);
        }

        if let Ok(time) = vl.set_min_time_span(Events::ro_ru(),Events::acq(),100,0,post_ro_ramp_del) {
            gre.post_ru_del_us = time.as_us();
            println!("set post-readout ramp delay to {:?} us", gre.post_ru_del_us);
        }

        if let Ok(time) = vl.set_min_time_span(Events::acq(),Events::ro_rd(),100,0,post_acq_del) {
            gre.post_acq_del_us = time.as_us();
            println!("set post-acq delay to {:?} us", gre.post_acq_del_us);
        }

        vl.set_averages(1);
        vl.set_rep_time(tr).unwrap();

        // calculate echo times
        // time from center of alpha pulse to end of readout ramp up
        let t0 = vl.get_time_span(Events::alpha(),Events::ro_ru(),50,100).unwrap();
        let t1 = vl.get_time_span(Events::alpha(),Events::ro_rd(),50,0).unwrap();
        // first echo forms between ru and rd, parameterized by echo location
        let te1 = (t1.as_ms() - t0.as_ms()) * self.echo_location + t0.as_ms();
        gre.te_ms = Some(te1);

        expl.add_loop(vl).unwrap();
        expl.no_overhead();

        // return loop and modified parameters
        (expl, gre)

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
    fn build(mgre: &Gre) -> Waveforms {
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
    c_gpe_x: GS,
    c_gpe_y: GS,
    c_gpe_z: GS,
    c_rfp: RFP,
}

impl EventControllers {
    fn build(gre: &Gre) -> EventControllers {

        let nuc = Nuc1H;
        let fov_x = Length::mm(gre.fov_mm[0]);
        let fov_y = Length::mm(gre.fov_mm[1]);
        let fov_z = Length::mm(gre.fov_mm[2]);
        let s_dt:Time = Freq::khz(gre.bandwidth_khz).inv().into();

        let p_space_steps:Vec<i32> = (0..gre.n_pspace_samples).map(|p| p as i32).collect();


        // effective phase encode time (shorter for half-sin lobe)
        let pe_dt = if gre.smooth_grad {
            // half-sin gradient
            Time::us(gre.pe_dur_us * HALF_SIN_SCALE)
        }else {
            // trapezoidal gradient
            Time::us(gre.pe_dur_us + gre.ramp_time_us)
        };

        // effective ramp time (shorter for smooth ramps)
        let t_ramp = if gre.smooth_grad {
            Time::us(gre.ramp_time_us * QUARTER_SIN_SCALE)
        }else {
            Time::us(gre.ramp_time_us)
        };

        // calculate imaging gradient strengths
        let gro = FieldGrad::from_fov(fov_x,s_dt,nuc);
        assert!(gro.si().abs() <= gre.grad_limit_tpm, "gro exceeds max gradient strength: {}",gro.si());

        let gpe_y = FieldGrad::from_fov(fov_y,pe_dt,nuc);
        let gpe_y_max = gpe_y.scale(gre.n_phase_y/2);
        assert!(gpe_y_max.si().abs() <= gre.grad_limit_tpm, "gpy exceeds max gradient strength: {}",gpe_y_max.si());

        let gpe_z = FieldGrad::from_fov(fov_z,pe_dt,nuc);
        let gpe_z_max = gpe_z.scale(gre.n_phase_z/2);
        assert!(gpe_z_max.si().abs() <= gre.grad_limit_tpm, "gpz exceeds max gradient strength: {}",gpe_z_max.si());

        // echo location coefficient (between 0 and 1). This scales the pre-phase gradient lobe
        let echo_location = gre.echo_location.clamp(0., 1.);

        // readout moment
        let ro_time:Time = (s_dt.scale(gre.n_read) + t_ramp).try_into().unwrap();
        let ro_moment = ro_time * gro;

        let pe_moment = - ro_moment * echo_location;
        // base prephase lobe strength
        let gpe_x:FieldGrad = (pe_moment / pe_dt).try_into().unwrap();




        // check gradient strength
        assert!(gpe_x.si().abs() <= gre.grad_limit_tpm, "gpe_x exceeds max gradient strength: {}",gpe_x.si());

        // echo readout
        let c_gro = EventControl::<FieldGrad>::new().with_constant_grad(gro).to_shared();
        // pre-phasing and p-space sampling. This is a constant pre-phase gradient modulated
        let p_space_lut = LUT::new("pspace",&p_space_steps).to_shared();
        let g_pspace = FieldGrad::mt_per_meter(gre.pspace_step_size_mtpm);
        let c_gpe_x = EventControl::<FieldGrad>::new()
            .with_source_loop(EXPERIMENT)
            .with_lut(&p_space_lut)
            .with_grad_scale(g_pspace)
            .with_constant_grad(gpe_x)
            .with_adj("gpre")
            .to_shared();

        // phase encoding steps for y and z axes
        let mut pe_steps_y:Vec<i32> = (0..gre.n_phase_y).map(|step| step as i32).map(|step| step - gre.n_phase_y as i32  / 2).collect();
        let mut pe_steps_z:Vec<i32> = (0..gre.n_phase_z).map(|step| step as i32).map(|step| step - gre.n_phase_z as i32  / 2).collect();

        if gre.gop_mode {
            pe_steps_y.fill(0);
            pe_steps_z.fill(0);
        }

        let lut_pe_y = LUT::new("pe_y_lut",&pe_steps_y).to_shared();
        let lut_pe_z = LUT::new("pe_z_lut",&pe_steps_z).to_shared();

        // view index is used for phase y index directly
        let c_gpe_y = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_y)
            .with_source_loop(VIEW)
            .with_mod(gre.n_phase_y as i32)
            .with_grad_scale(gpe_y)
            .to_shared();

        // phase z index is view_idx % n_phase_y
        let c_gpe_z = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_z)
            .with_source_loop(VIEW)
            .with_div_op(gre.n_phase_y as i32)
            .with_grad_scale(gpe_z)
            .to_shared();

        // user-adjustable rf power
        let c_rfp = EventControl::<f64>::new().with_adj(RF_POWER).to_shared();

        EventControllers {
            c_gro,
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
    e_acq: ACQEvent,
}

impl Events {

    fn build(mgre: &Gre) -> Events {
        let w = Waveforms::build(mgre);
        let ec = EventControllers::build(mgre);
        let e_alpha = RfEvent::new(Events::alpha(),&w.rf,&ec.c_rfp);
        let e_pe = GradEvent::new(Events::pe())
            .with_x(&w.pe).with_y(&w.pe).with_z(&w.pe)
            .with_strength_x(&ec.c_gpe_x).with_strength_y(&ec.c_gpe_y).with_strength_z(&ec.c_gpe_z);

        // readout ramps for echo 1
        let e_ro_ru = GradEvent::new(Events::ro_ru()).with_x(&w.ru).with_strength_x(&ec.c_gro);
        let e_ro_rd = GradEvent::new(Events::ro_rd()).with_x(&w.rd).with_strength_x(&ec.c_gro);
        let e_acq = ACQEvent::new(Events::acq(),mgre.n_read,Freq::khz(mgre.bandwidth_khz).inv());

        Events {
            e_alpha,
            e_pe,
            e_ro_ru,
            e_ro_rd,
            e_acq,
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

    fn acq() -> &'static str {
        "acq"
    }

}























