use std::collections::HashMap;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use array_lib::io_mrd::read_mrd;
use clap::Parser;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use num_complex::Complex32;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use serde::{Deserialize, Serialize};
use seq_lib::defs::{ECHO, GS, GW, RF, RFP, RF_POWER, VIEW};
use seq_lib::grad_pulses::{half_sin, quarter_sin_rd, quarter_sin_ru, ramp_down, ramp_up, trapezoid};
use seq_lib::grad_pulses::scale_factors::{HALF_SIN_SCALE, QUARTER_SIN_SCALE};
use seq_lib::{get_pe_table, rf_pulses, Args, PulseSequence, ToHeadfile, TOML};

const SEQ_NAME:&str = "mgre";

fn main() {
    Args::parse().run::<Mgre>();
}


#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Mgre {
    /// references the pulse sequence bin used to run the protocol
    seq_name: String,
    /// field of view in mm
    fov_mm: [f64;3],
    /// size of acquisition grid where `matrix[0]` is the number of readout samples
    matrix_size: [usize;3],
    /// number of gradient echoes to acquire
    n_echoes: usize,
    /// receiver bandwidth in kHz
    bandwidth_khz: f64,
    /// path to cs table with index entries `[y0,z0,y1,z1...etc.]`
    pe_table: Option<PathBuf>,
    /// number of phase encoding samples along y
    n_phase_y: Option<usize>,
    /// number of phase encoding samples along z
    n_phase_z: Option<usize>,
    /// relative location of gradient echo (0.5 is center in ro)
    echo_location: f64,
    /// strength of spoiler gradient in millitesla per meter
    spoiler_grad_mtpm: f64,
    /// duration of rf pulse in microseconds
    rf_pulse_dur_us: f64,
    /// gradient ramp time in microseconds
    ramp_time_us: f64,
    /// phase encoding duration in microseconds
    pe_dur_us: f64,
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
    /// number of dummy scans
    dummy_scans: usize,
    /// echo time
    te_ms: Option<f64>,
    /// echo spacing
    echo_spacing_ms: Option<f64>,
}

impl Default for Mgre {
    fn default() -> Mgre {
        Mgre {
            seq_name: SEQ_NAME.to_string(),
            fov_mm: [20.,12.8,12.8],
            matrix_size: [512,256,256],
            n_echoes: 6,
            bandwidth_khz: 50.,
            pe_table: None,
            n_phase_y: None,
            n_phase_z: None,
            rf_pulse_dur_us: 100.,
            ramp_time_us: 300.,
            pe_dur_us: 2000.,
            echo_location: 0.5,
            spoiler_grad_mtpm: 200.,
            sim_mode: true,
            gop_mode: true,
            rep_time_ms: 100.,
            post_alpha_del_us: 0.,
            post_pe_del_us: 13.,
            post_ru_del_us: 50.,
            post_acq_del_us: 50.,
            dummy_scans: 10,
            te_ms: None,
            echo_spacing_ms: None,
        }
    }
}

impl TOML for Mgre {}

impl ToHeadfile for Mgre {
    fn seq_name() -> &'static str {
        SEQ_NAME
    }
}

impl PulseSequence for Mgre {
    fn build_sequence(&mut self) -> SeqLoop {

        let events = Events::build(self);

        let mut echo_loop = SeqLoop::new(ECHO,self.n_echoes - 1);
        echo_loop.add_event(events.e_pe_re).unwrap();
        echo_loop.add_event(events.e_ro_ru2).unwrap();
        echo_loop.add_event(events.e_acq2).unwrap();
        echo_loop.add_event(events.e_ro_rd2).unwrap();
        echo_loop.set_time_span(Events::pe_re(),Events::ro_ru2(),100,0,Time::us(100)).unwrap();
        echo_loop.set_time_span(Events::ro_ru2(),Events::acq2(),100,0,Time::us(100)).unwrap();
        echo_loop.set_min_time_span(Events::acq2(),Events::ro_rd2(),100,0,Time::us(100)).unwrap();
        echo_loop.set_pre_calc(Time::us(700));
        echo_loop.set_rep_time(Time::ms(9)).unwrap();


        let n_views = if self.sim_mode {
            2
        }else if self.gop_mode {
            10000
        }else {
            self.n_phase_y.unwrap()
        };

        let mut view_loop = SeqLoop::new_main(VIEW, n_views);

        view_loop.add_event(events.e_alpha).unwrap();
        view_loop.add_event(events.e_pe).unwrap();
        view_loop.add_event(events.e_ro_ru).unwrap();
        view_loop.add_event(events.e_acq).unwrap();
        view_loop.add_event(events.e_ro_rd).unwrap();

        view_loop.set_time_span(Events::alpha(),Events::pe(),100,0,Time::us(100)).unwrap();
        view_loop.set_time_span(Events::pe(),Events::ro_ru(),100,0,Time::us(100)).unwrap();
        view_loop.set_time_span(Events::ro_ru(),Events::acq(),100,0,Time::us(100)).unwrap();
        view_loop.set_min_time_span(Events::acq(),Events::ro_rd(),100,0,Time::us(100)).unwrap();

        view_loop.add_loop(echo_loop).unwrap();

        view_loop.add_event(events.e_spoil).unwrap();
        view_loop.set_min_time_span(Events::ro_rd2(),Events::spoil(),100,0,Time::us(100)).unwrap();

        view_loop.set_min_time_span(Events::ro_rd(),Events::pe_re(),100,0,Time::us(500)).unwrap();

        view_loop.set_pre_calc(Time::ms(4));
        view_loop.set_rep_time(Time::ms(self.rep_time_ms)).unwrap();

        view_loop

    }

    fn adjustment_state(&self) -> HashMap<String, f64> {
        let mut s = HashMap::new();
        s.insert(RF_POWER.to_string(),1.0);
        s.insert("gpre".to_string(),0.);
        s.insert("gpx_re".to_string(),0.);
        s
    }

    fn finish_acquisition(&mut self, acq_dir: impl AsRef<Path>, results_dir: impl AsRef<Path>) {
        let (raw,dims,..) = read_mrd(acq_dir.as_ref().join(SEQ_NAME).with_extension("MRD"));

        let shape = dims.shape_squeeze();
        let n_read = shape[0];
        let n_views = shape[1];
        let n_echoes = shape[2];

        let n_dig = n_echoes.to_string().chars().count();

        let pe = get_pe_table(&acq_dir,self.dummy_scans);

        let vol_dim = ArrayDim::from_shape(&self.matrix_size);
        for (e,echo) in raw.chunks_exact(n_read * n_views).enumerate() {
            let mut vol = vol_dim.alloc(Complex32::ZERO);
            for (i,view) in echo.chunks_exact(n_read).skip(self.dummy_scans).enumerate() {
                let [y,z] = pe[i];
                for (x,sample) in view.iter().enumerate() {
                    let addr = vol_dim.calc_addr_signed(&[x as isize,y,z]);
                    vol[addr] = *sample;
                }
            }
            write_cfl(results_dir.as_ref().join(format!("m{:0width$}",e,width=n_dig)),&vol,vol_dim);
        }
    }

    fn gop_mode(&mut self) {
        self.sim_mode = false;
        self.gop_mode = true;
    }

    fn acq_mode(&mut self) {
        self.gop_mode = false;
        self.sim_mode = false;
    }

    fn display_mode(&mut self) {
        self.sim_mode = true;
    }

    fn sim_mode(&mut self) {
        self.sim_mode = true;
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
    c_gx_re: GS,
    c_gpe_x: GS,
    c_gpe_y: GS,
    c_gpe_z: GS,
    c_rfp: RFP,
    c_gs: GS,
}

impl EventControllers {
    fn build(mgre: &mut Mgre) -> EventControllers {

        let nuc = Nuc1H;
        let fov_x = Length::mm(mgre.fov_mm[0]);
        let fov_y = Length::mm(mgre.fov_mm[1]);
        let fov_z = Length::mm(mgre.fov_mm[2]);
        let s_dt:Time = Freq::khz(mgre.bandwidth_khz).inv().into();

        // effective phase encode time (shorter for half-sin lobe)
        let pe_dt = Time::us(mgre.pe_dur_us + mgre.ramp_time_us);
        let t_ramp = Time::us(mgre.ramp_time_us);

        // calculate imaging gradient strengths
        let gro = FieldGrad::from_fov(fov_x,s_dt,nuc);

        let gpe_y = FieldGrad::from_fov(fov_y,pe_dt,nuc);

        let gpe_z = FieldGrad::from_fov(fov_z,pe_dt,nuc);

        // echo location coefficient (between 0 and 1). This scales the pre-phase gradient lobe
        let echo_location = 0.5;

        // readout moment
        let ro_time:Time = (s_dt.scale(mgre.matrix_size[0]) + t_ramp).try_into().unwrap();
        let ro_moment = ro_time * gro;

        let pe_moment = - ro_moment * echo_location;

        // base prephase lobe strength
        let gpe_x:FieldGrad = (pe_moment / pe_dt).try_into().unwrap();

        // echo readout
        let c_gro = EventControl::<FieldGrad>::new().with_constant_grad(gro).to_shared();
        // pre-phasing and p-space sampling. This is a constant pre-phase gradient modulated

        let c_gpe_x = EventControl::<FieldGrad>::new()
            .with_constant_grad(gpe_x)
            .with_adj("gpre")
            .to_shared();

        let c_gx_re = EventControl::<FieldGrad>::new()
            .with_constant_grad(gpe_x.scale(2))
            .with_adj("gpx_re").to_shared();

        // phase encoding steps for y and z axes

        let s = read_to_string(mgre.pe_table.as_ref().expect("cs table must be defined")).expect("invalid cs table file");
        let mut cs_samples:Vec<i32> = s.lines().map(|idx| idx.parse::<i32>()
            .expect("failed to parse cs table")).collect();

        let coords = cs_samples.chunks_exact(2).collect::<Vec<&[i32]>>();

        let mut pe_steps_y = vec![];
        let mut pe_steps_z = vec![];

        for _ in 0..mgre.dummy_scans {
            pe_steps_y.push(0);
            pe_steps_z.push(0);
        }

        coords.iter().for_each(|pair|{
            pe_steps_y.push(pair[0]);
            pe_steps_z.push(pair[1]);
        });

        if mgre.gop_mode {
            pe_steps_y.fill(0);
            pe_steps_z.fill(0);
        }

        mgre.n_phase_y = Some(pe_steps_y.len());
        mgre.n_phase_z = Some(pe_steps_z.len());

        let lut_pe_y = LUT::new("pe_y_lut",&pe_steps_y).to_shared();
        let lut_pe_z = LUT::new("pe_z_lut",&pe_steps_z).to_shared();

        let c_gpe_y = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_y)
            .with_source_loop(VIEW)
            .with_grad_scale(gpe_y)
            .to_shared();

        let c_gpe_z = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_z)
            .with_source_loop(VIEW)
            .with_grad_scale(gpe_z)
            .to_shared();

        // user-adjustable rf power
        let c_rfp = EventControl::<f64>::new().with_adj(RF_POWER).to_shared();

        let c_gs = EventControl::<FieldGrad>::new()
            .with_constant_grad(FieldGrad::mt_per_meter(mgre.spoiler_grad_mtpm)).to_shared();

        EventControllers {
            c_gro,
            c_gx_re,
            c_gpe_x,
            c_gpe_y,
            c_gpe_z,
            c_rfp,
            c_gs,
        }

    }
}

struct Events {
    e_alpha: RfEvent,
    e_pe: GradEvent,
    e_ro_ru: GradEvent,
    e_acq: ACQEvent,
    e_ro_rd: GradEvent,
    e_ro_ru2: GradEvent,
    e_acq2: ACQEvent,
    e_ro_rd2: GradEvent,
    e_pe_re: GradEvent,
    e_spoil: GradEvent,
}

impl Events {

    fn build(mgre: &mut Mgre) -> Events {
        let w = Waveforms::build(mgre);
        let ec = EventControllers::build(mgre);
        let e_alpha = RfEvent::new(Events::alpha(),&w.rf,&ec.c_rfp);
        let e_pe = GradEvent::new(Events::pe())
            .with_x(&w.pe).with_y(&w.pe).with_z(&w.pe)
            .with_strength_x(&ec.c_gpe_x).with_strength_y(&ec.c_gpe_y).with_strength_z(&ec.c_gpe_z);

        let e_pe_re = GradEvent::new(Events::pe_re()).with_x(&w.pe).with_strength_x(&ec.c_gx_re);

        // readout ramps for echo 1
        let e_ro_ru = GradEvent::new(Events::ro_ru()).with_x(&w.ru).with_strength_x(&ec.c_gro);
        let e_ro_rd = GradEvent::new(Events::ro_rd()).with_x(&w.rd).with_strength_x(&ec.c_gro);
        let e_acq = ACQEvent::new(Events::acq(),mgre.matrix_size[0],Freq::khz(mgre.bandwidth_khz).inv());
        let e_spoil = GradEvent::new(Events::spoil())
            .with_x(&w.pe).with_strength_x(&ec.c_gs)
            .with_y(&w.pe).with_strength_y(&ec.c_gs)
            .with_z(&w.pe).with_strength_z(&ec.c_gs);

        let e_ro_ru2 = e_ro_ru.clone_with_label(Events::ro_ru2());
        let e_ro_rd2 = e_ro_rd.clone_with_label(Events::ro_rd2());
        let e_acq2 = e_acq.clone_with_label(Events::acq2());

        Events {
            e_alpha,
            e_pe,
            e_ro_ru,
            e_ro_rd,
            e_ro_ru2,
            e_ro_rd2,
            e_acq2,
            e_acq,
            e_spoil,
            e_pe_re,
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

    fn ro_ru2() -> &'static str {
        "ro_ru2"
    }

    fn ro_rd() -> &'static str {
        "ro_rd"
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

    fn spoil() -> &'static str {
        "spoil"
    }

    fn pe_re() -> &'static str {
        "pe_re"
    }

}