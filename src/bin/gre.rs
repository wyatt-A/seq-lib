use std::collections::HashMap;
use std::fs::{create_dir_all, read_to_string};
use std::path::{PathBuf, Path};
use array_lib::{ArrayDim, NormSqr};
use array_lib::io_nifti::write_nifti;
use clap::Parser;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::rs_fft::rs_fftn_batched;
use headfile::Headfile;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use mrs_ppl::compile::{build_ppl,compile_ppl};
use num_complex::Complex32;
use pulse_seq_view::run_viewer;
use seq_struct::acq_event::ACQEvent;
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::variable::LUT;
use serde::{Deserialize, Serialize};
use seq_lib::defs::{EXPERIMENT, GS, GW, RF, RFP, RF_POWER, VIEW};
use seq_lib::grad_pulses::{half_sin, quarter_sin_rd, quarter_sin_ru, ramp_down, ramp_up, trapezoid};
use seq_lib::{rf_pulses, Args, PulseSequence, ToHeadfile, TOML};
use seq_lib::grad_pulses::scale_factors::{HALF_SIN_SCALE, QUARTER_SIN_SCALE};

const SEQ_NAME: &str = "gre";

fn main() {

    let args = Args::parse();

    let setup_dir = args.setup_dir();
    let acq_dir = args.acq_dir();
    let sim_dir = args.sim_dir();

    if args.init {
        let gre = Gre::default();
        gre.to_file(args.param_file);
        return
    }

    if args.display {
        let mut gre = Gre::from_file(args.param_file);
        gre.gop_mode = false;
        gre.sim_mode = true;
        let seq_loop = gre.build_sequence();
        let user_state = gre.adjustment_state();
        let ps_data = seq_loop.render_timeline(&user_state).to_raw();
        run_viewer(ps_data).unwrap();
        return
    }

    if args.setup {
        let mut gre = Gre::from_file(args.param_file);
        gre.gop_mode = true;
        let seq_loop = gre.build_sequence();
        create_dir_all(&setup_dir).unwrap();
        build_ppl(&seq_loop, &setup_dir, SEQ_NAME, false);
        gre.to_file(setup_dir.join(SEQ_NAME));
        let hf = gre.headfile();
        hf.to_file(&setup_dir.join(format!("{SEQ_NAME}_setup"))).unwrap();
        if args.skip_ppl_compile {
            return
        }
        compile_ppl(&setup_dir);
        return
    }

    if args.acquire {
        let mut gre = Gre::from_file(args.param_file);
        gre.gop_mode = false;
        let seq_loop = gre.build_sequence();
        create_dir_all(&acq_dir).unwrap();
        build_ppl(&seq_loop, &acq_dir, SEQ_NAME, false);
        gre.to_file(acq_dir.join(SEQ_NAME));
        if args.skip_ppl_compile {
            return
        }
        compile_ppl(&acq_dir);
        // copy ppl params
        // run acquisition
        return
    }

    if args.finish {
        let mut gre = Gre::from_file(acq_dir.join(SEQ_NAME));
        finish_acquisition(acq_dir.join(format!("{SEQ_NAME}.mrd")), acq_dir.join(SEQ_NAME), &mut gre);
        gre.to_file(acq_dir.join(SEQ_NAME));
        let mut hf = gre.headfile();
        hf.write_timestamp();
        hf.to_file(&acq_dir.join(SEQ_NAME)).unwrap();
        return
    }

    if args.sim {
        let mut gre = Gre::from_file(args.param_file);
        gre.gop_mode = false;
        gre.sim_mode = true;
        let seq_loop = gre.build_sequence();
        create_dir_all(&sim_dir).unwrap();
        gre.to_file(sim_dir.join(SEQ_NAME));
        build_ppl(&seq_loop, &sim_dir, SEQ_NAME, true);
        compile_ppl(&sim_dir);
        return
    }

}


#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Gre {
    /// references the pulse sequence bin used to run the protocol
    seq_name: String,
    /// field of view in mm
    fov_mm: [f64;3],
    /// size of acquisition grid where `matrix[0]` is the number of readout samples
    matrix_size: [usize;3],
    /// number of p-space samples to acquire multiple echo times
    n_pspace_samples: usize,
    /// step size to modulate the pre-phase gradient in millitesla per meter
    pspace_step_size_mtpm: f64,
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
    te_estimate_ms: Option<f64>,
    /// gradient strength limit
    grad_limit_tpm: f64,
    /// time delta between center of rf pulse and start of ADC
    acq_start_ms: Option<f64>,
    /// measured time between echoes acquired across p-space
    delta_te_measured_ms: Option<Vec<f64>>,
    /// measured time of first echo based on dc sample and acq_start
    te_measured_ms: Option<f64>,
    /// readout sample index of k0 (DC)
    dc_indices:Option<Vec<isize>>
}

impl Default for Gre {
    fn default() -> Self {
        Gre {
            seq_name: SEQ_NAME.to_string(),
            fov_mm: [20.,12.8,12.8],
            matrix_size: [512,256,256],
            n_pspace_samples: 3,
            pspace_step_size_mtpm: 1.,
            bandwidth_khz: 50.,
            pe_table: None,
            n_phase_y: None,
            n_phase_z: None,
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
            grad_limit_tpm: 2.0,
            te_estimate_ms: None,
            acq_start_ms: None,
            delta_te_measured_ms: None,
            te_measured_ms: None,
            dc_indices: None,
        }
    }
}


const SAFE_INSERT:bool = true;
const UNSAFE_INSERT:bool = false;

impl ToHeadfile for Gre {
    fn headfile(&self) -> Headfile {
        let param_table = self.to_toml();
        let mut h = Headfile::new();
        h.dim_x(self.matrix_size[0]);
        h.dim_y(self.matrix_size[1]);
        h.dim_z(self.matrix_size[2]);
        h.ne(self.n_pspace_samples);
        h.fov_x(self.fov_mm[0]);
        h.fov_y(self.fov_mm[1]);
        h.fov_z(self.fov_mm[2]);
        h.ne(self.n_pspace_samples);
        h.insert_scalar("pspace_stepsize_mtpm", self.pspace_step_size_mtpm, UNSAFE_INSERT);
        h.insert_toml_table(param_table.as_table().expect("parameters must be a table"), SAFE_INSERT);
        h.bw(self.bandwidth_khz * 1e3 / 2.);
        //h.insert_scalar("te_estimate_ms", self.te_estimate_ms.unwrap(), UNSAFE_INSERT);
        h.tr((self.rep_time_ms * 1e3) as usize);
        if let Some(t) = self.te_estimate_ms {
            h.te(t);
        }
        if let Some(t) = self.delta_te_measured_ms.as_ref() {
            h.insert_list_1d("delta_te",t, UNSAFE_INSERT);
        }
        h
    }
}

impl TOML for Gre {}

impl PulseSequence for Gre {
    fn build_sequence(&mut self) -> SeqLoop {

        let tr = Time::ms(self.rep_time_ms);
        let pre_calc_time = Time::ms(4);
        let post_alpha_del = Time::us(self.post_alpha_del_us);
        let post_pe_del = Time::us(self.post_pe_del_us);
        let post_ro_ramp_del = Time::us(self.post_ru_del_us);
        let post_acq_del = Time::us(self.post_acq_del_us);

        let e = Events::build(self);

        let n_views = if self.sim_mode {
            10
        }else {
            // default to 4 views if the number of phase encoding steps has not been set
            let n_phase_y = self.n_phase_y.unwrap_or(2);
            let n_phase_z = self.n_phase_z.unwrap_or(2);
            assert_eq!(n_phase_y, n_phase_z);
            n_phase_y
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
            self.post_alpha_del_us = time.as_us();
            println!("set post-alpha delay to {:?} us", self.post_alpha_del_us);
        }else {
            panic!("failed to set post-alpha delay");
        }

        if let Ok(time) = vl.set_min_time_span(Events::pe(),Events::ro_ru(),100,0,post_pe_del) {
            self.post_pe_del_us = time.as_us();
            println!("set post-phase encode delay to {:?} us", self.post_pe_del_us);
        }

        if let Ok(time) = vl.set_min_time_span(Events::ro_ru(),Events::acq(),100,0,post_ro_ramp_del) {
            self.post_ru_del_us = time.as_us();
            println!("set post-readout ramp delay to {:?} us", self.post_ru_del_us);
        }

        if let Ok(time) = vl.set_min_time_span(Events::acq(),Events::ro_rd(),100,0,post_acq_del) {
            self.post_acq_del_us = time.as_us();
            println!("set post-acq delay to {:?} us", self.post_acq_del_us);
        }

        vl.set_averages(1);
        vl.set_rep_time(tr).unwrap();

        // record the start time of the first sample relative to the center of the excitation pulse
        self.acq_start_ms = Some(vl.get_time_span(Events::alpha(),Events::acq(),50,0).unwrap().as_ms());


        // calculate echo times
        // time from center of alpha pulse to end of readout ramp up
        let t0 = vl.get_time_span(Events::alpha(),Events::ro_ru(),50,100).unwrap();
        let t1 = vl.get_time_span(Events::alpha(),Events::ro_rd(),50,0).unwrap();
        // first echo forms between ru and rd, parameterized by echo location
        let te1 = (t1.as_ms() - t0.as_ms()) * self.echo_location + t0.as_ms();
        self.te_estimate_ms = Some(te1);

        expl.add_loop(vl).unwrap();
        expl.set_pre_calc(Time::us(50));
        //expl.no_overhead();

        // return loop and modified parameters
        expl

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
    fn build(gre: &mut Gre) -> EventControllers {

        let nuc = Nuc1H;
        let fov_x = Length::mm(gre.fov_mm[0]);
        let fov_y = Length::mm(gre.fov_mm[1]);
        let fov_z = Length::mm(gre.fov_mm[2]);
        let s_dt:Time = Freq::khz(gre.bandwidth_khz).inv().into();

        // p_space index steps. This will be multiplied by a gradient strength and added to the pre-phase lobe
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

        let gpe_z = FieldGrad::from_fov(fov_z,pe_dt,nuc);

        // echo location coefficient (between 0 and 1). This scales the pre-phase gradient lobe
        let echo_location = gre.echo_location.clamp(0., 1.);

        // readout moment
        let ro_time:Time = (s_dt.scale(gre.matrix_size[0]) + t_ramp).try_into().unwrap();
        let ro_moment = ro_time * gro;

        let pe_moment = - ro_moment * echo_location;

        // base prephase lobe strength
        let gpe_x:FieldGrad = (pe_moment / pe_dt).try_into().unwrap();
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

        let s = read_to_string(gre.pe_table.as_ref().expect("cs table must be defined")).expect("invalid cs table file");
        let cs_samples:Vec<i32> = s.lines().map(|idx| idx.parse::<i32>()
            .expect("failed to parse cs table")).collect();

        let mut pe_steps_y = vec![];
        let mut pe_steps_z = vec![];
        cs_samples.chunks_exact(2).for_each(|pair|{
            pe_steps_y.push(pair[0]);
            pe_steps_z.push(pair[1]);
        });

        if gre.gop_mode {
            pe_steps_y.fill(0);
            pe_steps_z.fill(0);
        }

        gre.n_phase_y = Some(pe_steps_y.len());
        gre.n_phase_z = Some(pe_steps_z.len());

        let max_y = pe_steps_y.iter().map(|x| x.abs()).max().unwrap();
        let max_z = pe_steps_z.iter().map(|x| x.abs()).max().unwrap();

        let gpe_y_max = gpe_y.scale(max_y);
        assert!(gpe_y_max.si().abs() <= gre.grad_limit_tpm, "gpy exceeds max gradient strength: {}",gpe_y_max.si());
        let gpe_z_max = gpe_z.scale(max_z);
        assert!(gpe_z_max.si().abs() <= gre.grad_limit_tpm, "gpz exceeds max gradient strength: {}",gpe_z_max.si());

        let lut_pe_y = LUT::new("pe_y_lut",&pe_steps_y).to_shared();
        let lut_pe_z = LUT::new("pe_z_lut",&pe_steps_z).to_shared();

        let c_gpe_y = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_y)
            .with_source_loop(VIEW)
            //.with_mod(gre.n_phase_y as i32)
            .with_grad_scale(gpe_y)
            .to_shared();

        let c_gpe_z = EventControl::<FieldGrad>::new()
            .with_lut(&lut_pe_z)
            .with_source_loop(VIEW)
            //.with_div_op(gre.n_phase_y as i32)
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

    fn build(mgre: &mut Gre) -> Events {
        let w = Waveforms::build(mgre);
        let ec = EventControllers::build(mgre);
        let e_alpha = RfEvent::new(Events::alpha(),&w.rf,&ec.c_rfp);
        let e_pe = GradEvent::new(Events::pe())
            .with_x(&w.pe).with_y(&w.pe).with_z(&w.pe)
            .with_strength_x(&ec.c_gpe_x).with_strength_y(&ec.c_gpe_y).with_strength_z(&ec.c_gpe_z);

        // readout ramps for echo 1
        let e_ro_ru = GradEvent::new(Events::ro_ru()).with_x(&w.ru).with_strength_x(&ec.c_gro);
        let e_ro_rd = GradEvent::new(Events::ro_rd()).with_x(&w.rd).with_strength_x(&ec.c_gro);
        let e_acq = ACQEvent::new(Events::acq(),mgre.matrix_size[0],Freq::khz(mgre.bandwidth_khz).inv());

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

/// format and analyze k-space data for recon. This fills in extra parameters
fn finish_acquisition(mrd_file:impl AsRef<Path>, out_cfl:impl AsRef<Path>, gre:&mut Gre) {

    // size of gridded data set
    let vol_data_dims = ArrayDim::from_shape(
        &[
            gre.matrix_size[0],
            gre.matrix_size[1],
            gre.matrix_size[2],
            gre.n_pspace_samples
        ]
    );

    // expected size of raw data from mrd file
    let raw_data_dims = ArrayDim::from_shape(
        &[
            gre.matrix_size[0],
            gre.n_phase_y.unwrap(),
            gre.n_pspace_samples,
        ]
    );

    let (raw_data, mrd_size, ..) = array_lib::io_mrd::read_mrd(mrd_file);
    assert_eq!(raw_data_dims.shape_squeeze(),mrd_size.shape_squeeze(),"unexpected raw data dimensions");

    // parse phase encode table
    let pe_table = gre.pe_table.as_ref().unwrap();
    let pe_table_str = read_to_string(pe_table).unwrap();
    let lines:Vec<&str> =  pe_table_str.lines().collect();
    let pe_table:Vec<[isize;2]> = lines.chunks_exact(2).map(|pair|{
        let y = pair[0].parse::<isize>().expect("failed to parse coordinate");
        let z = pair[1].parse::<isize>().expect("failed to parse coordinate");
        [y,z]
    }).collect();

    // allocate output data
    let mut vol_data = vol_data_dims.alloc(Complex32::ZERO);

    // find DC samples to determine echo times and spacings
    let mut dc_coords = vec![[0isize;3];gre.n_pspace_samples];
    let mut max_energy = vec![0f32;gre.n_pspace_samples];

    // grid raw data based on pe table
    let n_per_vol = raw_data_dims.shape()[0]*raw_data_dims.shape()[1];
    let n_per_line = raw_data_dims.shape()[0];
    // loop over echo data
    raw_data.chunks_exact(n_per_vol).enumerate().for_each(|(echo_idx,x)|{
        // loop over k-space lines
        x.chunks_exact(n_per_line).zip(pe_table.iter()).for_each(|(ksp_line,&[y,z])|{
            // loop over k-space samples
            ksp_line.iter().enumerate().for_each(|(x,sample)| {
                // record max energy sample and location
                if sample.norm_sqr() > max_energy[echo_idx] {
                    max_energy[echo_idx] = sample.norm_sqr();
                    dc_coords[echo_idx] = [x as isize,y,z];
                }
                // calculate address for k-space sample, and write into array
                let addr = vol_data_dims.calc_addr_signed(&[x as isize,y,z,echo_idx as isize]);
                vol_data[addr] = *sample;
            })
        })
    });

    let dwell_time = Freq::khz(gre.bandwidth_khz).inv();
    let te = gre.acq_start_ms.unwrap() + dwell_time.scale(dc_coords[0][0]).as_ms();
    let t_deltas = dc_coords.windows(2).enumerate().map(|(i,pair)|{
        // calculate sample difference between p-space samples
        let diff = pair[1][0] - pair[0][0];
        println!("echo diff {}: {diff} samples",i+1);
        dwell_time.scale(diff).as_ms()
    }).collect::<Vec<f64>>();

    println!("te_measured: {}",te);
    println!("t_deltas: {:#?}",t_deltas);

    gre.delta_te_measured_ms = Some(t_deltas);
    gre.te_measured_ms = Some(te);


    // set dc samples to first index of volume to be consistent with ifft

    // dims for single volume

    println!("shifting k-space for first-order phase correction");
    let vol_dim = ArrayDim::from_shape(&vol_data_dims.shape()[0..3]);
    let mut tmp_dst = vol_dim.alloc(Complex32::ZERO);
    let mut dc_indices = vec![];
    vol_data.chunks_exact_mut(vol_dim.numel()).enumerate().for_each(|(echo_idx,vol)|{
        // reverse-shift
        let shift = [
            -dc_coords[echo_idx][0],
            -dc_coords[echo_idx][1],
            -dc_coords[echo_idx][2]
        ];
        dc_indices.push(dc_coords[echo_idx][0]);
        // circshift vol into tmp, then write tmp back into vol
        tmp_dst.fill(Complex32::ZERO);
        vol_dim.circshift(&shift,vol,&mut tmp_dst);
        vol.copy_from_slice(&tmp_dst);
    });

    gre.dc_indices = Some(dc_indices);

    println!("writing ksp cfl");
    array_lib::io_cfl::write_cfl(&out_cfl,&vol_data,vol_data_dims);

    // do quick fft recon
    rs_fftn_batched(&mut vol_data,&vol_dim.shape_squeeze(),gre.n_pspace_samples,FftDirection::Inverse,NormalizationType::Unitary);

    // fft shift volumes
    vol_data.chunks_exact_mut(vol_dim.numel()).for_each(|vol|{
        // fftshift vol into tmp, then write tmp back into vol
        tmp_dst.fill(Complex32::ZERO);
        vol_dim.fftshift(&vol,&mut tmp_dst,true);
        vol.copy_from_slice(&tmp_dst);
    });

    println!("writing image cfl");
    let out_img = out_cfl.as_ref().with_file_name("img.cfl");
    array_lib::io_cfl::write_cfl(out_img,&vol_data,vol_data_dims);

    println!("writing phase images");
    vol_data.chunks_exact(vol_dim.numel()).enumerate().for_each(|(i,vol)|{
        let out_img = out_cfl.as_ref().with_file_name(format!("te_{i}.nii"));
        println!("{}",out_img.display());
        let phase:Vec<_> = vol.iter().map(|x| x.to_polar().1).collect();
        write_nifti(out_img,&phase,vol_dim);
    });

}