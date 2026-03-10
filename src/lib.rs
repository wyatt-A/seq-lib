use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use clap;
pub use seq_struct;
use seq_struct::compile::{Seq, Timeline};
use seq_struct::seq_loop::SeqLoop;
use headfile::Headfile;
use mrs_ppl::compile::{build_ppl, compile_ppl};
use pulse_seq_view::run_viewer;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub mod rf_pulses;
pub mod grad_pulses;
pub mod q_calc;

#[derive(clap::Parser)]
pub struct Args {

    /// base directory to write files
    #[arg(required_unless_present_any = ["init"])]
    pub base_dir: Option<PathBuf>,

    /// parameter file to read or write. Not required if finish is called (param file is in the acq directory)
    #[arg(required_unless_present_any = ["finish"])]
    #[clap(short, long)]
    pub param_file: Option<PathBuf>,

    /// display pulse sequence
    #[clap(long)]
    pub display: bool,

    /// initialize parameters with default values
    #[clap(long)]
    pub init: bool,

    /// build setup routine for pulse sequence
    #[clap(long)]
    pub setup: bool,

    /// build acquisition routine for pulse sequence
    #[clap(long)]
    pub acquire: bool,

    /// run finish routine after acquire to write headfile and other resources
    #[clap(long)]
    pub finish: bool,

    /// compile ppl for MRS simulation software
    #[clap(long)]
    pub sim: bool,

    /// do not compile ppl. Useful for testing without the scanner
    #[clap(short,long)]
    pub skip_ppl_compile: bool
}

impl Args {
    pub fn setup_dir(&self) -> PathBuf {
        self.base_dir.as_ref().unwrap().join("setup")
    }
    pub fn acq_dir(&self) -> PathBuf {
        self.base_dir.as_ref().unwrap().join("acq")
    }
    pub fn sim_dir(&self) -> PathBuf {
        self.base_dir.as_ref().unwrap().join("sim")
    }

    pub fn run<T:PulseSequence>(self) {


        if self.init {
            let param_file = self.param_file.as_ref().expect("param file is not defined");
            let gre = T::default();
            gre.to_file(param_file);
            return
        }

        let setup_dir = self.setup_dir();
        let acq_dir = self.acq_dir();
        let sim_dir = self.sim_dir();

        if self.display {
            let param_file = self.param_file.as_ref().expect("param file is not defined");
            let mut gre = T::from_file(param_file);
            gre.display_mode();
            let seq_loop = gre.build_sequence();
            let user_state = gre.adjustment_state();
            let ps_data = seq_loop.render_timeline(&user_state).to_raw();
            run_viewer(ps_data).unwrap();
            return
        }

        if self.setup {
            let param_file = self.param_file.as_ref().expect("param file is not defined");
            let mut gre = T::from_file(param_file);
            gre.gop_mode();
            let seq_loop = gre.build_sequence();
            create_dir_all(&setup_dir).unwrap();
            gre.to_file(setup_dir.join(T::seq_name()));
            let hf = gre.headfile();
            hf.to_file(&setup_dir.join(format!("{}_setup",T::seq_name()))).unwrap();
            if self.skip_ppl_compile {
                return
            }
            build_ppl(&seq_loop, &setup_dir, T::seq_name(), false);
            compile_ppl(&setup_dir);
            return
        }

        if self.acquire {
            let param_file = self.param_file.as_ref().expect("param file is not defined");
            let mut gre = T::from_file(param_file);
            gre.acq_mode();
            let seq_loop = gre.build_sequence();
            create_dir_all(&acq_dir).unwrap();
            build_ppl(&seq_loop, &acq_dir, T::seq_name(), false);
            gre.to_file(acq_dir.join(T::seq_name()));
            if self.skip_ppl_compile {
                return
            }
            compile_ppl(&acq_dir);
            // copy ppl params
            // run acquisition
            return
        }

        if self.finish {
            let mut gre = T::from_file(acq_dir.join(T::seq_name()));
            gre.finish_acquisition(&acq_dir);
            gre.to_file(acq_dir.join(T::seq_name()));
            let mut hf = gre.headfile();
            hf.write_timestamp();
            hf.to_file(&acq_dir.join(T::seq_name())).unwrap();
            return
        }

        if self.sim {
            let param_file = self.param_file.as_ref().expect("param file is not defined");
            let mut gre = T::from_file(param_file);
            gre.sim_mode();
            let seq_loop = gre.build_sequence();
            create_dir_all(&sim_dir).unwrap();
            gre.to_file(sim_dir.join(T::seq_name()));
            if self.skip_ppl_compile {
                return
            }
            build_ppl(&seq_loop, &sim_dir, T::seq_name(), true);
            compile_ppl(&sim_dir);
            return
        }
    }



}

pub mod defs {
    use std::rc::Rc;
    use mr_units::primitive::{Angle, FieldGrad};
    use seq_struct::grad_strength::EventControl;
    use seq_struct::rf_pulse::RfPulse;
    use seq_struct::waveform::Waveform;

    pub const VIEW:&str = "view";
    /// Gradient waveform
    pub type GW = Rc<Waveform>;
    /// Rf pulse
    pub type RF = Rc<RfPulse>;
    /// Gradient strength controller
    pub type GS = Rc<EventControl<FieldGrad>>;
    /// Rf power controller
    pub type RFP = Rc<EventControl<f64>>;
    /// Rf phase controller
    pub type RFPhase = Rc<EventControl<Angle>>;

    // loop names
    pub const ECHO:&str = "echo";
    pub const EXPERIMENT:&str = "experiment";
    pub const SLICE:&str = "slice";

    // rf power adj
    pub const RF_POWER:&str = "rf_power";
}

/// Specifies a data structure that compiles to a pulse sequence
pub trait PulseSequence: Default + ToHeadfile {
    /// main pulse sequence generation routine. This is where all the pulse sequence logic is
    /// implemented
    fn build_sequence(&mut self) -> SeqLoop;
    fn adjustment_state(&self) -> HashMap<String,f64>;

    /// run a finishing routine to format data and calculate additional parameters
    fn finish_acquisition(&mut self, acq_dir: impl AsRef<Path>);

    /// set the pulse sequence to gain optimization mode (setup mode)
    fn gop_mode(&mut self);
    /// set the pulse sequence to acquire mode
    fn acq_mode(&mut self);
    /// set the pulse sequence to display mode for plotting
    fn display_mode(&mut self);
    /// set the pulse sequence to simulation mode for MRS hardware simulation
    fn sim_mode(&mut self);

    fn render_timeline(&mut self,state:&HashMap<String,f64>) -> Timeline {
        self.build_sequence().render_timeline(state)
    }

    /// render sequence timeline `[t,Gx,Gy,Gz,Bx,By,rec]` where t (sec), G (T/m), B (T), rec (rad)
    fn render(&mut self,state:&HashMap<String,f64>) -> Seq {
        self.build_sequence().render_timeline(state).render()
    }

    fn render_to_file(&mut self, state:&HashMap<String,f64>, filename:impl AsRef<Path>) {
        self.build_sequence().render_timeline(state).write_to_file(filename)
    }

}

pub trait ToHeadfile: TOML {
    fn headfile(&self) -> Headfile {
        let t = self.to_toml();
        let tab = t.as_table().expect("failed to get toml table");
        let mut h = Headfile::new();
        h.insert_toml_table(&tab,false);
        h
    }

    fn seq_name() -> &'static str;
}

pub trait TOML: Serialize + DeserializeOwned {

    fn to_file(&self,filename: impl AsRef<Path>) {
        let mut f = File::create(filename.as_ref().with_extension("toml")).unwrap();
        f.write_all(self.to_toml_str().as_bytes()).unwrap();
    }

    fn from_file(filename:impl AsRef<Path>) -> Self {
        let mut f = File::open(filename.as_ref().with_extension("toml")).unwrap();
        let toml_str = std::io::read_to_string(&mut f).unwrap();
        toml::from_str(&toml_str).unwrap()
    }

    fn to_toml(&self) -> toml::Value {
        let toml_str = self.to_toml_str();
        toml::from_str(&toml_str).unwrap()
    }
    fn to_toml_str(&self) -> String {
        toml::to_string_pretty(self).unwrap()
    }
}