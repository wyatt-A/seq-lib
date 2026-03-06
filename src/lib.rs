use std::collections::HashMap;
use std::fs::{create_dir_all, File};
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use clap;
pub use seq_struct;
use seq_struct::compile::{Seq, Timeline};
use seq_struct::seq_loop::SeqLoop;
use headfile::Headfile;
use serde::Serialize;
use serde::de::DeserializeOwned;

pub mod rf_pulses;
pub mod grad_pulses;
pub mod q_calc;

#[derive(clap::Parser)]
pub struct Args {

    /// parameter file to read or write
    pub param_file: PathBuf,

    /// base directory to write files
    pub base_dir: PathBuf,

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

    /// compile ppl for MRS simulation software
    #[clap(long)]
    pub sim: bool,

    /// run finish routine after acquire to write headfile and other resources
    #[clap(long)]
    pub finish: bool,

    /// do not compile ppl. Useful for testing without the scanner
    #[clap(short,long)]
    pub skip_ppl_compile: bool
}

impl Args {
    pub fn setup_dir(&self) -> PathBuf {
        self.base_dir.join("setup")
    }
    pub fn acq_dir(&self) -> PathBuf {
        self.base_dir.join("acq")
    }
    pub fn sim_dir(&self) -> PathBuf {
        self.base_dir.join("sim")
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
pub trait PulseSequence: Default {
    /// main pulse sequence generation routine. This is where all the pulse sequence logic is
    /// implemented
    fn build_sequence(&mut self) -> SeqLoop;
    fn adjustment_state(&self) -> HashMap<String,f64>;

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