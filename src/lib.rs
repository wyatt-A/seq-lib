pub mod rf_pulses;
pub mod grad_pulses;
pub mod q_calc;


use std::collections::HashMap;
use std::path::{Path, PathBuf};
use clap::{Parser};
pub use seq_struct;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{Angle, FieldGrad, Freq, Length, Time};
use mr_units::quantity::Unit;
use seq_struct::acq_event::ACQEvent;
use seq_struct::compile::{Seq, Timeline};
use seq_struct::grad_strength::EventControl;
use seq_struct::gradient_event::GradEvent;
use seq_struct::rf_event::RfEvent;
use seq_struct::seq_loop::SeqLoop;
use seq_struct::waveform::Waveform;
use crate::rf_pulses::{hardpulse, hardpulse_composite, sinc5};
use headfile::Headfile;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;

#[derive(clap::Parser)]
pub struct Args {

    /// parameter file to read or write
    pub param_file: PathBuf,

    /// initialize parameters with default values
    #[clap(long)]
    pub init: bool,

    /// open parameter editor
    #[clap(long)]
    pub edit: bool,
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

#[derive(Parser, Debug)]
pub struct InputArgs {
    /// Input pulse sequence parameters
    pub input: PathBuf,

    /// Output sequence file path
    pub output: PathBuf,

    /// Write default pulse sequence settings
    #[arg(short, long)]
    pub default: bool,
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

pub trait ToHeadfile {
    fn headfile(&self) -> Headfile;
}

pub trait TOML: Serialize + DeserializeOwned {
    fn to_toml(&self) -> toml::Value {
        let toml_str = self.to_toml_str();
        toml::from_str(&toml_str).unwrap()
    }
    fn to_toml_str(&self) -> String {
        toml::to_string_pretty(self).unwrap()
    }
}