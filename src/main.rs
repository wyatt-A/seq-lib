use seq_lib::{ssme, PulseSequence};

fn main() {
    let p = ssme::SSMEParams::default();
    let state = p.adjustment_state();
    p.render_to_file(&state,"out.txt");
}
