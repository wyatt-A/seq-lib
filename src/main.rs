use seq_lib::{dwi3d, ssme, PulseSequence};

fn main() {
    // let p = ssme::SSMEParams::default();
    // let state = p.adjustment_state();
    // p.render_to_file(&state,"out.txt");

    let p = dwi3d::Dwi3DParams::default();
    let state = p.adjustment_state();
    p.render_to_file(&state,"out.ps");
    
}
