use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::FieldGrad;
use mr_units::quantity::Unit;
use seq_lib::{dwi3d, ssme, PulseSequence};
use seq_lib::dwi3d::Dwi3DParams;
use seq_lib::q_calc::{calc_b_matrix, calc_g_effective};

fn main() {
    // let p = ssme::SSMEParams::default();
    // let state = p.adjustment_state();
    // p.render_to_file(&state,"out.txt");

    let mut p = Dwi3DParams::default();

    let mut state = p.adjustment_state();
    *state.get_mut("diff_x").unwrap() = FieldGrad::tesla_per_meter(2.17).si();
    *state.get_mut("rf90_pow").unwrap() = 1.;
    *state.get_mut("rf180_pow").unwrap() = 2.;

    // *state.get_mut("diff_y").unwrap() = FieldGrad::tesla_per_meter(0).si();
    // *state.get_mut("diff_z").unwrap() = FieldGrad::tesla_per_meter(0).si();
    //
    // //let te = Dwi3DParams::report_echo_time(&p.compile()).as_sec();
    //
    // let seq = p.render(&state);
    //
    // let t:Vec<_> = seq.iter().map(|[t,..]| *t).collect();
    // let gx:Vec<_> = seq.iter().map(|[_,gx,..]| *gx).collect();
    // let gy:Vec<_> = seq.iter().map(|[_,_,gy,..]| *gy).collect();
    // let gz:Vec<_> = seq.iter().map(|[_,_,_,gz,..]| *gz).collect();
    //
    // // times when spins are considered inverted
    // let t_inv = [0.008942];
    // // echo time - the upper limit of integration
    // let t_echo = 0.01556;
    //
    // let mat = calc_b_matrix(&gx,&gy,&gz,&t,&t_inv,t_echo,Nuc1H);
    //
    // println!("b-mat: {:#?}",mat);
    //println!("{:?}",gx);
    p.render_to_file(&state,"out.ps");
    
}
