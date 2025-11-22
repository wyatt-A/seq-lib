use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::FieldGrad;
use mr_units::quantity::Unit;
use seq_lib::{dwi3d, ssme, PulseSequence};
use seq_lib::dwi3d::Dwi3DParams;
use seq_lib::q_calc::{calc_b_matrix, calc_g_effective};


#[test]
fn b_factor_example() {

    // instantiate diffusion weighted pulse sequence
    let mut p = Dwi3DParams::default();

    // set diffusion gradient strength
    let mut state = p.adjustment_state();
    *state.get_mut("diff_x").unwrap() = FieldGrad::tesla_per_meter(2).si();

    // compile loop structure
    let vl = p.compile();

    // gather inversion times and echo time
    let t_inv:Vec<_> = vl.find_occurrences("rf180",50).into_iter().map(|t|t.as_sec()).collect();
    let t_echo = vl.find_occurrences("acq",50).get(0).unwrap().as_sec();

    // renders all pulses over sequence loop
    let seq = vl.render_timeline(&state).render();

    // extract gradient waveforms
    let t:Vec<_> = seq.iter().map(|[t,..]| *t).collect();
    let gx:Vec<_> = seq.iter().map(|[_,gx,..]| *gx).collect();
    let gy:Vec<_> = seq.iter().map(|[_,_,gy,..]| *gy).collect();
    let gz:Vec<_> = seq.iter().map(|[_,_,_,gz,..]| *gz).collect();

    // calculate b-matrix from waveforms
    let mat = calc_b_matrix(&gx,&gy,&gz,&t,&t_inv,t_echo,Nuc1H);
    println!("b-matrix[Bxx,Byy,Bzz,Bxy,Bxz,Byz]: {:?}", mat);

    // render out sequence to file
    p.render_to_file(&state,"out.ps");

}



fn main() {

    let mut p = Dwi3DParams::default();

    let mut state = p.adjustment_state();
    *state.get_mut("diff_x").unwrap() = FieldGrad::tesla_per_meter(0).si();
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


    
}
