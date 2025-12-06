use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::FieldGrad;
use mr_units::quantity::Unit;
use seq_lib::{dwi3d, ssme, PulseSequence};
use seq_lib::dwi3d::Dwi3DParams;
use seq_lib::q_calc::{binary_solve, calc_b_matrix, calc_g_effective};
use seq_lib::rf_cal::RfCal;

#[test]
fn b_factor_example() {

    // instantiate diffusion weighted pulse sequence
    let mut p = Dwi3DParams::default();
    let mut s = p.adjustment_state();

    *s.get_mut("rf90_pow").unwrap() = 1.;
    *s.get_mut("rf180_pow").unwrap() = 2.;

    let mut f = |grad_strength:f64| {
        *s.get_mut("diff_x").unwrap() = FieldGrad::mt_per_meter(grad_strength).si();
        p.calc_b_matrix(&s).trace()
    };

    // find the x-gradient strength that produces target b-value
    let g = binary_solve(0.,3000.,3_000.,0.01,100,&mut f);

    println!("g = {}",g);
    println!("bval = {}",f(g));
    // render out sequence to file
    p.render_to_file(&s,"out");

}

#[test]
fn rf_cal_example() {
    let rfc = RfCal::default();
    let p = rfc.adjustment_state();
    rfc.render_to_file(&p,"rf_cal");
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
