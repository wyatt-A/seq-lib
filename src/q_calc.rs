// calculate q-space space values based on G and inversion pulses

use mr_units::constants::Nucleus;
use mr_units::quantity::Unit;

/// calculate the phase accumulation q(t) from G(t) or G_eff(t) (T s m^-1)
pub fn calc_phase(g:&[f64], t:&[f64], q:&mut [f64]) {
    assert_eq!(g.len(), t.len());
    assert_eq!(g.len(), q.len());
    cumtrapz(t, g, q);
}


/// calculate the b-factor of a nucleus given the square of phase accumulation q(t), some echo time t_end
pub fn calc_b_factor(q_sq:&[f64], t:&[f64], t_end:f64, nuc:Nucleus) -> Option<f64> {
    let t_idx = t.iter().position(|&ti| ti >= t_end)?;
    // gamma^2 * integral of q^2 wrt time
    Some(
        trapz(&t[0..t_idx],&q_sq[0..t_idx]) * nuc.gamma().si().powi(2)
    )
}

/// calculates the b-matrix of some nucleus given Gx(t), Gy(t), Gz(t), and t_end. The order of the
/// symmetric matrix elements are `[bxx,byy,bzz,bxy,bxz,byz]`. G(t) is assumes to have units T/m ,
/// and time in seconds. The result is in standard s * mm^-2 units
pub fn calc_b_matrix(gx:&[f64], gy:&[f64], gz:&[f64], t:&[f64], t_inv:&[f64], t_end:f64, nuc:Nucleus) -> [f64;6] {

    let mut qx = vec![0.; gx.len()];
    let mut qy = vec![0.; gy.len()];
    let mut qz = vec![0.; gz.len()];

    let mut gex = vec![0.; gx.len()];
    let mut gey = vec![0.; gy.len()];
    let mut gez = vec![0.; gz.len()];

    calc_g_effective(gx,t,t_inv,&mut gex);
    calc_g_effective(gy,t,t_inv,&mut gey);
    calc_g_effective(gz,t,t_inv,&mut gez);

    calc_phase(&gex,t,&mut qx);
    calc_phase(&gey,t,&mut qy);
    calc_phase(&gez,t,&mut qz);

    // calculate all cross and square terms

    // cross term temp variable for re-use
    let mut tmp = vec![0.;qx.len()];

    tmp.iter_mut().zip(qx.iter()).for_each(|(t,&q)| *t = q * q);
    let bxx = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    tmp.iter_mut().zip(qy.iter()).for_each(|(t,&q)| *t = q * q);
    let byy = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    tmp.iter_mut().zip(qz.iter()).for_each(|(t,&q)| *t = q * q);
    let bzz = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    tmp.iter_mut().zip(qx.iter().zip(qy.iter())).for_each(|(t,(&x,&y))| *t = x * y);
    let bxy = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    tmp.iter_mut().zip(qx.iter().zip(qz.iter())).for_each(|(t,(&x,&z))| *t = x * z);
    let bxz = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    tmp.iter_mut().zip(qy.iter().zip(qz.iter())).for_each(|(t,(&y,&z))| *t = y * z);
    let byz = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    [bxx,byy,bzz,bxy,bxz,byz]

}


/// calculate the effective gradient experienced by spins given a series of inversion pulses
pub fn calc_g_effective(g:&[f64], t:&[f64], t_inv:&[f64], g_eff:&mut [f64]) {

    assert_eq!(g.len(), t.len());
    assert_eq!(g.len(), g_eff.len());

    // Make a sorted copy of inversion times
    let mut t_inv = t_inv.to_owned();
    t_inv.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Current sign and index into t_inv
    let mut inv_idx = 0;
    let mut sign = 1.0; // +1 before the first inversion

    // Walk forward in time
    for ((g_eff_i, g_i), t_i) in g_eff.iter_mut().zip(g.iter()).zip(t.iter()) {
        // Flip sign for each inversion pulse that has already occurred
        while inv_idx < t_inv.len() && *t_i >= t_inv[inv_idx] {
            sign = -sign;
            inv_idx += 1;
        }
        *g_eff_i = sign * *g_i;
    }

}

/// cumulative trapezoidal integration
pub fn cumtrapz(t: &[f64], x: &[f64], y:&mut [f64]) {
    assert_eq!(
        t.len(),
        x.len(),
        "t and x must have the same length (got {} and {})",
        t.len(),
        x.len()
    );

    let n = t.len();

    // Empty input → empty output
    if n == 0 {
        return
    }

    y[0] = 0.0;

    for i in 1..n {
        let dt = t[i] - t[i - 1];
        let area = 0.5 * (x[i - 1] + x[i]) * dt;
        let cum = y[i - 1] + area;
        y[i] = cum;
        //y.push(cum);
    }

    //y
}

/// trapezoid method for numerical integration
pub fn trapz(t: &[f64], x: &[f64]) -> f64 {
    assert_eq!(
        t.len(),
        x.len(),
        "t and x must have the same length (got {} and {})",
        t.len(),
        x.len()
    );

    let n = t.len();
    if n < 2 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    for i in 1..n {
        let dt = t[i] - t[i - 1];
        sum += 0.5 * (x[i - 1] + x[i]) * dt;
    }
    sum
}