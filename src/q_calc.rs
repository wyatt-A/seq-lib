// calculate q-space space values based on G and inversion pulses

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use mr_units::constants::Nucleus;
use mr_units::constants::Nucleus::Nuc1H;
use mr_units::primitive::{FieldGrad, Time};
use mr_units::quantity::Unit;
use nalgebra::{Matrix3, SymmetricEigen};
use seq_struct::compile::Seq;
use seq_struct::seq_loop::SeqLoop;
use crate::PulseSequence;

/// loads b-vec file with format (shell_idx, ux, uy, uz)
pub fn load_bvecs(file:impl AsRef<Path>) -> (Vec<usize>,Vec<[f64;3]>) {
    let mut f = File::open(file.as_ref()).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    let mut shell_idx = vec![];
    let bvecs:Vec<_> = s.lines().skip(1).map(|line| {
        let line_entries:Vec<_> = line.split_ascii_whitespace().map(|s|s.trim().parse::<f64>().unwrap()).collect();
        shell_idx.push(line_entries[0] as usize);
        [line_entries[1],line_entries[2],line_entries[3]]
    }).collect();
    (shell_idx,bvecs)
}

pub fn grad_solve<P:PulseSequence>(p:&P, adj_var:&str, target_bval:f64, g_lower:FieldGrad, g_upper:FieldGrad, inv_pulses_labels:&[&str], echo_time:Time) -> FieldGrad {

    let mut adj_state = p.adjustment_state();
    let s = p.compile();

    // get the center point of each inversion pulse
    let mut inv_pulse_times:Vec<_> = inv_pulses_labels.iter().flat_map(|inv_pulse|{
        s.find_occurrences(inv_pulse,50).into_iter().map(|t|t.as_sec())
    }).collect();
    inv_pulse_times.sort_by(|a,b|a.si().partial_cmp(&b.si()).unwrap());

    let mut f = |g| {
        *adj_state.get_mut(adj_var).unwrap() = g;
        let w = s.render_timeline(&adj_state).render();
        _calc_b_matrix(&w,&inv_pulse_times,echo_time.as_sec(), Nuc1H).trace()
    };

    let b = f(g_lower.si());
    if b > target_bval {
        println!("min b-value for min gradient strength is {b} s/mm^2 bval");
        g_lower
    }else if f(g_upper.si()) < target_bval {
        println!("unable to achieve target b-value {target_bval} s/mm^2 with max gradient strength {} T/m",g_upper.si());
        panic!("unable to achieve target b-value")
    }else {
        let tolerance = 1e-6;
        let max_iter = 100;
        let grad_soltn = binary_solve(g_lower.si(),g_upper.si(),target_bval,tolerance,max_iter,f);
        //println!("solved for gradient strength of {} mT/m", 1000. * grad_soltn);
        FieldGrad::tesla_per_meter(grad_soltn)
    }

}

pub fn calc_b_matrix<P:PulseSequence>(p:&P, adj_state:&HashMap<String,f64>, inv_pulses_labels:&[&str], echo_time:Time, nucleus: Nucleus) -> BMat {
    let s = p.compile();
    // get the center point of each inversion pulse
    let mut inv_pulse_times:Vec<_> = inv_pulses_labels.iter().flat_map(|inv_pulse|{
        s.find_occurrences(inv_pulse,50).into_iter().map(|t|t.as_sec())
    }).collect();
    inv_pulse_times.sort_by(|a,b|a.si().partial_cmp(&b.si()).unwrap());
    let seq = s.render_timeline(adj_state).render();
    _calc_b_matrix(&seq,&inv_pulse_times,echo_time.as_sec(), nucleus)
}

//    // define anonymous function to model the b-value as a function of the diffusion gradient along x-axis
//     let f = |g| {
//         *adj.get_mut("diff_x").unwrap() = g;
//         let w = s.render_timeline(&adj).render();
//         calc_b_matrix(&w,&t_inv,t_echoes[0],Nuc1H).trace()
//     };
//
//     let gmax = 2.5; // T/m
//     let bval = 7.; // s/mm^2
//     let tolerance = 1e-6; // s/mm^2
//     let max_iter = 100;
//     // solve for the input gradient strength to achieve desired b-value within the limits of the system
//     let grad_soltn = binary_solve(0.,gmax,bval,tolerance,max_iter,f);
//     println!("solved for gradient strength of {} mT/m", 1000. * grad_soltn);






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

#[derive(Debug)]
/// B-Matrix to store b-factors calculated from a pulse sequence (s/mm^2)
pub struct BMat {
    pub bxx: f64,
    pub byy: f64,
    pub bzz: f64,
    pub bxy: f64,
    pub bxz: f64,
    pub byz: f64,
}

impl BMat {
    /// return the trace (b-value) of the matrix (s/mm^2)
    pub fn trace(&self) -> f64 {
        self.bxx + self.byy + self.bzz
    }

    /// returns the normalized b-vector using eigenvalue decomposition
    pub fn b_vec(&self) -> [f64; 3] {
        let b = Matrix3::new(
            self.bxx, self.bxy, self.bxz,
            self.bxy, self.byy, self.byz,
            self.bxz, self.byz, self.bzz,
        );

        // Eigen-decomposition for symmetric matrix
        let se = SymmetricEigen::new(b);

        // Find index of the largest eigenvalue
        let mut max_idx = 0usize;
        let mut max_val = se.eigenvalues[0];
        for i in 1..3 {
            if se.eigenvalues[i] > max_val {
                max_val = se.eigenvalues[i];
                max_idx = i;
            }
        }

        // Corresponding eigenvector is the encoding direction
        let v = se.eigenvectors.column(max_idx).into_owned();

        // Normalize to unit length
        let n = v.norm();
        if n > 0.0 {
            let v = v / n;
            [v.x, v.y, v.z]
        } else {
            [v.x, v.y, v.z]
        }
    }

}

/// calculates the b-matrix of some nucleus given Gx(t), Gy(t), Gz(t), and t_end. The order of the
/// symmetric matrix elements are `[bxx,byy,bzz,bxy,bxz,byz]`. G(t) is assumes to have units T/m ,
/// and time in seconds. The result is in standard s * mm^-2 units
fn _calc_b_matrix(s:&Seq, t_inv:&[f64], t_end:f64, nuc:Nucleus) -> BMat {

    let t = &s.time_sec;
    let gx = &s.gx_tpm;
    let gy = &s.gy_tpm;
    let gz = &s.gz_tpm;

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

    // x-diagonal (square term)
    tmp.iter_mut().zip(qx.iter()).for_each(|(t,&q)| *t = q * q);
    let bxx = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    // y-diagonal (square term)
    tmp.iter_mut().zip(qy.iter()).for_each(|(t,&q)| *t = q * q);
    let byy = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    // z-diagonal (square term)
    tmp.iter_mut().zip(qz.iter()).for_each(|(t,&q)| *t = q * q);
    let bzz = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    // xy cross term
    tmp.iter_mut().zip(qx.iter().zip(qy.iter())).for_each(|(t,(&x,&y))| *t = x * y);
    let bxy = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    // xz cross term
    tmp.iter_mut().zip(qx.iter().zip(qz.iter())).for_each(|(t,(&x,&z))| *t = x * z);
    let bxz = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    // yz cross term
    tmp.iter_mut().zip(qy.iter().zip(qz.iter())).for_each(|(t,(&y,&z))| *t = y * z);
    let byz = calc_b_factor(&tmp,t,t_end,nuc).unwrap() * 1e-6;

    BMat { bxx, byy, bzz, bxy, bxz, byz }

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

/// Binary solver (bisection method) for monotonic functions.
/// Finds x in [low, high] such that f(x) ≈ target.
///
/// Assumptions:
/// - f is continuous and monotonic on [low, high]
/// - f(low) and f(high) bracket the target (i.e. (f(low)-target)*(f(high)-target) <= 0)
pub fn binary_solve<F>(
    mut low: f64,
    mut high: f64,
    target: f64,
    tol: f64,
    max_iter: usize,
    mut f: F,
) -> f64
where
    F: FnMut(f64) -> f64,
{
    // Optional: sanity check (can be removed for speed)
    let f_low = f(low) - target;
    let f_high = f(high) - target;

    if f_low == 0.0 {
        return low;
    }
    if f_high == 0.0 {
        return high;
    }

    // If they don't bracket, you still *can* run,
    // but bisection's guarantee goes away.
    // Here we just continue, but you could panic! or return NaN.
    if f_low * f_high > 0.0 {
        eprintln!("Warning: binary_solve called with non-bracketing interval.");
    }

    let mut mid = 0.5 * (low + high);
    for _ in 0..max_iter {
        mid = 0.5 * (low + high);
        let f_mid = f(mid);

        // Stop if function value is close enough to target
        if (f_mid - target).abs() <= tol {
            return mid;
        }

        // Decide which half to keep, assuming monotonicity
        if (f_mid - target) * f_low <= 0.0 {
            // Root/target is between low and mid
            high = mid;
            // f_low stays the same
        } else {
            // Root/target is between mid and high
            low = mid;
            // update f_low so we keep correct sign info
            // (you could also recompute f_low if needed)
        }

        // Stop if interval itself is tiny
        if (high - low).abs() <= tol {
            return 0.5 * (low + high);
        }
    }

    // If we hit max_iter, return best current estimate
    mid
}