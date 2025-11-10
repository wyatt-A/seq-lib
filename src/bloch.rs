
/// returns the z-axis rotation in radians given a spin position (m), gradient strength (T / m), gyromagnetic ratio (rad s^-1 T^-1), delta b0 (T)
pub fn off_resonance(r:&[f64], g:&[f64], db0:f64, gamma:f64, dt:f64) -> f64 {
    let g_dot_r = g[0] * r[0] + g[1] * r[1] + g[2] * r[2];
    dt * gamma * (g_dot_r + db0)
}


/// returns the rotation matrix for the RF rotation. Bx and By are in Tesla, gamma is in rad s^-2 T^-1, dt in s
pub fn rot_matrix(m:&mut [f64], bx:f64, by:f64, gamma:f64, dt: f64 ) -> [f64;3] {

    let b = (bx * bx + by * by).sqrt();
    let theta = gamma * b * dt;
    let ux = bx / b;
    let uy = by / b;

    let c = theta.cos();
    let s = theta.sin();
    let d = 1. - c;

    let r11 = (d * ux * ux + c);
    let r22 = (d * uy * uy + c);
    let r33 = c;
    let r12 = d * ux * uy;
    let r31 = - s * uy;
    let r32 = s * ux;

}