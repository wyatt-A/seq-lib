use std::f32::consts::PI;
use std::path::Path;
use array_lib::ArrayDim;
use array_lib::io_cfl::{read_cfl, write_cfl};
use array_lib::io_mrd::read_mrd;
use dft_lib::common::{FftDirection, NormalizationType};
use dft_lib::rs_fft;
use nalgebra::DVector;
use num_complex::Complex32;
use rayon::prelude::*;

fn main() {
    //find_phase_shifts("/Users/wyatt/measure.MRD");
    find_phase_shifts("/Users/wyatt/phant_test");
}

fn find_phase_shifts(mrd_file:impl AsRef<Path>) {
    //let (data, dims, ..) = read_mrd(mrd_file);
    let (data,dims) = read_cfl(mrd_file);

    println!("read complete");
    let shape = dims.shape();

    let n_samples = shape[0];
    let n_phase = shape[1];
    let n_echoes = shape[4];
    let n_q = shape[5];

    println!("n_read samples: {}", n_samples);
    println!("n_phase steps: {}", n_phase);
    println!("n_echoes: {}", n_echoes);
    println!("n_directions: {}", n_q);

    // let fov_x = 25.6;
    // let fov_y = 12.8;
    // let fov_z = 12.8;
    // let n_x = 512;
    // let n_y = 256;
    // let n_z = 256;

    let fov_x = 12.0;
    let fov_y = 12.0;
    let fov_z = 12.0;
    let n_x = 256;
    let n_y = 256;
    let n_z = 256;

    // shift in k-space has units rad / mm. So a 1 sample shift with FOV of 1mm is 2pi/mm

    let n_dummies = 0; // dummies to trim off the data set
    let r = 3; // sample radius for image support
    let np = 2 * r + 1; // dimension for image support

    let vol_dim = ArrayDim::from_shape(&[n_samples, np, np]);

    // trim dummy scans
    let d_trimmed = ArrayDim::from_shape(&[n_samples, np, np, n_echoes, n_q]);
    let mut y = d_trimmed.alloc(Complex32::ZERO);

    // trim off the dummy scans
    data.par_chunks_exact(n_samples * n_phase * n_echoes).zip(y.par_chunks_exact_mut(n_samples * np * np * n_echoes)).for_each(|(data,y)|{
        data.par_chunks_exact(n_samples * n_phase).zip(y.par_chunks_exact_mut(n_samples * np * np)).for_each(|(data,y)|{
            data.par_chunks_exact(n_samples).skip(n_dummies).zip(y.par_chunks_exact_mut(n_samples)).for_each(|(data,y)|{
                y.copy_from_slice(data);
            })
        })
    });

    let mut corr = d_trimmed.alloc(Complex32::ZERO);

    let mut c = [0,0,0];
    // this is where the center of k-space should be
    vol_dim.fft_shift_coords(&[0,0,0],&mut c);

    let mut shift_dims = ArrayDim::from_shape(&[3,n_echoes * n_q]);
    let mut shifts = vec![];

    y.chunks_exact(n_samples * np * np).zip(corr.chunks_exact_mut(n_samples * np * np)).for_each(|(y,corr)|{
        // find the location of the DC sample (k0) - this is the base phase shift per FOV in units of 2pi radians
        let [dkx, dky, dkz, ..] = vol_dim.argmax_norm_sqr(y).unwrap();

        // phase shift error for each axis converted to radians per mm (or whatever unit FOV is in)
        shifts.push(
            (dkx as f32 - c[0] as f32) * 2. * PI / fov_x
        );

        shifts.push(
            (dky as f32 - c[1] as f32) * 2. * PI / fov_y
        );

        shifts.push(
            (dkz as f32 - c[2] as f32) * 2. * PI / fov_z
        );

        // reverse shift such that the sample is at address 0
        vol_dim.circshift(&[-(dkx as isize),-(dky as isize),-(dkz as isize)],y,corr);
        // normalize by the center phase value
        let (_,phase) = corr[0].to_polar();
        corr.iter_mut().for_each(|x| *x = *x * Complex32::from_polar(1.,-phase));
    });

    shifts.chunks_exact(3).for_each(|s|{
        println!("{:?}",s);
    });

    // small support block
    let support_dim = ArrayDim::from_shape(&[np, np, np, n_echoes * n_q]);
    let mut support = support_dim.alloc(Complex32::ZERO);
    let support_vol_dim = ArrayDim::from_shape(&[np, np, np]);

    // copy only the center k-space samples from corr volumes to support volumes
    let r = r as isize;
    corr.par_chunks_exact(n_samples * np * np).zip(support.par_chunks_exact_mut(np*np*np)).for_each(|(vol,supp)|{
        for x in -r..=r {
            for y in -r..=r {
                for z in -r..=r {
                    let vol_addr = vol_dim.calc_addr_signed(&[x,y,z]);
                    let sup_addr = support_vol_dim.calc_addr_signed(&[x,y,z]);
                    supp[sup_addr] = vol[vol_addr];
                }
            }
        }
    });
    println!("fft");
    // perform inverse fft on support region
    rs_fft::rs_fftn_batched(&mut support,&[np,np,np],n_echoes * n_q, FftDirection::Inverse, NormalizationType::Unitary);

    // crop support to the center 27 samples
    let img_dim = ArrayDim::from_shape(&[3, 3, 3]);
    let img_stack_dims = ArrayDim::from_shape(&[3, 3, 3, n_echoes * n_q]);
    let mut img_stack = img_stack_dims.alloc(Complex32::ZERO);

    support.par_chunks_exact(np * np * np).zip(img_stack.par_chunks_exact_mut(27)).for_each(|(supp,img)|{
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let supp_addr = support_vol_dim.calc_addr_signed(&[x,y,z]);
                    let img_addr = img_dim.calc_addr_signed(&[x,y,z]);
                    img[img_addr] = supp[supp_addr];
                }
            }
        }
    });

    // perform least-squares fit to estimate the linear phase map coefficients

    // build design matrix (coordinate points for the phase values)
    let mat_dims = ArrayDim::from_shape(&[27,4]);
    let mut mat = mat_dims.alloc(0f32);
    let mut i = 0;
    // voxel spacing of low-res image
    let res_x = fov_x/np as f32;
    let res_y = fov_y/np as f32;
    let res_z = fov_z/np as f32;

    // image space coordinates for each phase sample in the array
    let ax_x = [0.,res_x,-res_x];
    let ax_y = [0.,res_y,-res_y];
    let ax_z = [0.,res_z,-res_z];
    for z in ax_z {
        for y in ax_y {
            for x in ax_x {
                mat[mat_dims.calc_addr(&[i,0])] = x;
                mat[mat_dims.calc_addr(&[i,1])] = y;
                mat[mat_dims.calc_addr(&[i,2])] = z;
                mat[mat_dims.calc_addr(&[i,3])] = 1.;
                i += 1;
            }
        }
    }

    // perform SVD on design matrix to solve the system
    let mat_svd = nalgebra::DMatrix::from_column_slice(27,4,&mat).svd(true, true);

    // allocate memory for the solved phase map coefficients
    let coeffs_dims = ArrayDim::from_shape(&[4, n_echoes * n_q]);
    let mut coeffs = coeffs_dims.alloc(0f32);

    // solve for the phase map coefficients for each small FOV
    img_stack.par_chunks_exact(27).zip(coeffs.par_chunks_exact_mut(4)).for_each(|(vol,coeffs)|{
        let y = DVector::from_iterator(27,vol.iter().map(|x|x.to_polar().1));
        let o = mat_svd.solve(&y,f32::EPSILON).unwrap();
        // coefficient units are in radians per mm (or whatever units FOV is in)
        coeffs.copy_from_slice(o.as_slice());
    });

    // add the whole shift to the coefficients to represent the phase map
    coeffs.chunks_exact_mut(4).zip(shifts.chunks_exact(3)).for_each(|(coeffs,shifts)|{
        coeffs[0..3].iter_mut().zip(shifts.iter()).for_each(|(coeff,shift)|{
            *coeff += shift;
        });
    });

    coeffs.chunks_exact(4).for_each(|coeffs|
        println!("{:?}", &coeffs[0..3])
    );

    // save phase coefficient values to a cfl file. The units are in radians per mm for x,y,z and
    // radians for the affine term
    let coeffs:Vec<_> = coeffs.into_par_iter().map(|c| Complex32::new(c,0.)).collect();

    write_cfl("coeffs",&coeffs,coeffs_dims);

    write_cfl("support",&img_stack,img_stack_dims);



}