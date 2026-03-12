use std::fs::read_to_string;
use std::path::Path;
use array_lib::ArrayDim;
use array_lib::io_cfl::write_cfl;
use num_complex::Complex32;

fn main() {

    let seq_name = "dti";
    let acq_dir = Path::new("/Users/Wyatt/scratch/260312_03/acq");

    let vol_data_dims = ArrayDim::from_shape(&[512,256,256]);

    println!("loading raw data ...");
    let (raw_data, mrd_size, ..) = array_lib::io_mrd::read_mrd(acq_dir.join(format!("{seq_name}.MRD")));
    println!("{:?}",mrd_size);

    let s = read_to_string(acq_dir.join("lut.txt")).unwrap();
    let entries:Vec<_> = s.lines().skip(3)
        .map(|s| s.parse::<i32>().expect("failed to parse coordinate")).collect();

    let coords:Vec<Vec<isize>> = entries.chunks_exact(8192).map(|x|{
        x.iter().map(|x| *x as isize).collect()
    }).collect();

    println!("n = {}",coords[0].len());

    let mut pe_table = vec![];
    for i in 0..8192 {
        pe_table.push([coords[0][i],coords[1][i]]);
    }

    // allocate output data
    let mut vol_data = vol_data_dims.alloc(Complex32::ZERO);

    // find DC samples to determine echo times and spacings
    let mut dc_coords = vec![[0isize;3];1];
    let mut max_energy = vec![0f32;1];

    // grid raw data based on pe table
    let n_per_vol = mrd_size.shape()[0]*mrd_size.shape()[1];
    let n_per_line = mrd_size.shape()[0];
    // loop over echo data
    raw_data.chunks_exact(n_per_vol).enumerate().for_each(|(echo_idx,x)|{
        // loop over k-space lines
        x.chunks_exact(n_per_line).zip(pe_table.iter()).for_each(|(ksp_line,&[y,z])|{
            // loop over k-space samples
            ksp_line.iter().enumerate().for_each(|(x,sample)| {
                // record max energy sample and location
                if sample.norm_sqr() > max_energy[echo_idx] {
                    max_energy[echo_idx] = sample.norm_sqr();
                    dc_coords[echo_idx] = [x as isize,y,z];
                }
                // calculate address for k-space sample, and write into array
                let addr = vol_data_dims.calc_addr_signed(&[x as isize,y,z,echo_idx as isize]);
                vol_data[addr] = *sample;
            })
        })
    });

    let vol_dim = ArrayDim::from_shape(&vol_data_dims.shape()[0..3]);
    let mut tmp_dst = vol_dim.alloc(Complex32::ZERO);
    let mut dc_indices = vec![];
    vol_data.chunks_exact_mut(vol_dim.numel()).enumerate().for_each(|(echo_idx,vol)|{
        // reverse-shift
        let shift = [
            -dc_coords[echo_idx][0],
            -dc_coords[echo_idx][1],
            -dc_coords[echo_idx][2]
        ];
        dc_indices.push(dc_coords[echo_idx][0]);
        // circshift vol into tmp, then write tmp back into vol
        tmp_dst.fill(Complex32::ZERO);
        vol_dim.circshift(&shift,vol,&mut tmp_dst);
        vol.copy_from_slice(&tmp_dst);
    });


    write_cfl(acq_dir.join("ksp"),&vol_data,vol_data_dims);

}