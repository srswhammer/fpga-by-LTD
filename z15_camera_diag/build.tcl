set origin_dir [file normalize [file dirname [info script]]]
create_project camera_diag "$origin_dir/vivado" -part xc7z015clg485-2 -force
add_files [list "$origin_dir/camera_diag_top.v" "$origin_dir/i2c_dri.v" "$origin_dir/i2c_ov5640_rgb565_cfg.v"]
add_files -fileset constrs_1 "$origin_dir/camera_diag.xdc"
set_property top camera_diag_top [current_fileset]
update_compile_order -fileset sources_1
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
    error "Implementation did not complete"
}
file copy -force "$origin_dir/vivado/camera_diag.runs/impl_1/camera_diag_top.bit" "$origin_dir/camera_diag_top.bit"
puts "BITSTREAM=$origin_dir/camera_diag_top.bit"
