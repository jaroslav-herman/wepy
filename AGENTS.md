# Project preferences

- For electrochemical measurement data, prefer `.mpr` files by default. Use other file formats only when explicitly requested or when no suitable `.mpr` files are available.
- For requests such as “Show me polarization curves for sample XY”, use the established workflow: discover `.mpr` files with `we.load_files`, read them with `we.read_file_safe`, extract curves with `wepy.iv_curve.IV_curves_data`, and color the time evolution with `we.get_colors`.
- Keep the default colorscale in `we.get_colors`; do not override its colormap unless explicitly requested. Use `we.get_sample_name(sample_number, sample_folder / "sample_log.csv")` to label plots from `Sample Name` in each sample's folder.
- For graph-producing requests, run the generated Python script after creating it so the requested graph files are produced and validated.
- For requests such as “compare performance time evolution for some samples”, follow the established workflow: use `we.load_folders`, process preferred `.mpr` files with `we.read_file_safe`, extract IV curves with `wepy.iv_curve.IV_curves_data`, read per-sample names from each folder's `sample_log.csv`, preserve the default `we.get_colors()` scale, and create separate graphs for each requested cell voltage.
