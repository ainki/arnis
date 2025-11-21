use crate::coordinate_system::{geographic::LLBBox, transformation::geo_distance};
use crate::telemetry::{send_log, LogLevel};
use image::Rgb;
use rayon::prelude::*;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Maximum Y coordinate in Minecraft (build height limit)
const MAX_Y: i32 = 319;
/// Scale factor for converting real elevation to Minecraft heights
const BASE_HEIGHT_SCALE: f64 = 0.7;
/// AWS S3 Terrarium tiles endpoint (no API key required)
const AWS_TERRARIUM_URL: &str =
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
/// Mapbox Terrain RGB (mapbox-terrain / terrain-rgb) template.
/// Replace `{token}` with your Mapbox access token.
const MAPBOX_TERRAIN_URL: &str =
    "https://api.mapbox.com/v4/mapbox.terrain-rgb/{z}/{x}/{y}.pngraw?access_token={token}";
/// Terrain data source selection
#[derive(Clone, Debug)]
pub enum TerrainProvider {
    /// Default AWS Terrarium tiles
    Aws,
    /// Mapbox Terrain RGB (requires access token)
    Mapbox { token: String },
}

fn build_tile_url(provider: &TerrainProvider, tile_x: u32, tile_y: u32, zoom: u8) -> String {
    match provider {
        TerrainProvider::Aws => AWS_TERRARIUM_URL
            .replace("{z}", &zoom.to_string())
            .replace("{x}", &tile_x.to_string())
            .replace("{y}", &tile_y.to_string()),
        TerrainProvider::Mapbox { token } => MAPBOX_TERRAIN_URL
            .replace("{z}", &zoom.to_string())
            .replace("{x}", &tile_x.to_string())
            .replace("{y}", &tile_y.to_string())
            .replace("{token}", token),
    }
}
/// Terrarium format offset for height decoding
const TERRARIUM_OFFSET: f64 = 32768.0;
/// Minimum zoom level for terrain tiles
const MIN_ZOOM: u8 = 10;
/// Maximum zoom level for terrain tiles
const MAX_ZOOM: u8 = 15;

/// Decode elevation from RGB pixel based on provider format
fn decode_height(pixel: &Rgb<u8>, provider: &TerrainProvider) -> f64 {
    match provider {
        TerrainProvider::Aws => {
            // Terrarium format: (R * 256 + G + B/256) - 32768
            (pixel[0] as f64 * 256.0 + pixel[1] as f64 + pixel[2] as f64 / 256.0) - TERRARIUM_OFFSET
        }
        TerrainProvider::Mapbox { .. } => {
            // Mapbox RGB format: -10000 + ((R * 256 * 256 + G * 256 + B) * 0.1)
            -10000.0
                + ((pixel[0] as f64 * 256.0 * 256.0 + pixel[1] as f64 * 256.0 + pixel[2] as f64)
                    * 0.1)
        }
    }
}

/// Holds processed elevation data and metadata
#[derive(Clone)]
pub struct ElevationData {
    /// Height values in Minecraft Y coordinates
    pub(crate) heights: Vec<Vec<i32>>,
    /// Width of the elevation grid
    pub(crate) width: usize,
    /// Height of the elevation grid
    pub(crate) height: usize,
}

/// Calculates appropriate zoom level for the given bounding box
fn calculate_zoom_level(bbox: &LLBBox) -> u8 {
    let lat_diff: f64 = (bbox.max().lat() - bbox.min().lat()).abs();
    let lng_diff: f64 = (bbox.max().lng() - bbox.min().lng()).abs();
    let max_diff: f64 = lat_diff.max(lng_diff);
    let zoom: u8 = (-max_diff.log2() + 20.0) as u8;
    zoom.clamp(MIN_ZOOM, MAX_ZOOM)
}

fn lat_lng_to_tile(lat: f64, lng: f64, zoom: u8) -> (u32, u32) {
    let lat_rad: f64 = lat.to_radians();
    let n: f64 = 2.0_f64.powi(zoom as i32);
    let x: u32 = ((lng + 180.0) / 360.0 * n).floor() as u32;
    let y: u32 = ((1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor() as u32;
    (x, y)
}

/// Downloads a tile from terrain tile service
fn download_tile(
    client: &reqwest::blocking::Client,
    url: &str,
    tile_x: u32,
    tile_y: u32,
    zoom: u8,
    tile_path: &Path,
    provider: &TerrainProvider,
) -> Result<image::ImageBuffer<Rgb<u8>, Vec<u8>>, Box<dyn std::error::Error>> {
    println!("Fetching tile x={tile_x},y={tile_y},z={zoom} from {url}");

    let response: reqwest::blocking::Response = client.get(url).send()?;
    response.error_for_status_ref()?;
    let bytes = response.bytes()?;
    std::fs::write(tile_path, &bytes)?;
    let img: image::DynamicImage = image::load_from_memory(&bytes)?;
    let rgb_img = img.to_rgb8();

    // Validate tile: check if it has reasonable elevation data
    // Sample a few pixels to see if they decode to reasonable values
    let mut sample_count = 0;
    let mut valid_count = 0;
    for y in (0..rgb_img.height()).step_by(64) {
        for x in (0..rgb_img.width()).step_by(64) {
            let pixel = rgb_img.get_pixel(x, y);
            let height = decode_height(pixel, provider);
            sample_count += 1;
            // Reasonable elevation range: -500m to 9000m (covers Dead Sea to Everest)
            if (-500.0..=9000.0).contains(&height) {
                valid_count += 1;
            }
        }
    }

    // If less than 10% of samples are in valid range, tile is likely corrupted
    if sample_count > 0 && (valid_count as f64 / sample_count as f64) < 0.1 {
        eprintln!(
            "Warning: Tile x={tile_x},y={tile_y},z={zoom} appears corrupted ({}/{} samples valid)",
            valid_count, sample_count
        );
        send_log(
            LogLevel::Warning,
            &format!(
                "Tile {}/{}/{} appears corrupted or uses different encoding",
                zoom, tile_x, tile_y
            ),
        );
    }

    Ok(rgb_img)
}

pub fn fetch_elevation_data(
    bbox: &LLBBox,
    scale: f64,
    ground_level: i32,
) -> Result<ElevationData, Box<dyn std::error::Error>> {
    // Preserve existing behavior: default to AWS Terrarium tiles
    fetch_elevation_data_with_provider(bbox, scale, ground_level, TerrainProvider::Aws)
}

/// Fetch elevation data using a selected `TerrainProvider`.
pub fn fetch_elevation_data_with_provider(
    bbox: &LLBBox,
    scale: f64,
    ground_level: i32,
    provider: TerrainProvider,
) -> Result<ElevationData, Box<dyn std::error::Error>> {
    // Move the main implementation into an inner function that accepts a provider reference
    fetch_elevation_data_inner(bbox, scale, ground_level, &provider)
}

fn fetch_elevation_data_inner(
    bbox: &LLBBox,
    scale: f64,
    ground_level: i32,
    provider: &TerrainProvider,
) -> Result<ElevationData, Box<dyn std::error::Error>> {
    let (base_scale_z, base_scale_x) = geo_distance(bbox.min(), bbox.max());

    // Apply same floor() and scale operations as CoordTransformer.llbbox_to_xzbbox()
    let scale_factor_z: f64 = base_scale_z.floor() * scale;
    let scale_factor_x: f64 = base_scale_x.floor() * scale;

    // Calculate zoom and tiles
    let zoom: u8 = calculate_zoom_level(bbox);
    let tiles: Vec<(u32, u32)> = get_tile_coordinates(bbox, zoom);

    // Match grid dimensions with Minecraft world size
    let grid_width: usize = scale_factor_x as usize;
    let grid_height: usize = scale_factor_z as usize;

    // Initialize height grid with proper dimensions
    let height_grid: Vec<Vec<f64>> = vec![vec![f64::NAN; grid_width]; grid_height];

    let tile_cache_dir = Path::new("./arnis-tile-cache");
    if !tile_cache_dir.exists() {
        std::fs::create_dir_all(tile_cache_dir)?;
    }

    // Process tiles in parallel
    let height_grid = Arc::new(Mutex::new(height_grid));
    let extreme_values_found: Arc<Mutex<Vec<(&u32, &u32, usize, usize, u8, u8, u8, f64)>>> =
        Arc::new(Mutex::new(Vec::new()));
    let errors = Arc::new(Mutex::new(Vec::new()));

    tiles.par_iter().for_each(|(tile_x, tile_y)| {
        // Create a client per thread to avoid sharing issues
        let client = reqwest::blocking::Client::new();

        let result: Result<(), Box<dyn std::error::Error>> = (|| {
            // Check if tile is already cached
            let tile_path = tile_cache_dir.join(format!("z{zoom}_x{tile_x}_y{tile_y}.png"));

            // Build URL for this tile according to the selected provider
            let tile_url = build_tile_url(provider, *tile_x, *tile_y, zoom);

            let rgb_img: image::ImageBuffer<Rgb<u8>, Vec<u8>> = if tile_path.exists() {
                // Try to load cached tile
                match image::open(&tile_path) {
                    Ok(img) => {
                        println!(
                            "Loading cached tile x={tile_x},y={tile_y},z={zoom} from {}",
                            tile_path.display()
                        );
                        img.to_rgb8()
                    }
                    Err(e) => {
                        eprintln!(
                            "Cached tile at {} is corrupted or invalid: {}. Re-downloading...",
                            tile_path.display(),
                            e
                        );
                        send_log(
                            LogLevel::Warning,
                            "Cached tile is corrupted or invalid. Re-downloading...",
                        );

                        // Remove the corrupted file
                        let _ = std::fs::remove_file(&tile_path);

                        // Re-download the tile
                        download_tile(
                            &client, &tile_url, *tile_x, *tile_y, zoom, &tile_path, provider,
                        )?
                    }
                }
            } else {
                // Download the tile for the first time
                download_tile(
                    &client, &tile_url, *tile_x, *tile_y, zoom, &tile_path, provider,
                )?
            };

            // Process pixels for this tile
            // Pre-calculate tile constants outside the loop
            let tile_x_f64 = *tile_x as f64;
            let tile_y_f64 = *tile_y as f64;
            let zoom_pow = 2.0_f64.powi(zoom as i32);
            let bbox_lng_min = bbox.min().lng();
            let bbox_lng_max = bbox.max().lng();
            let bbox_lat_min = bbox.min().lat();
            let bbox_lat_max = bbox.max().lat();
            let bbox_lng_range = bbox_lng_max - bbox_lng_min;
            let bbox_lat_range = bbox_lat_max - bbox_lat_min;

            let mut local_extreme_values = Vec::new();
            let mut local_updates = Vec::with_capacity(256 * 256);

            // Process pixels in batches to minimize lock contention
            const BATCH_SIZE: usize = 4096; // Process ~64x64 pixels before acquiring lock

            // Only process pixels that fall within the requested bbox
            for (y, row) in rgb_img.rows().enumerate() {
                for (x, pixel) in row.enumerate() {
                    // Convert tile pixel coordinates back to geographic coordinates
                    let pixel_lng = ((tile_x_f64 + x as f64 / 256.0) / zoom_pow) * 360.0 - 180.0;
                    let pixel_lat_rad = std::f64::consts::PI
                        * (1.0 - 2.0 * (tile_y_f64 + y as f64 / 256.0) / zoom_pow);
                    let pixel_lat = pixel_lat_rad.sinh().atan().to_degrees();

                    // Skip pixels outside the requested bounding box
                    if pixel_lat < bbox_lat_min
                        || pixel_lat > bbox_lat_max
                        || pixel_lng < bbox_lng_min
                        || pixel_lng > bbox_lng_max
                    {
                        continue;
                    }

                    // Map geographic coordinates to grid coordinates
                    let rel_x = (pixel_lng - bbox_lng_min) / bbox_lng_range;
                    let rel_y = 1.0 - (pixel_lat - bbox_lat_min) / bbox_lat_range;

                    let scaled_x = (rel_x * grid_width as f64).round() as usize;
                    let scaled_y = (rel_y * grid_height as f64).round() as usize;

                    if scaled_y >= grid_height || scaled_x >= grid_width {
                        continue;
                    }

                    // Decode height using provider-specific format
                    let mut height: f64 = decode_height(pixel, provider);

                    // Clamp to Earth's realistic elevation range to handle corrupted data
                    const MIN_EARTH_ELEVATION: f64 = -500.0;
                    const MAX_EARTH_ELEVATION: f64 = 9000.0;

                    let original_height = height;
                    height = height.clamp(MIN_EARTH_ELEVATION, MAX_EARTH_ELEVATION);

                    // Track extreme values for debugging (before clamping)
                    if !(-1000.0..=10000.0).contains(&original_height) {
                        local_extreme_values.push((
                            tile_x,
                            tile_y,
                            x,
                            y,
                            pixel[0],
                            pixel[1],
                            pixel[2],
                            original_height,
                        ));
                    }

                    local_updates.push((scaled_y, scaled_x, height));

                    // Apply batch if it's full to reduce lock hold time
                    if local_updates.len() >= BATCH_SIZE {
                        let mut grid = height_grid.lock().unwrap();
                        for &(y, x, h) in &local_updates {
                            grid[y][x] = h;
                        }
                        drop(grid);
                        local_updates.clear();
                    }
                }
            }

            // Apply remaining updates
            if !local_updates.is_empty() {
                let mut grid = height_grid.lock().unwrap();
                for (y, x, height) in local_updates {
                    grid[y][x] = height;
                }
                drop(grid);
            }

            // Add extreme values to shared collection
            if !local_extreme_values.is_empty() {
                let mut extremes = extreme_values_found.lock().unwrap();
                extremes.extend(local_extreme_values);
            }

            Ok(())
        })();

        // Store errors for later reporting
        if let Err(e) = result {
            eprintln!("Error processing tile {}/{}: {}", tile_x, tile_y, e);
            errors
                .lock()
                .unwrap()
                .push(format!("Tile {}/{}: {}", tile_x, tile_y, e));
        }
    });

    // Check if there were any errors
    let errors = Arc::try_unwrap(errors).unwrap().into_inner().unwrap();
    if !errors.is_empty() {
        eprintln!(
            "Encountered {} errors during tile processing:",
            errors.len()
        );
        for (i, error) in errors.iter().take(5).enumerate() {
            eprintln!("  {}: {}", i + 1, error);
        }
        if errors.len() > 5 {
            eprintln!("  ... and {} more", errors.len() - 5);
        }
        return Err(format!("Failed to process {} tiles", errors.len()).into());
    }

    let mut height_grid = Arc::try_unwrap(height_grid).unwrap().into_inner().unwrap();
    let extreme_values_found = Arc::try_unwrap(extreme_values_found)
        .unwrap()
        .into_inner()
        .unwrap();

    // Log first few extreme values
    for (tile_x, tile_y, x, y, r, g, b, height) in extreme_values_found.iter().take(5) {
        eprintln!("Extreme value found: tile({tile_x},{tile_y}) pixel({x},{y}) RGB({r},{g},{b}) = {height}m");
    }

    // Report on extreme values found
    if !extreme_values_found.is_empty() {
        eprintln!(
            "Found {} total extreme elevation values during tile processing",
            extreme_values_found.len()
        );
        eprintln!("This may indicate corrupted tile data or areas with invalid elevation data");
        eprintln!("The system has clamped these values to realistic Earth elevation ranges (-500m to 9000m)");
        eprintln!("If you're seeing flat terrain where you expect hills/mountains, the elevation tiles may be corrupted.");
        eprintln!("Try deleting the 'arnis-tile-cache' folder to re-download tiles.");
    }

    // Fill in any NaN values by interpolating from nearest valid values
    fill_nan_values(&mut height_grid);

    // Filter extreme outliers that might be due to corrupted tile data
    filter_elevation_outliers(&mut height_grid);

    // Calculate blur sigma based on grid resolution
    // Reference points for tuning:
    const SMALL_GRID_REF: f64 = 100.0; // Reference grid size
    const SMALL_SIGMA_REF: f64 = 15.0; // Sigma for 100x100 grid
    const LARGE_GRID_REF: f64 = 1000.0; // Reference grid size
    const LARGE_SIGMA_REF: f64 = 7.0; // Sigma for 1000x1000 grid

    let grid_size: f64 = (grid_width.min(grid_height) as f64).max(1.0);

    let sigma: f64 = if grid_size <= SMALL_GRID_REF {
        // Linear scaling for small grids
        SMALL_SIGMA_REF * (grid_size / SMALL_GRID_REF)
    } else {
        // Logarithmic scaling for larger grids
        let ln_small: f64 = SMALL_GRID_REF.ln();
        let ln_large: f64 = LARGE_GRID_REF.ln();
        let log_grid_size: f64 = grid_size.ln();
        let t: f64 = (log_grid_size - ln_small) / (ln_large - ln_small);
        SMALL_SIGMA_REF + t * (LARGE_SIGMA_REF - SMALL_SIGMA_REF)
    };

    /* eprintln!(
        "Grid: {}x{}, Blur sigma: {:.2}",
        grid_width, grid_height, sigma
    ); */

    // Continue with the existing blur and conversion to Minecraft heights...
    let blurred_heights: Vec<Vec<f64>> = apply_gaussian_blur(&height_grid, sigma);

    // Find min/max in raw data
    let mut min_height: f64 = f64::MAX;
    let mut max_height: f64 = f64::MIN;
    let mut extreme_low_count = 0;
    let mut extreme_high_count = 0;

    for row in &blurred_heights {
        for &height in row {
            min_height = min_height.min(height);
            max_height = max_height.max(height);

            // Count extreme values that might indicate data issues
            if height < -1000.0 {
                extreme_low_count += 1;
            }
            if height > 10000.0 {
                extreme_high_count += 1;
            }
        }
    }

    eprintln!("Height data range: {min_height} to {max_height} m");
    if extreme_low_count > 0 {
        eprintln!(
            "WARNING: Found {extreme_low_count} pixels with extremely low elevations (< -1000m)"
        );
    }
    if extreme_high_count > 0 {
        eprintln!(
            "WARNING: Found {extreme_high_count} pixels with extremely high elevations (> 10000m)"
        );
    }

    let height_range: f64 = max_height - min_height;
    // Apply scale factor to height scaling
    // Use a provider-specific multiplier because some providers (e.g., Mapbox terrain-rgb)
    // may have less pronounced elevation changes at the same sampling/resolution.
    let provider_height_mult: f64 = match provider {
        TerrainProvider::Mapbox { .. } => 1.0, // Mapbox heights are already accurate, no amplification needed
        _ => 1.0,
    };

    let mut height_scale: f64 = BASE_HEIGHT_SCALE * scale.sqrt() * provider_height_mult; // sqrt to make height scaling less extreme
    let mut scaled_range: f64 = height_range * height_scale;

    // Adaptive scaling: ensure we don't exceed reasonable Y range
    let available_y_range = (MAX_Y - ground_level) as f64;
    let safety_margin = 0.9; // Use 90% of available range
    let max_allowed_range = available_y_range * safety_margin;

    if scaled_range > max_allowed_range {
        let adjustment_factor = max_allowed_range / scaled_range;
        height_scale *= adjustment_factor;
        scaled_range = height_range * height_scale;
        eprintln!(
            "Height range too large, applying scaling adjustment factor: {adjustment_factor:.3}"
        );
        eprintln!("Adjusted scaled range: {scaled_range:.1} blocks");
    }

    // Convert to scaled Minecraft Y coordinates (parallel)
    let mc_heights: Vec<Vec<i32>> = blurred_heights
        .par_iter()
        .map(|row| {
            row.iter()
                .map(|&h| {
                    // Scale the height differences
                    let relative_height: f64 = (h - min_height) / height_range;
                    let scaled_height: f64 = relative_height * scaled_range;
                    // With terrain enabled, ground_level is used as the MIN_Y for terrain
                    ((ground_level as f64 + scaled_height).round() as i32)
                        .clamp(ground_level, MAX_Y)
                })
                .collect()
        })
        .collect();

    let mut min_block_height: i32 = i32::MAX;
    let mut max_block_height: i32 = i32::MIN;
    for row in &mc_heights {
        for &height in row {
            min_block_height = min_block_height.min(height);
            max_block_height = max_block_height.max(height);
        }
    }
    eprintln!("Minecraft height data range: {min_block_height} to {max_block_height} blocks");

    Ok(ElevationData {
        heights: mc_heights,
        width: grid_width,
        height: grid_height,
    })
}

fn get_tile_coordinates(bbox: &LLBBox, zoom: u8) -> Vec<(u32, u32)> {
    // Convert lat/lng to tile coordinates
    let (x1, y1) = lat_lng_to_tile(bbox.min().lat(), bbox.min().lng(), zoom);
    let (x2, y2) = lat_lng_to_tile(bbox.max().lat(), bbox.max().lng(), zoom);

    let mut tiles: Vec<(u32, u32)> = Vec::new();
    for x in x1.min(x2)..=x1.max(x2) {
        for y in y1.min(y2)..=y1.max(y2) {
            tiles.push((x, y));
        }
    }
    tiles
}

fn apply_gaussian_blur(heights: &[Vec<f64>], sigma: f64) -> Vec<Vec<f64>> {
    let kernel_size: usize = (sigma * 3.0).ceil() as usize * 2 + 1;
    let kernel: Vec<f64> = create_gaussian_kernel(kernel_size, sigma);

    // Apply blur
    let mut blurred: Vec<Vec<f64>> = heights.to_owned();

    // Horizontal pass (parallel)
    blurred.par_iter_mut().for_each(|row| {
        let original_row = row.clone();
        for (i, val) in row.iter_mut().enumerate() {
            let mut sum: f64 = 0.0;
            let mut weight_sum: f64 = 0.0;
            for (j, k) in kernel.iter().enumerate() {
                let idx: i32 = i as i32 + j as i32 - kernel_size as i32 / 2;
                if idx >= 0 && idx < original_row.len() as i32 {
                    sum += original_row[idx as usize] * k;
                    weight_sum += k;
                }
            }
            *val = sum / weight_sum;
        }
    });

    // Vertical pass (parallel)
    let height: usize = blurred.len();
    let width: usize = blurred[0].len();

    let vertical_blurred: Vec<Vec<f64>> = (0..width)
        .into_par_iter()
        .map(|x| {
            let temp: Vec<_> = blurred
                .iter()
                .take(height)
                .map(|row: &Vec<f64>| row[x])
                .collect();

            (0..height)
                .map(|y| {
                    let mut sum: f64 = 0.0;
                    let mut weight_sum: f64 = 0.0;
                    for (j, k) in kernel.iter().enumerate() {
                        let idx: i32 = y as i32 + j as i32 - kernel_size as i32 / 2;
                        if idx >= 0 && idx < height as i32 {
                            sum += temp[idx as usize] * k;
                            weight_sum += k;
                        }
                    }
                    sum / weight_sum
                })
                .collect()
        })
        .collect();

    // Transpose back to row-major order
    let mut result = vec![vec![0.0; width]; height];
    for y in 0..height {
        for x in 0..width {
            result[y][x] = vertical_blurred[x][y];
        }
    }

    result
}

fn create_gaussian_kernel(size: usize, sigma: f64) -> Vec<f64> {
    let mut kernel: Vec<f64> = vec![0.0; size];
    let center: f64 = size as f64 / 2.0;

    for (i, value) in kernel.iter_mut().enumerate() {
        let x: f64 = i as f64 - center;
        *value = (-x * x / (2.0 * sigma * sigma)).exp();
    }

    let sum: f64 = kernel.iter().sum();
    for k in kernel.iter_mut() {
        *k /= sum;
    }

    kernel
}

fn fill_nan_values(height_grid: &mut [Vec<f64>]) {
    let height: usize = height_grid.len();
    let width: usize = height_grid[0].len();

    let mut changes_made: bool = true;
    let mut iteration = 0;

    while changes_made {
        changes_made = false;
        iteration += 1;

        // Collect all NaN positions and calculate their interpolated values in parallel
        let updates: Vec<(usize, usize, f64)> = (0..height)
            .into_par_iter()
            .flat_map(|y| {
                let mut row_updates = Vec::new();
                for x in 0..width {
                    if height_grid[y][x].is_nan() {
                        let mut sum: f64 = 0.0;
                        let mut count: i32 = 0;

                        // Check neighboring cells
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                let ny: i32 = y as i32 + dy;
                                let nx: i32 = x as i32 + dx;

                                if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                                    let val: f64 = height_grid[ny as usize][nx as usize];
                                    if !val.is_nan() {
                                        sum += val;
                                        count += 1;
                                    }
                                }
                            }
                        }

                        if count > 0 {
                            row_updates.push((y, x, sum / count as f64));
                        }
                    }
                }
                row_updates
            })
            .collect();

        // Apply all updates
        if !updates.is_empty() {
            changes_made = true;
            for (y, x, value) in updates {
                height_grid[y][x] = value;
            }
        }

        // Limit iterations to prevent infinite loops on large NaN regions
        if iteration > 1000 {
            eprintln!("Warning: NaN interpolation stopped after 1000 iterations");
            break;
        }
    }
}

fn filter_elevation_outliers(height_grid: &mut [Vec<f64>]) {
    let height = height_grid.len();
    let width = height_grid[0].len();

    // Collect all valid height values to calculate statistics
    let mut all_heights: Vec<f64> = Vec::new();
    for row in height_grid.iter() {
        for &h in row {
            if !h.is_nan() && h.is_finite() {
                all_heights.push(h);
            }
        }
    }

    if all_heights.is_empty() {
        return;
    }

    // Sort to find percentiles
    all_heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = all_heights.len();

    // Use 1st and 99th percentiles to define reasonable bounds
    let p1_idx = (len as f64 * 0.01) as usize;
    let p99_idx = (len as f64 * 0.99) as usize;
    let min_reasonable = all_heights[p1_idx];
    let max_reasonable = all_heights[p99_idx];

    eprintln!("Filtering outliers outside range: {min_reasonable:.1}m to {max_reasonable:.1}m");

    // Parallelize outlier filtering
    let outliers_filtered: usize = height_grid
        .par_iter_mut()
        .take(height)
        .map(|row| {
            let mut row_outliers = 0;
            for h in row.iter_mut().take(width) {
                if !h.is_nan() && (*h < min_reasonable || *h > max_reasonable) {
                    *h = f64::NAN;
                    row_outliers += 1;
                }
            }
            row_outliers
        })
        .sum();

    if outliers_filtered > 0 {
        eprintln!("Filtered {outliers_filtered} elevation outliers, interpolating replacements...");
        // Re-run the NaN filling to interpolate the filtered values
        fill_nan_values(height_grid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terrarium_height_decoding() {
        // Test known Terrarium RGB values
        // Sea level (0m) in Terrarium format should be (128, 0, 0) = 32768 - 32768 = 0
        let sea_level_pixel = [128, 0, 0];
        let height = (sea_level_pixel[0] as f64 * 256.0
            + sea_level_pixel[1] as f64
            + sea_level_pixel[2] as f64 / 256.0)
            - TERRARIUM_OFFSET;
        assert_eq!(height, 0.0);

        // Test simple case: height of 1000m
        // 1000 + 32768 = 33768 = 131 * 256 + 232
        let test_pixel = [131, 232, 0];
        let height =
            (test_pixel[0] as f64 * 256.0 + test_pixel[1] as f64 + test_pixel[2] as f64 / 256.0)
                - TERRARIUM_OFFSET;
        assert_eq!(height, 1000.0);

        // Test below sea level (-100m)
        // -100 + 32768 = 32668 = 127 * 256 + 156
        let below_sea_pixel = [127, 156, 0];
        let height = (below_sea_pixel[0] as f64 * 256.0
            + below_sea_pixel[1] as f64
            + below_sea_pixel[2] as f64 / 256.0)
            - TERRARIUM_OFFSET;
        assert_eq!(height, -100.0);
    }

    #[test]
    fn test_aws_url_generation() {
        let url = AWS_TERRARIUM_URL
            .replace("{z}", "15")
            .replace("{x}", "17436")
            .replace("{y}", "11365");
        assert_eq!(
            url,
            "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/15/17436/11365.png"
        );
    }

    #[test]
    fn test_mapbox_url_generation() {
        let token = "pk.testtoken";
        let provider = super::TerrainProvider::Mapbox {
            token: token.to_string(),
        };

        let url = super::build_tile_url(&provider, 17436, 11365, 15);
        assert!(url.contains("mapbox.terrain-rgb/15/17436/11365"));
        assert!(url.contains(token));
    }

    #[test]
    #[ignore] // This test requires internet connection, run with --ignored
    fn test_aws_tile_fetch() {
        use reqwest::blocking::Client;

        let client = Client::new();
        let url = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/15/17436/11365.png";

        let response = client.get(url).send();
        assert!(response.is_ok());

        let response = response.unwrap();
        assert!(response.status().is_success());
        assert!(response
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .contains("image"));
    }
}
