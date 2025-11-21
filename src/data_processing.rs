use crate::args::Args;
use crate::block_definitions::{BEDROCK, DIRT, GRASS_BLOCK, STONE};
use crate::coordinate_system::cartesian::XZBBox;
use crate::coordinate_system::geographic::LLBBox;
use crate::element_processing::*;
use crate::ground::Ground;
use crate::osm_parser::ProcessedElement;
use crate::progress::emit_gui_progress_update;
use crate::telemetry::{send_log, LogLevel};
use crate::world_editor::WorldEditor;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub const MIN_Y: i32 = -64;

pub fn generate_world(
    elements: Vec<ProcessedElement>,
    xzbbox: XZBBox,
    llbbox: LLBBox,
    ground: Ground,
    args: &Args,
) -> Result<(), String> {
    let mut editor: WorldEditor = WorldEditor::new(args.path.clone(), &xzbbox, llbbox);

    println!("{} Processing data...", "[4/7]".bold());

    // Set ground reference in the editor to enable elevation-aware block placement
    editor.set_ground(&ground);

    println!("{} Processing terrain...", "[5/7]".bold());
    emit_gui_progress_update(25.0, "Processing terrain...");

    // Process data
    let elements_count: usize = elements.len();
    let process_pb: ProgressBar = ProgressBar::new(elements_count as u64);
    process_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:45.white/black}] {pos}/{len} elements ({eta}) {msg}")
        .unwrap()
        .progress_chars("█▓░"));

    let progress_increment_prcs: f64 = 45.0 / elements_count as f64;
    let current_progress = AtomicU64::new(250); // 25.0 * 10

    // Wrap editor in Arc<Mutex> for thread-safe access
    let editor = Arc::new(Mutex::new(editor));
    let process_pb = Arc::new(Mutex::new(process_pb));

    // Process elements in parallel - each thread processes one element at a time
    // Use Arc to share references across threads
    elements.par_iter().for_each(|element| {
        // Scope to ensure lock is released quickly
        {
            let mut editor = editor.lock().unwrap();

            if args.debug {
                let pb = process_pb.lock().unwrap();
                pb.set_message(format!(
                    "(Element ID: {} / Type: {})",
                    element.id(),
                    element.kind()
                ));
            }

            match element {
                ProcessedElement::Way(way) => {
                    if way.tags.contains_key("building") || way.tags.contains_key("building:part") {
                        buildings::generate_buildings(&mut editor, way, args, None);
                    } else if way.tags.contains_key("highway") {
                        highways::generate_highways(&mut editor, element, args, &elements);
                    } else if way.tags.contains_key("landuse") {
                        landuse::generate_landuse(&mut editor, way, args);
                    } else if way.tags.contains_key("natural") {
                        natural::generate_natural(&mut editor, element, args);
                    } else if way.tags.contains_key("amenity") {
                        amenities::generate_amenities(&mut editor, element, args);
                    } else if way.tags.contains_key("leisure") {
                        leisure::generate_leisure(&mut editor, way, args);
                    } else if way.tags.contains_key("barrier") {
                        barriers::generate_barriers(&mut editor, element);
                    } else if let Some(val) = way.tags.get("waterway") {
                        if val == "dock" {
                            // docks count as water areas
                            water_areas::generate_water_area_from_way(&mut editor, way);
                        } else {
                            waterways::generate_waterways(&mut editor, way);
                        }
                    } else if way.tags.contains_key("bridge") {
                        //bridges::generate_bridges(&mut editor, way, ground_level); // TODO FIX
                    } else if way.tags.contains_key("railway") {
                        railways::generate_railways(&mut editor, way);
                    } else if way.tags.contains_key("roller_coaster") {
                        railways::generate_roller_coaster(&mut editor, way);
                    } else if way.tags.contains_key("aeroway")
                        || way.tags.contains_key("area:aeroway")
                    {
                        highways::generate_aeroway(&mut editor, way, args);
                    } else if way.tags.get("service") == Some(&"siding".to_string()) {
                        highways::generate_siding(&mut editor, way);
                    } else if way.tags.contains_key("man_made") {
                        man_made::generate_man_made(&mut editor, element, args);
                    }
                }
                ProcessedElement::Node(node) => {
                    if node.tags.contains_key("door") || node.tags.contains_key("entrance") {
                        doors::generate_doors(&mut editor, node);
                    } else if node.tags.contains_key("natural")
                        && node.tags.get("natural") == Some(&"tree".to_string())
                    {
                        natural::generate_natural(&mut editor, element, args);
                    } else if node.tags.contains_key("amenity") {
                        amenities::generate_amenities(&mut editor, element, args);
                    } else if node.tags.contains_key("barrier") {
                        barriers::generate_barrier_nodes(&mut editor, node);
                    } else if node.tags.contains_key("highway") {
                        highways::generate_highways(&mut editor, element, args, &elements);
                    } else if node.tags.contains_key("tourism") {
                        tourisms::generate_tourisms(&mut editor, node);
                    } else if node.tags.contains_key("man_made") {
                        man_made::generate_man_made_nodes(&mut editor, node);
                    }
                }
                ProcessedElement::Relation(rel) => {
                    if rel.tags.contains_key("building") || rel.tags.contains_key("building:part") {
                        buildings::generate_building_from_relation(&mut editor, rel, args);
                    } else if rel.tags.contains_key("water")
                        || rel
                            .tags
                            .get("natural")
                            .map(|val| val == "water" || val == "bay")
                            .unwrap_or(false)
                    {
                        water_areas::generate_water_areas_from_relation(&mut editor, rel);
                    } else if rel.tags.contains_key("natural") {
                        natural::generate_natural_from_relation(&mut editor, rel, args);
                    } else if rel.tags.contains_key("landuse") {
                        landuse::generate_landuse_from_relation(&mut editor, rel, args);
                    } else if rel.tags.get("leisure") == Some(&"park".to_string()) {
                        leisure::generate_leisure_from_relation(&mut editor, rel, args);
                    } else if rel.tags.contains_key("man_made") {
                        man_made::generate_man_made(
                            &mut editor,
                            &ProcessedElement::Relation(rel.clone()),
                            args,
                        );
                    }
                }
            } // Close match
        } // Close lock scope

        // Update progress after releasing lock
        let new_progress =
            current_progress.fetch_add((progress_increment_prcs * 10.0) as u64, Ordering::Relaxed);
        if new_progress % 3 == 0 {
            // Update every ~0.3%
            emit_gui_progress_update(new_progress as f64 / 10.0, "");
        }

        let pb = process_pb.lock().unwrap();
        pb.inc(1);
        drop(pb);
    });

    let process_pb = Arc::try_unwrap(process_pb)
        .unwrap_or_else(|arc| {
            eprintln!("Warning: Progress bar still has references");
            (*arc).lock().unwrap().clone().into()
        })
        .into_inner()
        .unwrap();
    process_pb.finish();

    // Extract editor from Arc<Mutex>
    let editor = Arc::try_unwrap(editor)
        .unwrap_or_else(|_| panic!("Failed to unwrap editor - still has references"))
        .into_inner()
        .unwrap();

    // Generate ground layer
    let total_blocks: u64 = xzbbox.bounding_rect().total_blocks();

    println!("{} Generating ground...", "[6/7]".bold());
    emit_gui_progress_update(70.0, "Generating ground...");

    let ground_pb: ProgressBar = ProgressBar::new(total_blocks);
    ground_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:45}] {pos}/{len} blocks ({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let total_iterations_grnd: f64 = total_blocks as f64;
    let progress_increment_grnd: f64 = 20.0 / total_iterations_grnd;

    let groundlayer_block = GRASS_BLOCK;

    // Wrap editor in Mutex for thread-safe access
    let editor = Mutex::new(editor);
    let block_counter = AtomicU64::new(0);
    let gui_progress = AtomicU64::new(700); // Start at 70.0 * 10

    // Generate x,z coordinate pairs
    let coords: Vec<(i32, i32)> = (xzbbox.min_x()..=xzbbox.max_x())
        .flat_map(|x| (xzbbox.min_z()..=xzbbox.max_z()).map(move |z| (x, z)))
        .collect();

    // Capture fillground flag before parallel execution
    let fillground = args.fillground;

    // Process blocks in parallel chunks to reduce lock contention
    // Larger chunks = fewer lock acquisitions = better parallelism
    let chunk_size = 256.min(coords.len() / (rayon::current_num_threads() * 2).max(1));

    // Process blocks in parallel chunks
    coords.par_chunks(chunk_size).for_each(|chunk| {
        // Pre-calculate absolute_y values for the entire chunk
        let absolute_ys: Vec<i32> = if fillground {
            let editor = editor.lock().unwrap();
            chunk
                .iter()
                .map(|(x, z)| editor.get_absolute_y(*x, -3, *z))
                .collect()
        } else {
            vec![]
        };

        // Now process the entire chunk with a single lock acquisition
        let mut editor = editor.lock().unwrap();

        for (i, (x, z)) in chunk.iter().enumerate() {
            // Only place ground blocks if nothing exists at ground level (y=0)
            // This prevents overwriting buildings and other structures
            if !editor.block_at(*x, 0, *z) {
                // Add default dirt and grass layer
                editor.set_block(groundlayer_block, *x, 0, *z, None, None);
                editor.set_block(DIRT, *x, -1, *z, None, None);
                editor.set_block(DIRT, *x, -2, *z, None, None);
            }

            // Fill underground with stone (only if not already filled)
            if fillground {
                let absolute_y_for_stone = absolute_ys[i];
                // Fill from bedrock+1 to 3 blocks below ground with stone
                // The fill_blocks function should already check for existing blocks
                editor.fill_blocks_absolute(
                    STONE,
                    *x,
                    MIN_Y + 1,
                    *z,
                    *x,
                    absolute_y_for_stone,
                    *z,
                    None,
                    None,
                );
            }
            // Generate a bedrock level at MIN_Y
            editor.set_block_absolute(BEDROCK, *x, MIN_Y, *z, None, Some(&[BEDROCK]));
        }

        let chunk_len = chunk.len() as u64;
        let count = block_counter.fetch_add(chunk_len, Ordering::Relaxed);

        drop(editor); // Release lock before progress update

        // Update progress bar with actual chunk size
        ground_pb.inc(chunk_len);

        // Update GUI progress
        let new_progress = (70.0 + (count as f64 * progress_increment_grnd)) * 10.0;
        let prev = gui_progress.fetch_max(new_progress as u64, Ordering::Relaxed);
        if new_progress as u64 - prev > 2 {
            // Update every 0.2%
            emit_gui_progress_update(new_progress / 10.0, "");
        }
    });

    // Final progress updates
    let final_count = block_counter.load(Ordering::Relaxed);
    // Ensure progress bar reaches exactly the total
    if final_count < total_blocks {
        ground_pb.inc(total_blocks - final_count);
    }
    ground_pb.finish();

    // Extract editor from Mutex
    let mut editor = editor.into_inner().unwrap();

    // Save world
    editor.save();

    // Update player spawn Y coordinate based on terrain height after generation
    #[cfg(feature = "gui")]
    if let Some(spawn_coords) = &args.spawn_point {
        use crate::gui::update_player_spawn_y_after_generation;
        let bbox_string = format!(
            "{},{},{},{}",
            args.bbox.min().lng(),
            args.bbox.min().lat(),
            args.bbox.max().lng(),
            args.bbox.max().lat()
        );

        if let Err(e) = update_player_spawn_y_after_generation(
            &args.path,
            Some(*spawn_coords),
            bbox_string,
            args.scale,
            &ground,
        ) {
            let warning_msg = format!("Failed to update spawn point Y coordinate: {}", e);
            eprintln!("Warning: {}", warning_msg);
            send_log(LogLevel::Warning, &warning_msg);
        }
    }

    emit_gui_progress_update(100.0, "Done! World generation completed.");
    println!("{}", "Done! World generation completed.".green().bold());
    Ok(())
}
