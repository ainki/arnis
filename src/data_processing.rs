use crate::args::Args;
use crate::block_definitions::{BEDROCK, DIRT, GRASS_BLOCK, STONE};
use crate::coordinate_system::cartesian::XZBBox;
use crate::coordinate_system::geographic::LLBBox;
use crate::element_processing::*;
use crate::ground::Ground;
use crate::osm_parser::{
    ProcessedElement, ProcessedMember, ProcessedMemberRole, ProcessedRelation,
};
use crate::progress::emit_gui_progress_update;
use crate::telemetry::{send_log, LogLevel};
use crate::world_editor::WorldEditor;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

fn is_building_relation(relation: &ProcessedRelation) -> bool {
    relation.tags.contains_key("building")
        || relation.tags.contains_key("building:part")
        || relation
            .tags
            .get("type")
            .is_some_and(|value| value == "building")
}

fn member_is_part(member: &ProcessedMember) -> bool {
    member.role == ProcessedMemberRole::Part || member.way.tags.contains_key("building:part")
}

fn detect_building_part_relations(elements: &[ProcessedElement]) -> (HashSet<u64>, HashSet<u64>) {
    let mut outlines_with_parts: HashSet<u64> = HashSet::new();
    let mut relations_with_parts: HashSet<u64> = HashSet::new();

    for element in elements {
        if let ProcessedElement::Relation(relation) = element {
            if !is_building_relation(relation) {
                continue;
            }

            let has_part_members = relation.members.iter().any(member_is_part);

            if has_part_members {
                relations_with_parts.insert(relation.id);

                for member in relation
                    .members
                    .iter()
                    .filter(|member| !member_is_part(member))
                {
                    outlines_with_parts.insert(member.way.id);
                }
            }
        }
    }

    (outlines_with_parts, relations_with_parts)
}

pub const MIN_Y: i32 = -64;

pub fn generate_world(
    elements: Vec<ProcessedElement>,
    xzbbox: XZBBox,
    llbbox: LLBBox,
    ground: Ground,
    args: &Args,
) -> Result<(), String> {
    // Configure Rayon to ensure all cores are used
    let num_threads = rayon::current_num_threads();
    println!("Using {} threads for parallel processing", num_threads);

    let (outlines_with_parts, relations_with_parts) = detect_building_part_relations(&elements);

    if args.debug && !relations_with_parts.is_empty() {
        let mut sample_outlines: Vec<u64> = outlines_with_parts.iter().copied().collect();
        sample_outlines.sort_unstable();
        sample_outlines.truncate(10);
        println!(
            "Detected {} building relation(s) with parts (skipping {} outline way(s))",
            relations_with_parts.len(),
            outlines_with_parts.len()
        );
        println!("Sample outline way IDs flagged: {:?}", sample_outlines);
    }

    let mut editor: WorldEditor = WorldEditor::new(args.path.clone(), &xzbbox, llbbox);

    println!("{} Processing data...", "[4/7]".bold());

    // Set ground reference in the editor to enable elevation-aware block placement
    editor.set_ground(&ground);

    println!("{} Processing terrain...", "[5/7]".bold());
    emit_gui_progress_update(25.0, "Processing terrain...");

    // Process data
    let elements_count: usize = elements.len();
    println!(
        "Processing {} elements with chunk-level locking for maximum parallelism",
        elements_count
    );
    let process_pb: ProgressBar = ProgressBar::new(elements_count as u64);
    process_pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:45.white/black}] {pos}/{len} elements ({eta}) {msg}")
        .unwrap()
        .progress_chars("█▓░"));

    let progress_increment_prcs: f64 = 45.0 / elements_count as f64;
    let current_progress = AtomicU64::new(0); // Count processed elements

    // WorldEditor now has interior mutability via Arc<Mutex<WorldToModify>>
    // No need for external locking - all threads can work truly in parallel!
    let editor = Arc::new(editor);
    let process_pb = Arc::new(Mutex::new(process_pb));

    // Shuffle elements to distribute heavy elements across the work queue
    // This prevents all complex elements from being processed sequentially
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    let mut elements_vec: Vec<&ProcessedElement> = elements.iter().collect();
    elements_vec.shuffle(&mut thread_rng());

    // Process each element individually to maximize work stealing
    // Force minimum chunk size to ensure fine-grained work distribution
    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::ParallelIterator;

    elements_vec
        .par_iter()
        .with_min_len(1)
        .with_max_len(1)
        .for_each(|element| {
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
                    if outlines_with_parts.contains(&way.id) && !way.tags.contains_key("building:part")
                    {
                        // Skip outline ways that are part of building relations with parts
                        return;
                    }

                    if way.tags.contains_key("building") || way.tags.contains_key("building:part") {
                        buildings::generate_buildings(&editor, way, args, None);
                    } else if way.tags.contains_key("highway") {
                        highways::generate_highways(&editor, element, args, &elements);
                    } else if way.tags.contains_key("landuse") {
                        landuse::generate_landuse(&editor, way, args);
                    } else if way.tags.contains_key("natural") {
                        natural::generate_natural(&editor, element, args);
                    } else if way.tags.contains_key("amenity") {
                        amenities::generate_amenities(&editor, element, args);
                    } else if way.tags.contains_key("leisure") {
                        leisure::generate_leisure(&editor, way, args);
                    } else if way.tags.contains_key("barrier") {
                        barriers::generate_barriers(&editor, element);
                    } else if let Some(val) = way.tags.get("waterway") {
                        if val == "dock" {
                            // docks count as water areas
                            water_areas::generate_water_area_from_way(&editor, way);
                        } else {
                            waterways::generate_waterways(&editor, way);
                        }
                    } else if way.tags.contains_key("bridge") {
                        //bridges::generate_bridges(&editor, way, ground_level); // TODO FIX
                    } else if way.tags.contains_key("railway") {
                        railways::generate_railways(&editor, way);
                    } else if way.tags.contains_key("roller_coaster") {
                        railways::generate_roller_coaster(&editor, way);
                    } else if way.tags.contains_key("aeroway")
                        || way.tags.contains_key("area:aeroway")
                    {
                        highways::generate_aeroway(&editor, way, args);
                    } else if way.tags.get("service") == Some(&"siding".to_string()) {
                        highways::generate_siding(&editor, way);
                    } else if way.tags.contains_key("man_made") {
                        man_made::generate_man_made(&editor, element, args);
                    }
                }
                ProcessedElement::Node(node) => {
                    if node.tags.contains_key("door") || node.tags.contains_key("entrance") {
                        doors::generate_doors(&editor, node);
                    } else if node.tags.contains_key("natural")
                        && node.tags.get("natural") == Some(&"tree".to_string())
                    {
                        natural::generate_natural(&editor, element, args);
                    } else if node.tags.contains_key("amenity") {
                        amenities::generate_amenities(&editor, element, args);
                    } else if node.tags.contains_key("barrier") {
                        barriers::generate_barrier_nodes(&editor, node);
                    } else if node.tags.contains_key("highway") {
                        highways::generate_highways(&editor, element, args, &elements);
                    } else if node.tags.contains_key("tourism") {
                        tourisms::generate_tourisms(&editor, node);
                    } else if node.tags.contains_key("man_made") {
                        man_made::generate_man_made_nodes(&editor, node);
                    }
                }
                ProcessedElement::Relation(rel) => {
                    if is_building_relation(rel) {
                        if relations_with_parts.contains(&rel.id) {
                            if args.debug {
                                println!(
                                    "Skipping relation {} because its parts were handled individually",
                                    rel.id
                                );
                            }
                            return;
                        }
                        buildings::generate_building_from_relation(&editor, rel, args);
                    } else if rel.tags.contains_key("water")
                        || rel
                            .tags
                            .get("natural")
                            .map(|val| val == "water" || val == "bay")
                            .unwrap_or(false)
                    {
                        water_areas::generate_water_areas_from_relation(&editor, rel);
                    } else if rel.tags.contains_key("natural") {
                        natural::generate_natural_from_relation(&editor, rel, args);
                    } else if rel.tags.contains_key("landuse") {
                        landuse::generate_landuse_from_relation(&editor, rel, args);
                    } else if rel.tags.get("leisure") == Some(&"park".to_string()) {
                        leisure::generate_leisure_from_relation(&editor, rel, args);
                    } else if rel.tags.contains_key("man_made") {
                        man_made::generate_man_made(
                            &editor,
                            &ProcessedElement::Relation(rel.clone()),
                            args,
                        );
                    }
                }
            } // Close match

            // Update progress - track element count for occasional updates
            let elem_num = current_progress.fetch_add(1, Ordering::Relaxed);

            // Only update progress bar and GUI every ~50 elements (reduces lock contention)
            if elem_num % 50 == 0 {
                let new_progress =
                    ((25.0 + (elem_num as f64 * progress_increment_prcs)) * 10.0) as u64;
                if new_progress % 30 == 0 {
                    emit_gui_progress_update(new_progress as f64 / 10.0, "");
                }

                let pb = process_pb.lock().unwrap();
                pb.set_position(elem_num);
                drop(pb);
            }
        }); // Close par_iter forEach

    // Final progress bar update
    {
        let pb = process_pb.lock().unwrap();
        pb.set_position(elements.len() as u64);
    }

    let process_pb = Arc::try_unwrap(process_pb)
        .unwrap_or_else(|arc| {
            eprintln!("Warning: Progress bar still has references");
            (*arc).lock().unwrap().clone().into()
        })
        .into_inner()
        .unwrap();
    process_pb.finish();

    // Extract editor from Arc
    let editor = Arc::try_unwrap(editor)
        .unwrap_or_else(|_| panic!("Failed to unwrap editor - still has references"));

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

    // Editor is already Arc-wrapped from earlier
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
            chunk
                .iter()
                .map(|(x, z)| editor.get_absolute_y(*x, -3, *z))
                .collect()
        } else {
            vec![]
        };

        // Editor is already thread-safe with interior mutability
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

        // Update progress bar with actual chunk size
        ground_pb.inc(chunk_len);

        // Update GUI progress
        let new_progress = (70.0 + (count as f64 * progress_increment_grnd)) * 10.0;
        let new_progress_u64 = new_progress as u64;
        let prev = gui_progress.fetch_max(new_progress_u64, Ordering::Relaxed);
        if new_progress_u64 > prev.saturating_add(2) {
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

    // Editor is already mutable through interior mutability

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinate_system::geographic::LLBBox;
    use crate::osm_parser::parse_osm_data;
    use serde_json::Value;

    #[test]
    fn relation_outline_is_flagged_for_skipping() {
        let json = std::fs::read_to_string("tests/data/relation_12032571.json")
            .expect("fixture missing: relation_12032571.json");
        let json_value: Value = serde_json::from_str(&json).unwrap();
        let bbox = LLBBox::new(60.1542, 24.6287, 60.1545, 24.6293).unwrap();
        let (elements, _) = parse_osm_data(json_value, bbox, 1.0, false);

        let (outlines, relations) = detect_building_part_relations(&elements);

        assert!(relations.contains(&12032571));
        assert!(outlines.contains(&37104503));
    }

    #[test]
    fn relation_13981741_outline_is_flagged() {
        let json = std::fs::read_to_string("tests/data/relation_13981741.json")
            .expect("fixture missing: relation_13981741.json");
        let json_value: Value = serde_json::from_str(&json).unwrap();
        let bbox = LLBBox::new(60.1480, 24.6520, 60.1515, 24.6585).unwrap();
        let (elements, _) = parse_osm_data(json_value, bbox, 1.0, false);

        let (outlines, relations) = detect_building_part_relations(&elements);

        let outline_way = elements.iter().find_map(|element| {
            if let ProcessedElement::Way(way) = element {
                (way.id == 972283013).then_some(way)
            } else {
                None
            }
        });

        assert!(relations.contains(&13981741));
        assert!(outlines.contains(&972283013));
        assert!(outline_way.is_some());
        assert!(!outline_way.unwrap().tags.contains_key("building:part"));
    }

    #[test]
    fn relation_11484587_outline_is_flagged() {
        let json = std::fs::read_to_string("tests/data/relation_11484587.json")
            .expect("fixture missing: relation_11484587.json");
        let json_value: Value = serde_json::from_str(&json).unwrap();
        let bbox = LLBBox::new(60.1483, 24.6523, 60.1493, 24.6539).unwrap();
        let (elements, _) = parse_osm_data(json_value, bbox, 1.0, false);

        let (outlines, relations) = detect_building_part_relations(&elements);

        let outline_way = elements.iter().find_map(|element| {
            if let ProcessedElement::Way(way) = element {
                (way.id == 840899088).then_some(way)
            } else {
                None
            }
        });

        assert!(relations.contains(&11484587));
        assert!(outlines.contains(&840899088));
        assert!(outline_way.is_some());
        assert!(!outline_way.unwrap().tags.contains_key("building:part"));
    }
}
