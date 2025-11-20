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
use std::collections::HashSet;

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
    let mut editor: WorldEditor = WorldEditor::new(args.path.clone(), &xzbbox, llbbox);

    let (outlines_with_parts, relations_with_parts) = detect_building_part_relations(&elements);

    if args.debug && !relations_with_parts.is_empty() {
        let mut sample_outlines: Vec<u64> = outlines_with_parts.iter().copied().collect();
        sample_outlines.sort_unstable();
        sample_outlines.truncate(10);
        println!(
            "Detected {} building relation(s) with parts (skipping {} outline way(s))",
            relations_with_parts.len(),
            outlines_with_parts.len()
        );ƒ
        println!("Sample outline way IDs flagged: {:?}", sample_outlines);
    }

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
    let mut current_progress_prcs: f64 = 25.0;
    let mut last_emitted_progress: f64 = current_progress_prcs;

    for element in &elements {
        process_pb.inc(1);
        current_progress_prcs += progress_increment_prcs;
        if (current_progress_prcs - last_emitted_progress).abs() > 0.25 {
            emit_gui_progress_update(current_progress_prcs, "");
            last_emitted_progress = current_progress_prcs;
        }

        if args.debug {
            process_pb.set_message(format!(
                "(Element ID: {} / Type: {})",
                element.id(),
                element.kind()
            ));
        } else {
            process_pb.set_message("");
        }

        match element {
            ProcessedElement::Way(way) => {
                if outlines_with_parts.contains(&way.id) && !way.tags.contains_key("building:part")
                {
                    // println!(
                    //     "Skipping outline way {} because its relation has building parts",
                    //     way.id
                    // );
                    continue;
                }

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
                } else if way.tags.contains_key("aeroway") || way.tags.contains_key("area:aeroway")
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
                if is_building_relation(rel) {
                    if relations_with_parts.contains(&rel.id) {
                        if args.debug {
                            println!(
                                "Skipping relation {} because its parts were handled individually",
                                rel.id
                            );
                        }
                        continue;
                    }
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
        }
    }

    process_pb.finish();

    // Generate ground layer
    let total_blocks: u64 = xzbbox.bounding_rect().total_blocks();
    let desired_updates: u64 = 1500;
    let batch_size: u64 = (total_blocks / desired_updates).max(1);

    let mut block_counter: u64 = 0;

    println!("{} Generating ground...", "[6/7]".bold());
    emit_gui_progress_update(70.0, "Generating ground...");

    let ground_pb: ProgressBar = ProgressBar::new(total_blocks);
    ground_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:45}] {pos}/{len} blocks ({eta})")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let mut gui_progress_grnd: f64 = 70.0;
    let mut last_emitted_progress: f64 = gui_progress_grnd;
    let total_iterations_grnd: f64 = total_blocks as f64;
    let progress_increment_grnd: f64 = 20.0 / total_iterations_grnd;

    let groundlayer_block = GRASS_BLOCK;

    for x in xzbbox.min_x()..=xzbbox.max_x() {
        for z in xzbbox.min_z()..=xzbbox.max_z() {
            // Add default dirt and grass layer if there isn't a stone layer already
            if !editor.check_for_block(x, 0, z, Some(&[STONE])) {
                editor.set_block(groundlayer_block, x, 0, z, None, None);
                editor.set_block(DIRT, x, -1, z, None, None);
                editor.set_block(DIRT, x, -2, z, None, None);
            }

            // Fill underground with stone
            if args.fillground {
                // Fill from bedrock+1 to 3 blocks below ground with stone
                editor.fill_blocks_absolute(
                    STONE,
                    x,
                    MIN_Y + 1,
                    z,
                    x,
                    editor.get_absolute_y(x, -3, z),
                    z,
                    None,
                    None,
                );
            }
            // Generate a bedrock level at MIN_Y
            editor.set_block_absolute(BEDROCK, x, MIN_Y, z, None, Some(&[BEDROCK]));

            block_counter += 1;
            // Use manual % check since is_multiple_of() is unstable on stable Rust
            #[allow(clippy::manual_is_multiple_of)]
            if block_counter % batch_size == 0 {
                ground_pb.inc(batch_size);
            }

            gui_progress_grnd += progress_increment_grnd;
            if (gui_progress_grnd - last_emitted_progress).abs() > 0.25 {
                emit_gui_progress_update(gui_progress_grnd, "");
                last_emitted_progress = gui_progress_grnd;
            }
        }
    }

    // Set sign for player orientation
    /*editor.set_sign(
        "↑".to_string(),
        "Generated World".to_string(),
        "This direction".to_string(),
        "".to_string(),
        9,
        -61,
        9,
        6,
    );*/

    ground_pb.inc(block_counter % batch_size);
    ground_pb.finish();

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
