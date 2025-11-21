use crate::args::Args;
use crate::block_definitions::*;
use crate::bresenham::bresenham_line;
use crate::colors::color_text_to_rgb_tuple;
use crate::coordinate_system::cartesian::XZPoint;
use crate::element_processing::subprocessor::buildings_interior::generate_building_interior;
use crate::floodfill::flood_fill_area;
use crate::osm_parser::{ProcessedMemberRole, ProcessedRelation, ProcessedWay, ProcessedNode};
use crate::world_editor::WorldEditor;
use rand::Rng;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Duration;

/// Enum representing different roof types
#[derive(Debug, Clone, Copy, PartialEq)]
enum RoofType {
    Gabled,    // Two sloping sides meeting at a ridge
    Hipped, // All sides slope downwards to walls (including Half-hipped, Gambrel, Mansard variations)
    Skillion, // Single sloping surface
    Pyramidal, // All sides come to a point at the top
    Dome,   // Rounded, hemispherical structure
    Flat,   // Default flat roof
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoofOrientationHint {
    Along,
    Across,
}

#[derive(Clone)]
pub(crate) struct HolePolygon<'a> {
    way: &'a ProcessedWay,
    add_walls: bool,
}

#[inline]
pub fn generate_buildings(
    editor: &WorldEditor,
    element: &ProcessedWay,
    args: &Args,
    relation_levels: Option<i32>,
    hole_polygons: Option<&[HolePolygon<'_>]>,
) {
    let is_building_part = element.tags.contains_key("building:part");
    let is_roof_part = is_roof_building_part(&element.tags);

    // Get min_level (or equivalent) first so we can use it both for start_level and building height calculations
    let min_level = element
        .tags
        .get("building:min_level")
        .or_else(|| element.tags.get("min_level"))
        .or_else(|| {
            if is_building_part {
                element.tags.get("level")
            } else {
                None
            }
        })
        .and_then(|value| parse_level_to_i32(value))
        .unwrap_or(0);

    let scale_factor = args.scale;

    let min_level_offset = multiply_scale(min_level * 4, scale_factor);

    let min_height_meters = element
        .tags
        .get("building:min_height")
        .or_else(|| element.tags.get("min_height"))
        .and_then(|value| parse_height_to_f64(value));

    let min_height_offset = min_height_meters
        .map(|meters| (meters * scale_factor).floor() as i32)
        .unwrap_or(0);

    let elevation_offset = min_level_offset.max(min_height_offset);

    // Calculate y-offset for non-terrain mode for absolute positioning
    let abs_terrain_offset = if !args.terrain { args.ground_level } else { 0 };

    // Cache floodfill result: compute once and reuse throughout
    let polygon_coords: Vec<(i32, i32)> = element.nodes.iter().map(|n| (n.x, n.z)).collect();
    let mut cached_floor_area: Vec<(i32, i32)> =
        flood_fill_area(&polygon_coords, args.timeout.as_ref());

    let mut has_inner_wall_holes = false;
    let mut part_coverage_ratio = 0.0;
    let mut part_coverage_points = 0usize;

    if let Some(holes) = hole_polygons {
        if !holes.is_empty() {
            let mut hole_points: HashSet<(i32, i32)> = HashSet::new();
            let mut part_points: HashSet<(i32, i32)> = HashSet::new();
            for hole in holes {
                let hole_coords: Vec<(i32, i32)> =
                    hole.way.nodes.iter().map(|n| (n.x, n.z)).collect();
                if hole_coords.len() < 3 {
                    continue;
                }
                let hole_area = flood_fill_area(&hole_coords, args.timeout.as_ref());
                if hole.add_walls {
                    has_inner_wall_holes = true;
                }
                for point in hole_area {
                    if !hole.add_walls {
                        part_points.insert(point);
                    }
                    hole_points.insert(point);
                }
            }

            if !hole_points.is_empty() {
                cached_floor_area.retain(|point| !hole_points.contains(point));
            }

            part_coverage_points = part_points.len();
        }
    }

    let cached_footprint_size = cached_floor_area.len();
    if cached_footprint_size > 0 {
        part_coverage_ratio = part_coverage_points as f64 / cached_footprint_size as f64;
    }
    if cached_footprint_size == 0 {
        return;
    }

    // Use fixed starting Y coordinate based on maximum ground level when terrain is enabled
    let start_y_offset = if args.terrain {
        // Get nodes' XZ points to find minimum elevation across the footprint
        let building_points: Vec<XZPoint> = element
            .nodes
            .iter()
            .map(|n| {
                XZPoint::new(
                    n.x - editor.get_min_coords().0,
                    n.z - editor.get_min_coords().1,
                )
            })
            .collect();

        // Collect ground levels for all footprint points (if available)
        let mut levels: Vec<i32> = Vec::with_capacity(building_points.len());
        for point in &building_points {
            if let Some(ground) = editor.get_ground() {
                levels.push(ground.level(*point));
            }
        }

        // If we have no samples, fall back to provided ground level
        if levels.is_empty() {
            args.ground_level + elevation_offset
        } else {
            // Use the minimum ground level across the footprint so buildings rest on the lowest sampled point
            let min_sample = *levels.iter().min().unwrap();

            // Use the minimum sampled level + min_level offset as the fixed base for the entire building
            min_sample + elevation_offset
        }
    } else {
        // When terrain is disabled, just use the precomputed offset
        elevation_offset
    };

    // Calculate building bounds and floor area before processing interior
    let min_x = element.nodes.iter().map(|n| n.x).min().unwrap_or(0);
    let max_x = element.nodes.iter().map(|n| n.x).max().unwrap_or(0);
    let min_z = element.nodes.iter().map(|n| n.z).min().unwrap_or(0);
    let max_z = element.nodes.iter().map(|n| n.z).max().unwrap_or(0);

    let mut previous_node: Option<(i32, i32)> = None;
    let mut corner_addup: (i32, i32, i32) = (0, 0, 0);

    // Get building type for type-specific block selection
    let building_type = element
        .tags
        .get("building")
        .or_else(|| element.tags.get("building:part"))
        .map(|s| s.as_str())
        .unwrap_or("yes");

    let wall_block: Block = if element.tags.get("historic") == Some(&"castle".to_string()) {
        // Historic forts and castles should use stone/brick materials
        get_castle_wall_block()
    } else {
        element
            .tags
            .get("building:colour")
            .and_then(|building_colour: &String| {
                color_text_to_rgb_tuple(building_colour)
                    .map(|rgb: (u8, u8, u8)| get_building_wall_block_for_color(rgb))
            })
            .unwrap_or_else(get_fallback_building_block)
    };

    let floor_block: Block = get_random_floor_block();

    // Select window type based on building type
    let window_block: Block = get_window_block_for_building_type(building_type);

    // Set to store processed flood fill points
    let mut processed_points: HashSet<(i32, i32)> = HashSet::new();
    let mut building_height: i32 = ((6.0 * scale_factor) as i32).max(3); // Default building height with scale and minimum
    let mut is_tall_building = false;
    let mut has_explicit_levels = false;
    let mut rng = rand::thread_rng();
    let use_vertical_windows = rng.gen_bool(0.7);
    let use_accent_roof_line = rng.gen_bool(0.25);

    // Random accent block selection for this building
    let accent_blocks = [
        POLISHED_ANDESITE,
        SMOOTH_STONE,
        STONE_BRICKS,
        MUD_BRICKS,
        ANDESITE,
        CHISELED_STONE_BRICKS,
    ];
    let accent_block = accent_blocks[rng.gen_range(0..accent_blocks.len())];

    // Skip if 'layer' or 'level' is negative in the tags
    if let Some(layer) = element.tags.get("layer") {
        if layer.parse::<i32>().unwrap_or(0) < 0 {
            return;
        }
    }

    if let Some(level) = element.tags.get("level") {
        if level.parse::<i32>().unwrap_or(0) < 0 {
            return;
        }
    }

    // Determine building height from tags
    if let Some(levels_total) = element
        .tags
        .get("building:levels")
        .or_else(|| element.tags.get("levels"))
        .and_then(|value| parse_level_to_i32(value))
    {
        has_explicit_levels = true;
        if levels_total >= 1 {
            let effective_levels = if is_building_part {
                (levels_total - min_level).max(1)
            } else {
                levels_total
            };

            // Use declared building levels directly (match OSM data)
            building_height = multiply_scale(effective_levels * 4 + 2, scale_factor);
            building_height = building_height.max(3);

            // Mark as tall building if more than 7 stories (use effective floors)
            if effective_levels > 7 {
                is_tall_building = true;
            }
        }
    }

    // building_levels_opt removed (no longer used)

    if !has_explicit_levels {
        if let Some(height_meters) = element
            .tags
            .get("building:height")
            .or_else(|| element.tags.get("height"))
            .and_then(|value| parse_height_to_f64(value))
        {
            let mut effective_height = height_meters;
            if is_building_part {
                if let Some(min_h) = min_height_meters {
                    effective_height = (height_meters - min_h).max(1.0);
                } else if min_level > 0 {
                    let approximate_base = (min_level * 4) as f64;
                    effective_height = (height_meters - approximate_base).max(1.0);
                }
            }

            building_height = (effective_height * scale_factor).floor() as i32;
            building_height = building_height.max(3);

            // Mark as tall building if height suggests more than 7 stories
            if effective_height > 28.0 {
                is_tall_building = true;
            }
        }
    }

    if let Some(levels) = relation_levels {
        // Relation-provided levels follow the declared OSM value
        building_height = multiply_scale(levels * 4, scale_factor);
        building_height = building_height.max(3);

        // Mark as tall building if more than 7 stories
        if levels > 7 {
            is_tall_building = true;
        }
    }

    if is_roof_part {
        let roof_type = element
            .tags
            .get("roof:shape")
            .map(|value| roof_type_from_str(value))
            .unwrap_or(RoofType::Flat);

        // Keep the previously computed building height to anchor the roof at the
        // part's top (e.g. roof sitting on top of lower floors). When no height
        // metadata exists this defaults to a small positive offset.
        let roof_vertical_offset = start_y_offset + building_height.max(1) - 1;

        generate_roof(
            editor,
            element,
            roof_vertical_offset,
            0,
            floor_block,
            wall_block,
            accent_block,
            roof_type,
            &cached_floor_area,
            abs_terrain_offset,
            args,
        );

        return;
    }

    // Determine accent line usage based on whether building has multiple floors
    let has_multiple_floors = building_height > 6;
    let use_accent_lines = has_multiple_floors && rng.gen_bool(0.2);
    let use_vertical_accent = has_multiple_floors && !use_accent_lines && rng.gen_bool(0.1);

    if let Some(amenity_type) = element.tags.get("amenity") {
        if amenity_type == "shelter" {
            let roof_block: Block = STONE_BRICK_SLAB;

            // Use cached floor area instead of recalculating
            let roof_area: &Vec<(i32, i32)> = &cached_floor_area;

            // Place fences and roof slabs at each corner node directly
            for node in &element.nodes {
                let x: i32 = node.x;
                let z: i32 = node.z;

                for shelter_y in 1..=multiply_scale(4, scale_factor) {
                    editor.set_block(OAK_FENCE, x, shelter_y, z, None, None);
                }
                editor.set_block(roof_block, x, 5, z, None, None);
            }

            // Flood fill the roof area - parallelize for large roofs
            if roof_area.len() > 500 {
                roof_area.par_iter().for_each(|(x, z)| {
                    editor.set_block(roof_block, *x, 5, *z, None, None);
                });
            } else {
                for (x, z) in roof_area.iter() {
                    editor.set_block(roof_block, *x, 5, *z, None, None);
                }
            }

            return;
        }
    }

    if let Some(building_type) = element.tags.get("building") {
        if building_type == "garage" {
            building_height = ((2.0 * scale_factor) as i32).max(3);
        } else if building_type == "shed" {
            building_height = ((2.0 * scale_factor) as i32).max(3);

            if element.tags.contains_key("bicycle_parking") {
                let ground_block: Block = OAK_PLANKS;
                let roof_block: Block = STONE_BLOCK_SLAB;

                // Use cached floor area instead of recalculating
                let floor_area: &Vec<(i32, i32)> = &cached_floor_area;

                // Fill the floor area - parallelize for large areas
                if floor_area.len() > 500 {
                    floor_area.par_iter().for_each(|(x, z)| {
                        editor.set_block(ground_block, *x, 0, *z, None, None);
                    });
                } else {
                    for (x, z) in floor_area.iter() {
                        editor.set_block(ground_block, *x, 0, *z, None, None);
                    }
                }

                // Place fences and roof slabs at each corner node directly
                for node in &element.nodes {
                    let x: i32 = node.x;
                    let z: i32 = node.z;

                    for dy in 1..=4 {
                        editor.set_block(OAK_FENCE, x, dy, z, None, None);
                    }
                    editor.set_block(roof_block, x, 5, z, None, None);
                }

                // Flood fill the roof area
                for (x, z) in floor_area.iter() {
                    editor.set_block(roof_block, *x, 5, *z, None, None);
                }

                return;
            }
        } else if building_type == "parking"
            || element
                .tags
                .get("parking")
                .is_some_and(|p| p == "multi-storey")
        {
            // Parking building structure

            // If an explicit `building:levels` is provided, treat as (levels + 1), else derive from height.
            let num_levels: i32 = if let Some(levels) = element
                .tags
                .get("building:levels")
                .and_then(|s| s.parse::<i32>().ok())
            {
                // Apply +1 for parking
                (levels + 1).max(1)
            } else {
                // Derived levels: floor every 4 blocks. Ensure a sensible minimum for parking structures.
                let derived = (building_height / 4).max(1);
                // If no explicit levels were given, enforce a minimum parking size (2 levels)
                derived.max(2)
            };

            // Use cached floor area instead of recalculating
            let floor_area: &Vec<(i32, i32)> = &cached_floor_area;

            for level in 0..num_levels {
                let current_level_y = level * 4;

                // Build walls
                for node in &element.nodes {
                    let x: i32 = node.x;
                    let z: i32 = node.z;

                    // Build walls up to the current level
                    for y in (current_level_y + 1)..=(current_level_y + 4) {
                        editor.set_block(STONE_BRICKS, x, y, z, None, None);
                    }
                }

                // Fill the floor area for each level - parallelize for large buildings
                let block_type = if level == 0 {
                    SMOOTH_STONE
                } else {
                    COBBLESTONE
                };

                // Only parallelize if floor area is large enough to benefit
                if floor_area.len() > 1000 {
                    floor_area.par_iter().for_each(|(x, z)| {
                        editor.set_block(block_type, *x, current_level_y, *z, None, None);
                    });
                } else {
                    for (x, z) in floor_area {
                        editor.set_block(block_type, *x, current_level_y, *z, None, None);
                    }
                }
            }

            // Outline for each level
            for level in 0..num_levels {
                let current_level_y = level * 4;

                // Use the nodes to create the outline
                let mut prev_outline = None;
                for node in &element.nodes {
                    let x = node.x;
                    let z = node.z;

                    if let Some((prev_x, prev_z)) = prev_outline {
                        let outline_points =
                            bresenham_line(prev_x, current_level_y, prev_z, x, current_level_y, z);
                        for (bx, _, bz) in outline_points {
                            editor.set_block(
                                SMOOTH_STONE,
                                bx,
                                current_level_y,
                                bz,
                                Some(&[COBBLESTONE, COBBLESTONE_WALL]),
                                None,
                            );
                            editor.set_block(
                                STONE_BRICK_SLAB,
                                bx,
                                current_level_y + 2,
                                bz,
                                None,
                                None,
                            );
                            if bx % 2 == 0 {
                                editor.set_block(
                                    COBBLESTONE_WALL,
                                    bx,
                                    current_level_y + 1,
                                    bz,
                                    None,
                                    None,
                                );
                            }
                        }
                    }
                    prev_outline = Some((x, z));
                }
            }

            return;
        } else if building_type == "roof" && !element.tags.contains_key("roof:shape") {
            let roof_height: i32 = 5;

            // Iterate through the nodes to create the roof edges using Bresenham's line algorithm
            for node in &element.nodes {
                let x: i32 = node.x;
                let z: i32 = node.z;

                if let Some(prev) = previous_node {
                    let bresenham_points: Vec<(i32, i32, i32)> =
                        bresenham_line(prev.0, roof_height, prev.1, x, roof_height, z);
                    for (bx, _, bz) in bresenham_points {
                        editor.set_block(STONE_BRICK_SLAB, bx, roof_height, bz, None, None);
                        // Set roof block at edge
                    }
                }

                for y in 1..=(roof_height - 1) {
                    editor.set_block(COBBLESTONE_WALL, x, y, z, None, None);
                }

                previous_node = Some((x, z));
            }

            // Use cached floor area
            let roof_area: &Vec<(i32, i32)> = &cached_floor_area;

            // Fill the interior of the roof with STONE_BRICK_SLAB
            for (x, z) in roof_area.iter() {
                editor.set_block(STONE_BRICK_SLAB, *x, roof_height, *z, None, None);
                // Set roof block
            }

            return;
        } else if building_type == "apartments" {
            // If building has no height attribute, assign a defined height
            if building_height == ((6.0 * scale_factor) as i32).max(3) {
                building_height = ((15.0 * scale_factor) as i32).max(3);
            }
        } else if building_type == "hospital" {
            // If building has no height attribute, assign a defined height
            if building_height == ((6.0 * scale_factor) as i32).max(3) {
                building_height = ((23.0 * scale_factor) as i32).max(3);
            }
        } else if building_type == "bridge" {
            generate_bridge(editor, element, args.timeout.as_ref());
            return;
        }
    }

    // Process nodes to create walls and corners (outer shell)
    const PART_COVERAGE_SKIP_THRESHOLD: f64 = 0.65;
    let suppress_outer_walls =
        !has_inner_wall_holes && part_coverage_ratio >= PART_COVERAGE_SKIP_THRESHOLD;

    if !suppress_outer_walls {
        build_wall_ring(
            &element.nodes,
            editor,
            args,
            wall_block,
            window_block,
            accent_block,
            start_y_offset,
            building_height,
            is_tall_building,
            use_vertical_windows,
            use_accent_roof_line,
            use_accent_lines,
            use_vertical_accent,
            min_level,
            min_height_offset,
            abs_terrain_offset,
            &mut corner_addup,
        );
    }

    // Add inner courtyard walls for multipolygon holes and building parts
    if let Some(hole_polygons) = hole_polygons {
        for hole in hole_polygons {
            if !hole.add_walls {
                continue;
            }

            build_wall_ring(
                &hole.way.nodes,
                editor,
                args,
                wall_block,
                window_block,
                accent_block,
                start_y_offset,
                building_height,
                is_tall_building,
                use_vertical_windows,
                use_accent_roof_line,
                use_accent_lines,
                use_vertical_accent,
                min_level,
                min_height_offset,
                abs_terrain_offset,
                &mut corner_addup,
            );
        }
    }

    // Flood-fill interior with floor variation
    {
        // Use cached floor area
        let floor_area: &Vec<(i32, i32)> = &cached_floor_area;

        // Calculate floor heights for each level based on building height
        let mut floor_levels = Vec::new();

        // Always add the ground floor
        floor_levels.push(start_y_offset);

        // Calculate additional floors if building has sufficient height
        if building_height > 6 {
            // Determine number of floors (approximately 1 floor per 4 blocks of height)
            let num_upper_floors = (building_height / 4).max(1);

            // Add Y coordinates for each upper floor - match the intermediate floor placement
            // Main building code places intermediate floors at start_y_offset + 2 + 4, start_y_offset + 2 + 8, etc.
            for floor in 1..num_upper_floors {
                floor_levels.push(start_y_offset + 2 + (floor * 4));
            }
        }

        for (x, z) in floor_area.iter().cloned() {
            if processed_points.insert((x, z)) {
                // Create foundation columns for the floor area when using terrain
                if args.terrain {
                    // Calculate actual ground level at this position
                    if let Some(ground) = editor.get_ground() {
                        ground.level(XZPoint::new(
                            x - editor.get_min_coords().0,
                            z - editor.get_min_coords().1,
                        ))
                    } else {
                        args.ground_level
                    };
                }

                // Set floor at start_y_offset
                editor.set_block_absolute(
                    floor_block,
                    x,
                    start_y_offset + abs_terrain_offset,
                    z,
                    None,
                    None,
                );

                // Set level ceilings if height > 4
                if building_height > 4 {
                    for h in (start_y_offset + 2 + 4..start_y_offset + building_height).step_by(4) {
                        if x % 5 == 0 && z % 5 == 0 {
                            // Light fixtures
                            editor.set_block_absolute(
                                GLOWSTONE,
                                x,
                                h + abs_terrain_offset,
                                z,
                                None,
                                None,
                            );
                        } else {
                            editor.set_block_absolute(
                                floor_block,
                                x,
                                h + abs_terrain_offset,
                                z,
                                None,
                                None,
                            );
                        }
                    }
                } else if x % 5 == 0 && z % 5 == 0 {
                    editor.set_block_absolute(
                        GLOWSTONE,
                        x,
                        start_y_offset + building_height + abs_terrain_offset,
                        z,
                        None,
                        None,
                    );
                }

                // Only set ceiling at proper height if we don't use a specific roof shape or roof generation is disabled
                if !args.roof
                    || !element.tags.contains_key("roof:shape")
                    || element.tags.get("roof:shape").unwrap() == "flat"
                {
                    editor.set_block_absolute(
                        floor_block,
                        x,
                        start_y_offset + building_height + abs_terrain_offset + 1,
                        z,
                        None,
                        None,
                    );
                }
            }
        }

        // Generate interior features
        if args.interior {
            // Only generate interiors for buildings that aren't special types
            let building_type = element
                .tags
                .get("building")
                .map(|s| s.as_str())
                .unwrap_or("yes");
            let skip_interior = matches!(
                building_type,
                "garage" | "shed" | "parking" | "roof" | "bridge"
            );

            if !skip_interior && floor_area.len() > 100 {
                // Only for buildings with sufficient floor area
                generate_building_interior(
                    editor,
                    floor_area,
                    min_x,
                    min_z,
                    max_x,
                    max_z,
                    start_y_offset,
                    building_height,
                    wall_block,
                    &floor_levels,
                    args,
                    element,
                    abs_terrain_offset,
                );
            }
        }
    }

    // Process roof shapes if specified and roof generation is enabled
    if args.roof {
        if let Some(roof_shape) = element.tags.get("roof:shape") {
            let roof_type = roof_type_from_str(roof_shape);

            generate_roof(
                editor,
                element,
                start_y_offset,
                building_height,
                floor_block,
                wall_block,
                accent_block,
                roof_type,
                &cached_floor_area,
                abs_terrain_offset,
                args,
            );
        } else if !is_building_part {
            // Handle buildings without explicit roof:shape tag
            let building_type = element
                .tags
                .get("building")
                .map(|s| s.as_str())
                .unwrap_or("yes");

            // Estimate number of floors from the generated height (≈4 blocks per floor)
            let approx_levels = ((building_height as f64)
                / (4.0 * args.scale.max(0.1)))
                .ceil() as i32;

            let is_low_rise_residential = approx_levels <= 3
                && matches!(
                    building_type,
                    "house"
                        | "detached"
                        | "semi_detached"
                        | "semidetached_house"
                        | "terrace"
                        | "hut"
                        | "cabin"
                        | "bungalow"
                        | "farm"
                        | "residential"
                        | "yes"
                );

            if is_low_rise_residential {
                // Use cached footprint area and size instead of recalculating
                let footprint_size = cached_footprint_size;

                // Maximum footprint size threshold for gabled roofs
                let max_footprint_for_gabled = 800;

                let mut rng = rand::thread_rng();
                if footprint_size <= max_footprint_for_gabled && rng.gen_bool(0.9) {
                    generate_roof(
                        editor,
                        element,
                        start_y_offset,
                        building_height,
                        floor_block,
                        wall_block,
                        accent_block,
                        RoofType::Gabled,
                        &cached_floor_area,
                        abs_terrain_offset,
                        args,
                    );
                }
                // If footprint too large or not selected for gabled roof, building gets default flat roof (no action needed)
            }
            // Other building types without roof:shape get default flat roof (no action needed)
        }
    } else {
        // Default flat roof - already handled by the building generation code
    }
}

fn build_wall_ring(
    nodes: &[ProcessedNode],
    editor: &WorldEditor,
    args: &Args,
    wall_block: Block,
    window_block: Block,
    accent_block: Block,
    start_y_offset: i32,
    building_height: i32,
    is_tall_building: bool,
    use_vertical_windows: bool,
    use_accent_roof_line: bool,
    use_accent_lines: bool,
    use_vertical_accent: bool,
    min_level: i32,
    min_height_offset: i32,
    abs_terrain_offset: i32,
    corner_addup: &mut (i32, i32, i32),
) {
    if nodes.len() < 2 {
        return;
    }

    let mut previous_node: Option<(i32, i32)> = None;
    for node in nodes {
        let x = node.x;
        let z = node.z;

        if let Some(prev) = previous_node {
            let bresenham_points =
                bresenham_line(prev.0, start_y_offset, prev.1, x, start_y_offset, z);
            for (bx, _, bz) in bresenham_points {
                if args.terrain && min_level == 0 && min_height_offset == 0 {
                    let local_ground_level = if let Some(ground) = editor.get_ground() {
                        ground.level(XZPoint::new(
                            bx - editor.get_min_coords().0,
                            bz - editor.get_min_coords().1,
                        ))
                    } else {
                        args.ground_level
                    };

                    for y in local_ground_level..start_y_offset + 1 {
                        editor.set_block_absolute(
                            wall_block,
                            bx,
                            y + abs_terrain_offset,
                            bz,
                            None,
                            None,
                        );
                    }
                }

                for h in (start_y_offset + 1)..=(start_y_offset + building_height) {
                    if is_tall_building && use_vertical_windows {
                        if h > start_y_offset + 1 && (bx + bz) % 3 == 0 {
                            editor.set_block_absolute(
                                window_block,
                                bx,
                                h + abs_terrain_offset,
                                bz,
                                None,
                                None,
                            );
                        } else {
                            editor.set_block_absolute(
                                wall_block,
                                bx,
                                h + abs_terrain_offset,
                                bz,
                                None,
                                None,
                            );
                        }
                    } else if h > start_y_offset + 1 && h % 4 != 0 && (bx + bz) % 6 < 3 {
                        editor.set_block_absolute(
                            window_block,
                            bx,
                            h + abs_terrain_offset,
                            bz,
                            None,
                            None,
                        );
                    } else {
                        let use_accent_line =
                            use_accent_lines && h > start_y_offset + 1 && h % 4 == 0;
                        let use_vertical_accent_here = use_vertical_accent
                            && h > start_y_offset + 1
                            && h % 4 == 0
                            && (bx + bz) % 6 < 3;

                        if use_accent_line || use_vertical_accent_here {
                            editor.set_block_absolute(
                                accent_block,
                                bx,
                                h + abs_terrain_offset,
                                bz,
                                None,
                                None,
                            );
                        } else {
                            editor.set_block_absolute(
                                wall_block,
                                bx,
                                h + abs_terrain_offset,
                                bz,
                                None,
                                None,
                            );
                        }
                    }
                }

                let roof_line_block = if use_accent_roof_line {
                    accent_block
                } else {
                    wall_block
                };
                editor.set_block_absolute(
                    roof_line_block,
                    bx,
                    start_y_offset + building_height + abs_terrain_offset + 1,
                    bz,
                    None,
                    None,
                );

                corner_addup.0 += bx;
                corner_addup.1 += bz;
                corner_addup.2 += 1;
            }
        }

        previous_node = Some((x, z));
    }
}

fn multiply_scale(value: i32, scale_factor: f64) -> i32 {
    // Use bit operations for faster multiplication when possible
    if scale_factor == 1.0 {
        value
    } else if scale_factor == 2.0 {
        value << 1
    } else if scale_factor == 4.0 {
        value << 2
    } else {
        let result = (value as f64) * scale_factor;
        result.floor() as i32
    }
}

fn parse_level_to_i32(value: &str) -> Option<i32> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }

    trimmed
        .parse::<i32>()
        .ok()
        .or_else(|| trimmed.parse::<f64>().ok().map(|num| num.floor() as i32))
}

fn parse_height_to_f64(value: &str) -> Option<f64> {
    let normalized = value.trim();
    if normalized.is_empty() {
        return None;
    }

    normalized.trim_end_matches('m').trim().parse::<f64>().ok()
}

fn is_roof_building_part(tags: &HashMap<String, String>) -> bool {
    tags
        .get("building:part")
        .map(|value| {
            value
                .split(|c| matches!(c, ';' | ','))
                .any(|token| token.trim().eq_ignore_ascii_case("roof"))
        })
        .unwrap_or(false)
}

fn roof_type_from_str(value: &str) -> RoofType {
    match value.trim().to_ascii_lowercase().as_str() {
        "flat" => RoofType::Flat,
        "gabled" | "gambrel" | "saltbox" | "mansard" | "clipped_gable" => RoofType::Gabled,
        "hipped" | "half-hipped" | "halfhipped" | "jerkinhead" => RoofType::Hipped,
        "skillion" | "shed" | "lean_to" | "lean-to" | "leanto" | "sawtooth" => RoofType::Skillion,
        "pyramidal" | "pavilion" | "cone" | "conical" | "tent" => RoofType::Pyramidal,
        "dome" | "onion" | "round" | "vault" => RoofType::Dome,
        _ => RoofType::Flat,
    }
}

/// Unified function to generate various roof types
#[allow(clippy::too_many_arguments)]
#[inline]
fn generate_roof(
    editor: &WorldEditor,
    element: &ProcessedWay,
    start_y_offset: i32,
    building_height: i32,
    floor_block: Block,
    wall_block: Block,
    accent_block: Block,
    roof_type: RoofType,
    cached_floor_area: &[(i32, i32)],
    abs_terrain_offset: i32,
    args: &Args,
) {
    let roof_height_override = parse_roof_height(&element.tags, args.scale);
    let roof_direction_vector = parse_roof_direction(&element.tags).map(direction_to_vector);
    let roof_orientation_hint = parse_roof_orientation_hint(&element.tags);

    // Use the provided cached floor area instead of recalculating
    let floor_area = cached_floor_area;

    // Pre-calculate building bounds once
    let (min_x, max_x, min_z, max_z) = element.nodes.iter().fold(
        (i32::MAX, i32::MIN, i32::MAX, i32::MIN),
        |(min_x, max_x, min_z, max_z), n| {
            (
                min_x.min(n.x),
                max_x.max(n.x),
                min_z.min(n.z),
                max_z.max(n.z),
            )
        },
    );

    let center_x = (min_x + max_x) >> 1; // Bit shift is faster than division
    let center_z = (min_z + max_z) >> 1;

    // Set base height for roof to be at least one block above building top
    let base_height = start_y_offset + building_height + 1;

    match roof_type {
        RoofType::Flat => {
            // Simple flat roof
            let flat_layers = roof_height_override.unwrap_or(1).max(1);
            for &(x, z) in floor_area {
                for layer in 0..flat_layers {
                    editor.set_block_absolute(
                        floor_block,
                        x,
                        base_height + layer + abs_terrain_offset,
                        z,
                        None,
                        None,
                    );
                }
            }
        }

        RoofType::Gabled => {
            // Pre-calculate building dimensions once
            let width = max_x - min_x;
            let length = max_z - min_z;
            let building_size = width.max(length);

            let default_height = (3.0 + (building_size as f64 * 0.15).ln().max(1.0)) as i32;
            let roof_height_boost = roof_height_override.unwrap_or(default_height).max(2);
            let roof_peak_height = base_height + roof_height_boost;

            // Pre-determine orientation and material
            let long_axis_is_x = width >= length;
            let mut is_wider_than_long = width > length;
            if let Some((dir_x, dir_z)) = roof_direction_vector {
                if dir_z.abs() > dir_x.abs() {
                    is_wider_than_long = true;
                } else if dir_x.abs() > dir_z.abs() {
                    is_wider_than_long = false;
                }
            } else if let Some(hint) = roof_orientation_hint {
                is_wider_than_long = match hint {
                    RoofOrientationHint::Along => long_axis_is_x,
                    RoofOrientationHint::Across => !long_axis_is_x,
                };
            }
            let max_distance = if is_wider_than_long {
                (length >> 1).max(1)
            } else {
                (width >> 1).max(1)
            };

            // 50% accent block, otherwise wall block for roof
            let mut rng = rand::thread_rng();
            let roof_block = if rng.gen_bool(0.5) {
                accent_block
            } else {
                wall_block
            };

            // Pre-allocate with capacity hint for better performance
            let mut roof_heights = Vec::with_capacity(floor_area.len());
            let mut blocks_to_place = Vec::with_capacity(floor_area.len() * 4);

            // First pass: calculate all roof heights using vectorized operations
            for &(x, z) in floor_area {
                let distance_to_ridge = if is_wider_than_long {
                    (z - center_z).abs()
                } else {
                    (x - center_x).abs()
                };

                let roof_height = if distance_to_ridge == 0
                    && ((is_wider_than_long && z == center_z)
                        || (!is_wider_than_long && x == center_x))
                {
                    roof_peak_height
                } else {
                    let slope_ratio = distance_to_ridge as f64 / max_distance.max(1) as f64;
                    (roof_peak_height as f64 - (slope_ratio * roof_height_boost as f64)) as i32
                }
                .max(base_height);

                roof_heights.push(((x, z), roof_height));
            }

            // Second pass: batch process blocks with pre-computed stair materials
            let stair_block_material = get_stair_block_for_material(roof_block);

            for &((x, z), roof_height) in &roof_heights {
                // Check neighboring heights efficiently using iterator
                let has_lower_neighbor = roof_heights
                    .iter()
                    .filter_map(|&((nx, nz), nh)| {
                        if (nx - x).abs() + (nz - z).abs() == 1 {
                            Some(nh)
                        } else {
                            None
                        }
                    })
                    .any(|nh| nh < roof_height);

                // Fill from base height to calculated roof height
                for y in base_height..=roof_height {
                    if y == roof_height && has_lower_neighbor {
                        // Pre-compute stair direction
                        let stair_block_with_props = if is_wider_than_long {
                            if z < center_z {
                                create_stair_with_properties(
                                    stair_block_material,
                                    StairFacing::South,
                                    StairShape::Straight,
                                )
                            } else {
                                create_stair_with_properties(
                                    stair_block_material,
                                    StairFacing::North,
                                    StairShape::Straight,
                                )
                            }
                        } else if x < center_x {
                            create_stair_with_properties(
                                stair_block_material,
                                StairFacing::East,
                                StairShape::Straight,
                            )
                        } else {
                            create_stair_with_properties(
                                stair_block_material,
                                StairFacing::West,
                                StairShape::Straight,
                            )
                        };

                        blocks_to_place.push((x, y, z, roof_block, Some(stair_block_with_props)));
                    } else {
                        blocks_to_place.push((x, y, z, roof_block, None));
                    }
                }
            }

            // Batch place all blocks to reduce function call overhead
            for (x, y, z, block, stair_props) in blocks_to_place {
                if let Some(stair_block) = stair_props {
                    editor.set_block_with_properties_absolute(
                        stair_block,
                        x,
                        y + abs_terrain_offset,
                        z,
                        None,
                        None,
                    );
                } else {
                    editor.set_block_absolute(block, x, y + abs_terrain_offset, z, None, None);
                }
            }
        }

        RoofType::Hipped => {
            // Calculate building dimensions and determine the long axis
            let width = max_x - min_x;
            let length = max_z - min_z;

            // Determine if building is significantly rectangular or more square-shaped
            let is_rectangular =
                (width as f64 / length as f64 > 1.3) || (length as f64 / width as f64 > 1.3);
            let mut long_axis_is_x = width >= length;
            if let Some((dir_x, dir_z)) = roof_direction_vector {
                if dir_x.abs() > dir_z.abs() {
                    long_axis_is_x = true;
                } else if dir_z.abs() > dir_x.abs() {
                    long_axis_is_x = false;
                }
            }
            if let Some(hint) = roof_orientation_hint {
                long_axis_is_x = match hint {
                    RoofOrientationHint::Along => long_axis_is_x,
                    RoofOrientationHint::Across => !long_axis_is_x,
                };
            }

            // Make roof taller and more pointy
            let default_peak = if width.max(length) > 20 { 7 } else { 5 };
            let roof_rise = roof_height_override.unwrap_or(default_peak).max(3);
            let roof_peak_height = base_height + roof_rise;

            // 50% accent block, otherwise wall block for roof
            let mut rng = rand::thread_rng();
            let roof_block = if rng.gen_bool(0.5) {
                accent_block
            } else {
                wall_block
            };

            // Find the building's approximate center line along the long axis
            if is_rectangular {
                // First pass: calculate all roof heights
                let mut roof_heights = HashMap::new();

                for &(x, z) in floor_area {
                    // Calculate distance to the ridge line
                    let distance_to_ridge = if long_axis_is_x {
                        // Distance in Z direction for X-axis ridge
                        (z - center_z).abs()
                    } else {
                        // Distance in X direction for Z-axis ridge
                        (x - center_x).abs()
                    };

                    // Calculate maximum distance from ridge to edge
                    let max_distance_from_ridge = if long_axis_is_x {
                        (max_z - min_z) / 2
                    } else {
                        (max_x - min_x) / 2
                    };

                    // Create proper slope from ridge (high) to edges (low)
                    let slope_factor = if max_distance_from_ridge > 0 {
                        distance_to_ridge as f64 / max_distance_from_ridge as f64
                    } else {
                        0.0
                    };

                    // Ridge gets peak height, edges get base height
                    let roof_height = roof_peak_height
                        - (slope_factor * (roof_peak_height - base_height) as f64) as i32;
                    let roof_y = roof_height.max(base_height);
                    roof_heights.insert((x, z), roof_y);
                }

                // Second pass: place blocks with stairs at height transitions
                for &(x, z) in floor_area {
                    let roof_height = roof_heights[&(x, z)];

                    // Fill from base to calculated height
                    for y in base_height..=roof_height {
                        if y == roof_height {
                            // Check if this is a height transition point by looking at neighboring blocks
                            let has_lower_neighbor =
                                [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)].iter().any(
                                    |(nx, nz)| {
                                        roof_heights
                                            .get(&(*nx, *nz))
                                            .is_some_and(|&nh| nh < roof_height)
                                    },
                                );

                            if has_lower_neighbor {
                                // Determine stair direction based on ridge orientation and position
                                let stair_block_material = get_stair_block_for_material(roof_block);
                                let stair_block_with_props = if long_axis_is_x {
                                    // Ridge runs along X, slopes in Z direction
                                    if z < center_z {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::South,
                                            StairShape::Straight,
                                        ) // Facing toward center (south)
                                    } else {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::North,
                                            StairShape::Straight,
                                        ) // Facing toward center (north)
                                    }
                                } else {
                                    // Ridge runs along Z, slopes in X direction
                                    if x < center_x {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::East,
                                            StairShape::Straight,
                                        ) // Facing toward center (east)
                                    } else {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::West,
                                            StairShape::Straight,
                                        ) // Facing toward center (west)
                                    }
                                };
                                editor.set_block_with_properties_absolute(
                                    stair_block_with_props,
                                    x,
                                    y + abs_terrain_offset,
                                    z,
                                    None,
                                    None,
                                );
                            } else {
                                // Use regular roof block where height doesn't change (ridge area)
                                editor.set_block_absolute(
                                    roof_block,
                                    x,
                                    y + abs_terrain_offset,
                                    z,
                                    None,
                                    None,
                                );
                            }
                        } else {
                            // Fill interior with solid blocks
                            editor.set_block_absolute(
                                roof_block,
                                x,
                                y + abs_terrain_offset,
                                z,
                                None,
                                None,
                            );
                        }
                    }
                }
            } else {
                // For more complex or square buildings, use distance from center approach

                // First pass: calculate all roof heights based on distance from center
                let mut roof_heights = HashMap::new();

                for &(x, z) in floor_area {
                    // Calculate distance from center point
                    let dx = (x - center_x) as f64;
                    let dz = (z - center_z) as f64;
                    let distance_from_center = (dx * dx + dz * dz).sqrt();

                    // Calculate maximum possible distance from center to any corner
                    let max_distance = {
                        let corner_distances = [
                            ((min_x - center_x).pow(2) + (min_z - center_z).pow(2)) as f64,
                            ((min_x - center_x).pow(2) + (max_z - center_z).pow(2)) as f64,
                            ((max_x - center_x).pow(2) + (min_z - center_z).pow(2)) as f64,
                            ((max_x - center_x).pow(2) + (max_z - center_z).pow(2)) as f64,
                        ];
                        corner_distances
                            .iter()
                            .fold(0.0f64, |a, &b| a.max(b))
                            .sqrt()
                    };

                    // Create slope from center (high) to edges (low)
                    let distance_factor = if max_distance > 0.0 {
                        (distance_from_center / max_distance).min(1.0)
                    } else {
                        0.0
                    };

                    // Center gets peak height, edges get base height
                    let roof_height = roof_peak_height
                        - (distance_factor * (roof_peak_height - base_height) as f64) as i32;
                    let roof_y = roof_height.max(base_height);
                    roof_heights.insert((x, z), roof_y);
                }

                // Second pass: place blocks with stairs at height transitions
                for &(x, z) in floor_area {
                    let roof_height = roof_heights[&(x, z)];

                    // Fill from base height to calculated roof height
                    for y in base_height..=roof_height {
                        if y == roof_height {
                            // Check if this is a height transition point by looking at neighboring blocks
                            let has_lower_neighbor =
                                [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)].iter().any(
                                    |(nx, nz)| {
                                        roof_heights
                                            .get(&(*nx, *nz))
                                            .is_some_and(|&nh| nh < roof_height)
                                    },
                                );

                            if has_lower_neighbor {
                                // For complex buildings, determine stair direction based on slope toward center
                                let center_dx = x - center_x;
                                let center_dz = z - center_z;

                                let stair_block_material = get_stair_block_for_material(roof_block);
                                let stair_block = if center_dx.abs() > center_dz.abs() {
                                    // Primary slope is in X direction
                                    if center_dx > 0 {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::West,
                                            StairShape::Straight,
                                        ) // Facing toward center
                                    } else {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::East,
                                            StairShape::Straight,
                                        ) // Facing toward center
                                    }
                                } else {
                                    // Primary slope is in Z direction
                                    if center_dz > 0 {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::North,
                                            StairShape::Straight,
                                        ) // Facing toward center
                                    } else {
                                        create_stair_with_properties(
                                            stair_block_material,
                                            StairFacing::South,
                                            StairShape::Straight,
                                        ) // Facing toward center
                                    }
                                };

                                editor.set_block_with_properties_absolute(
                                    stair_block,
                                    x,
                                    y + abs_terrain_offset,
                                    z,
                                    None,
                                    None,
                                );
                            } else {
                                // Use regular roof block where height doesn't change
                                editor.set_block_absolute(
                                    roof_block,
                                    x,
                                    y + abs_terrain_offset,
                                    z,
                                    None,
                                    None,
                                );
                            }
                        } else {
                            // Fill interior with solid blocks
                            editor.set_block_absolute(
                                roof_block,
                                x,
                                y + abs_terrain_offset,
                                z,
                                None,
                                None,
                            );
                        }
                    }
                }
            }
        }

        RoofType::Skillion => {
            // Skillion roof - single sloping surface
            let building_size = (max_x - min_x).max(max_z - min_z);

            // Scale roof height based on building size (4-10 blocks) unless explicitly specified
            let default_roof_height = (building_size / 3).clamp(4, 10);
            let roof_rise = roof_height_override.unwrap_or(default_roof_height).max(2);

            // Determine slope axis from roof:direction or fall back to dominant building axis
            let fallback_axis = if (max_x - min_x) >= (max_z - min_z) {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            };
            let slope_axis = roof_direction_vector.unwrap_or(fallback_axis);
            let stair_facing = facing_from_axis(slope_axis);

            // Precompute min/max projection along the slope axis
            let mut min_proj = f64::INFINITY;
            let mut max_proj = f64::NEG_INFINITY;
            for &(x, z) in floor_area {
                let proj = project_onto_axis(x, z, slope_axis);
                min_proj = min_proj.min(proj);
                max_proj = max_proj.max(proj);
            }
            let proj_range = (max_proj - min_proj).max(1.0);

            // First pass: calculate all roof heights
            let mut roof_heights = HashMap::new();
            for &(x, z) in floor_area {
                let proj = project_onto_axis(x, z, slope_axis);
                let slope_progress = ((proj - min_proj) / proj_range).clamp(0.0, 1.0);
                let roof_height = base_height + (slope_progress * roof_rise as f64) as i32;
                roof_heights.insert((x, z), roof_height);
            }

            // 50% accent block, otherwise wall block for roof
            let mut rng = rand::thread_rng();
            let roof_block = if rng.gen_bool(0.5) {
                accent_block
            } else {
                wall_block
            };

            // Second pass: place blocks with stairs only where height increases
            for &(x, z) in floor_area {
                let roof_height = roof_heights[&(x, z)];

                // Fill from base height to calculated roof height to create solid roof
                for y in base_height..=roof_height {
                    if y == roof_height {
                        // Check if this is a height transition point by looking at neighboring blocks
                        let has_lower_neighbor = [(x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)]
                            .iter()
                            .any(|(nx, nz)| {
                                roof_heights
                                    .get(&(*nx, *nz))
                                    .is_some_and(|&nh| nh < roof_height)
                            });

                        if has_lower_neighbor {
                            // Place stairs at height transitions for a stepped appearance
                            let stair_block_material = get_stair_block_for_material(roof_block);
                            let stair_block_with_props =
                                create_stair_with_properties(
                                    stair_block_material,
                                    stair_facing,
                                    StairShape::Straight,
                                );
                            editor.set_block_with_properties_absolute(
                                stair_block_with_props,
                                x,
                                y + abs_terrain_offset,
                                z,
                                None,
                                None,
                            );
                        } else {
                            // Use regular roof material where height doesn't change
                            editor.set_block_absolute(
                                roof_block,
                                x,
                                y + abs_terrain_offset,
                                z,
                                None,
                                None,
                            );
                        }
                    } else {
                        // Fill interior with solid blocks
                        editor.set_block_absolute(
                            roof_block,
                            x,
                            y + abs_terrain_offset,
                            z,
                            None,
                            None,
                        );
                    }
                }
            }
        }

        RoofType::Pyramidal => {
            // Pyramidal roof - all sides slope to a single central peak point
            let building_size = (max_x - min_x).max(max_z - min_z);

            // Calculate peak height based on building size (taller peak for larger buildings)
            let default_peak = (building_size / 3).clamp(3, 8);
            let roof_rise = roof_height_override.unwrap_or(default_peak).max(2);
            let peak_height = base_height + roof_rise;

            // 50% accent block, otherwise wall block for roof
            let mut rng = rand::thread_rng();
            let roof_block = if rng.gen_bool(0.5) {
                accent_block
            } else {
                wall_block
            };

            // First pass: calculate all roof heights
            let mut roof_heights = HashMap::new();
            for &(x, z) in floor_area {
                // Calculate distance from this point to the center
                let dx = (x - center_x).abs() as f64;
                let dz = (z - center_z).abs() as f64;

                // Use the maximum distance to either edge to determine slope
                // This creates the pyramid effect where all sides slope equally
                let distance_to_edge = dx.max(dz);

                // Calculate maximum distance from center to any edge
                let max_distance = ((max_x - min_x) / 2).max((max_z - min_z) / 2) as f64;

                // Calculate height based on distance from center
                // Points closer to center are higher, creating the pyramid slope
                let height_factor = if max_distance > 0.0 {
                    (1.0 - (distance_to_edge / max_distance)).max(0.0f64)
                } else {
                    1.0
                };

                let roof_height =
                    base_height + (height_factor * (peak_height - base_height) as f64) as i32;
                roof_heights.insert((x, z), roof_height);
            }

            // Second pass: place blocks with stairs at the surface
            for &(x, z) in floor_area {
                let roof_height = roof_heights[&(x, z)];

                // Fill from base height to calculated roof height to create solid pyramid
                for y in base_height..=roof_height {
                    if y == roof_height {
                        // Place stairs at the surface with correct facing direction
                        // Determine which direction the stairs should face based on the slope
                        let dx = x - center_x;
                        let dz = z - center_z;

                        // Check if there are higher neighbors to determine stair orientation
                        let north_height = roof_heights
                            .get(&(x, z - 1))
                            .copied()
                            .unwrap_or(base_height);
                        let south_height = roof_heights
                            .get(&(x, z + 1))
                            .copied()
                            .unwrap_or(base_height);
                        let west_height = roof_heights
                            .get(&(x - 1, z))
                            .copied()
                            .unwrap_or(base_height);
                        let east_height = roof_heights
                            .get(&(x + 1, z))
                            .copied()
                            .unwrap_or(base_height);

                        // Check for corner situations where two directions have lower neighbors
                        let has_lower_north = north_height < roof_height;
                        let has_lower_south = south_height < roof_height;
                        let has_lower_west = west_height < roof_height;
                        let has_lower_east = east_height < roof_height;

                        // Check for corner situations (two adjacent directions are lower)
                        let stair_block_material = get_stair_block_for_material(roof_block);
                        let stair_block = if has_lower_north && has_lower_west {
                            create_stair_with_properties(
                                stair_block_material,
                                StairFacing::East,
                                StairShape::OuterRight,
                            )
                        } else if has_lower_north && has_lower_east {
                            create_stair_with_properties(
                                stair_block_material,
                                StairFacing::South,
                                StairShape::OuterRight,
                            )
                        } else if has_lower_south && has_lower_west {
                            create_stair_with_properties(
                                stair_block_material,
                                StairFacing::East,
                                StairShape::OuterLeft,
                            )
                        } else if has_lower_south && has_lower_east {
                            create_stair_with_properties(
                                stair_block_material,
                                StairFacing::North,
                                StairShape::OuterLeft,
                            )
                        } else {
                            // Single direction
                            if dx.abs() > dz.abs() {
                                // Primary slope is in X direction
                                if dx > 0 && east_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::West,
                                        StairShape::Straight,
                                    ) // Facing west (stairs face toward center)
                                } else if dx < 0 && west_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::East,
                                        StairShape::Straight,
                                    ) // Facing east (stairs face toward center)
                                } else if dz > 0 && south_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::North,
                                        StairShape::Straight,
                                    ) // Facing north (stairs face toward center)
                                } else if dz < 0 && north_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::South,
                                        StairShape::Straight,
                                    ) // Facing south (stairs face toward center)
                                } else {
                                    BlockWithProperties::simple(roof_block) // Use regular block if no clear slope direction
                                }
                            } else {
                                // Primary slope is in Z direction
                                if dz > 0 && south_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::North,
                                        StairShape::Straight,
                                    ) // Facing north (stairs face toward center)
                                } else if dz < 0 && north_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::South,
                                        StairShape::Straight,
                                    ) // Facing south (stairs face toward center)
                                } else if dx > 0 && east_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::West,
                                        StairShape::Straight,
                                    ) // Facing west (stairs face toward center)
                                } else if dx < 0 && west_height < roof_height {
                                    create_stair_with_properties(
                                        stair_block_material,
                                        StairFacing::East,
                                        StairShape::Straight,
                                    ) // Facing east (stairs face toward center)
                                } else {
                                    BlockWithProperties::simple(roof_block) // Use regular block if no clear slope direction
                                }
                            }
                        };

                        editor.set_block_with_properties_absolute(
                            stair_block,
                            x,
                            y + abs_terrain_offset,
                            z,
                            None,
                            None,
                        );
                    } else {
                        // Fill interior with solid blocks
                        editor.set_block_absolute(
                            roof_block,
                            x,
                            y + abs_terrain_offset,
                            z,
                            None,
                            None,
                        );
                    }
                }
            }
        }

        RoofType::Dome => {
            // Dome roof - rounded hemispherical structure
            let radius = ((max_x - min_x).max(max_z - min_z).max(2) / 2) as f64;

            // 50% accent block, otherwise wall block for roof
            let mut rng = rand::thread_rng();
            let roof_block = if rng.gen_bool(0.5) {
                accent_block
            } else {
                wall_block
            };

            let default_height = (radius * 0.8).max(2.0);
            let dome_height = roof_height_override
                .map(|h| h.max(2) as f64)
                .unwrap_or(default_height);

            for &(x, z) in floor_area {
                let distance_from_center = ((x - center_x).pow(2) + (z - center_z).pow(2)) as f64;
                let normalized_distance = (distance_from_center.sqrt() / radius).min(1.0);

                // Use hemisphere equation to determine the height
                let height_factor = (1.0 - normalized_distance * normalized_distance).sqrt();
                let surface_height = base_height + (height_factor * dome_height) as i32;

                // Fill from the base to the surface
                for y in base_height..=surface_height {
                    editor.set_block_absolute(roof_block, x, y + abs_terrain_offset, z, None, None);
                }
            }
        }
    }
}

fn parse_roof_height(tags: &HashMap<String, String>, scale_factor: f64) -> Option<i32> {
    if let Some(value) = tags.get("roof:height") {
        if let Some(height_meters) = parse_height_to_f64(value) {
            let scaled = (height_meters * scale_factor).floor() as i32;
            if scaled > 0 {
                return Some(scaled.max(1));
            }
        }
    }

    if let Some(levels_value) = tags.get("roof:levels") {
        if let Ok(levels) = levels_value.trim().parse::<i32>() {
            if levels > 0 {
                return Some(multiply_scale(levels * 4, scale_factor).max(1));
            }
        }
    }

    None
}

fn parse_roof_direction(tags: &HashMap<String, String>) -> Option<f64> {
    let raw = tags
        .get("roof:direction")
        .or_else(|| tags.get("direction"))?
        .trim()
        .trim_end_matches('°')
        .to_string();

    if raw.is_empty() {
        return None;
    }

    if let Ok(angle) = raw.parse::<f64>() {
        if angle.is_finite() {
            let mut normalized = angle % 360.0;
            if normalized < 0.0 {
                normalized += 360.0;
            }
            return Some(normalized);
        }
    }

    let uppercase = raw
        .replace('-', "")
        .replace('_', "")
        .to_uppercase();
    match uppercase.as_str() {
        "N" | "NORTH" => Some(0.0),
        "NNE" => Some(22.5),
        "NE" | "NORTHEAST" => Some(45.0),
        "ENE" => Some(67.5),
        "E" | "EAST" => Some(90.0),
        "ESE" => Some(112.5),
        "SE" | "SOUTHEAST" => Some(135.0),
        "SSE" => Some(157.5),
        "S" | "SOUTH" => Some(180.0),
        "SSW" => Some(202.5),
        "SW" | "SOUTHWEST" => Some(225.0),
        "WSW" => Some(247.5),
        "W" | "WEST" => Some(270.0),
        "WNW" => Some(292.5),
        "NW" | "NORTHWEST" => Some(315.0),
        "NNW" => Some(337.5),
        _ => None,
    }
}

fn parse_roof_orientation_hint(tags: &HashMap<String, String>) -> Option<RoofOrientationHint> {
    tags.get("roof:orientation").and_then(|value| {
        match value.trim().to_lowercase().as_str() {
            "along" | "parallel" => Some(RoofOrientationHint::Along),
            "across" | "perpendicular" => Some(RoofOrientationHint::Across),
            _ => None,
        }
    })
}

fn direction_to_vector(angle_deg: f64) -> (f64, f64) {
    // Flip 180 degrees so ridge orientation matches in-game expectations
    let corrected_angle = (angle_deg + 180.0) % 360.0;
    // In OSM, 0 degrees points to geographic north. In Minecraft's coordinate system,
    // decreasing Z heads north. Map accordingly and normalize the vector for robustness.
    let radians = corrected_angle.to_radians();
    let dx = radians.sin();
    let dz = -radians.cos();
    let length = (dx * dx + dz * dz).sqrt().max(1e-6);
    (dx / length, dz / length)
}

fn project_onto_axis(x: i32, z: i32, axis: (f64, f64)) -> f64 {
    x as f64 * axis.0 + z as f64 * axis.1
}

fn facing_from_axis(axis: (f64, f64)) -> StairFacing {
    if axis.0.abs() >= axis.1.abs() {
        if axis.0 >= 0.0 {
            StairFacing::East
        } else {
            StairFacing::West
        }
    } else if axis.1 >= 0.0 {
        StairFacing::South
    } else {
        StairFacing::North
    }
}

pub fn generate_building_from_relation(
    editor: &WorldEditor,
    relation: &ProcessedRelation,
    args: &Args,
) {
    // Extract levels from relation tags
    let relation_levels = relation
        .tags
        .get("building:levels")
        .and_then(|l: &String| l.parse::<i32>().ok())
        .unwrap_or(2); // Default to 2 levels

    // Treat both explicit inner rings and building:part members as voids when
    // generating the base outline so courtyards stay hollow and detailed parts
    // do not get overwritten by the shell.
    let mut hole_members: Vec<HolePolygon> = relation
        .members
        .iter()
        .filter_map(|member| match member.role {
            ProcessedMemberRole::Inner => Some(HolePolygon {
                way: &member.way,
                add_walls: true,
            }),
            ProcessedMemberRole::Part => Some(HolePolygon {
                way: &member.way,
                add_walls: false,
            }),
            _ => None,
        })
        .collect();

    // Deduplicate to avoid redundant flood-fill passes when the same geometry
    // is referenced multiple times inside a relation.
    hole_members.sort_by_key(|entry| entry.way.id);
    hole_members.dedup_by(|a, b| {
        if a.way.id == b.way.id {
            b.add_walls |= a.add_walls;
            true
        } else {
            false
        }
    });

    let hole_polygons: Option<&[HolePolygon]> = if hole_members.is_empty() {
        None
    } else {
        Some(hole_members.as_slice())
    };

    // Process the outer way to create the building walls, carving out building:part footprints if needed
    for member in &relation.members {
        if member.role == ProcessedMemberRole::Outer {
            generate_buildings(
                editor,
                &member.way,
                args,
                Some(relation_levels),
                hole_polygons,
            );
        }
    }

    // Handle inner ways (holes, courtyards, etc.)
    /*for member in &relation.members {
        if member.role == ProcessedMemberRole::Inner {
            let polygon_coords: Vec<(i32, i32)> =
                member.way.nodes.iter().map(|n| (n.x, n.z)).collect();
            let hole_area: Vec<(i32, i32)> =
                flood_fill_area(&polygon_coords, args.timeout.as_ref());

            for (x, z) in hole_area {
                // Remove blocks in the inner area to create a hole
                editor.set_block(AIR, x, ground_level, z, None, Some(&[SPONGE]));
            }
        }
    }*/
}

/// Generates a bridge structure, paying attention to the "level" tag.
fn generate_bridge(
    editor: &WorldEditor,
    element: &ProcessedWay,
    floodfill_timeout: Option<&Duration>,
) {
    let floor_block: Block = STONE;
    let railing_block: Block = STONE_BRICKS;

    // Process the nodes to create bridge pathways and railings
    let mut previous_node: Option<(i32, i32)> = None;
    for node in &element.nodes {
        let x: i32 = node.x;
        let z: i32 = node.z;

        // Calculate bridge level based on the "level" tag
        let bridge_y_offset = if let Some(level_str) = element.tags.get("level") {
            if let Ok(level) = level_str.parse::<i32>() {
                (level * 3) + 1
            } else {
                1 // Default elevation
            }
        } else {
            1 // Default elevation
        };

        // Create bridge path using Bresenham's line
        if let Some(prev) = previous_node {
            let bridge_points: Vec<(i32, i32, i32)> =
                bresenham_line(prev.0, bridge_y_offset, prev.1, x, bridge_y_offset, z);

            for (bx, by, bz) in bridge_points {
                // Place railing blocks
                editor.set_block(railing_block, bx, by + 1, bz, None, None);
                editor.set_block(railing_block, bx, by, bz, None, None);
            }
        }

        previous_node = Some((x, z));
    }

    // Flood fill the area between the bridge path nodes
    let polygon_coords: Vec<(i32, i32)> = element.nodes.iter().map(|n| (n.x, n.z)).collect();

    let bridge_area: Vec<(i32, i32)> = flood_fill_area(&polygon_coords, floodfill_timeout);

    // Calculate bridge level based on the "level" tag
    let bridge_y_offset = if let Some(level_str) = element.tags.get("level") {
        if let Ok(level) = level_str.parse::<i32>() {
            (level * 3) + 1
        } else {
            1 // Default elevation
        }
    } else {
        1 // Default elevation
    };

    // Place floor blocks
    for (x, z) in bridge_area {
        editor.set_block(floor_block, x, bridge_y_offset, z, None, None);
    }
}
