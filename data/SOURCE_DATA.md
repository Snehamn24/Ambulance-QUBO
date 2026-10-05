# Public data provenance

Dataset: OpenStreetMap road geometry and tags near BMSCE, Bengaluru, India.

Source API: https://api.openstreetmap.org/api/0.6/map?bbox=77.560,12.936,77.570,12.946

Attribution: © OpenStreetMap contributors. License: Open Data Commons Open Database License (ODbL) 1.0. https://www.openstreetmap.org/copyright

The original XML is bundled as `bmsce_area.osm`; the processed graph contains 375 nodes and 1,024 directed arcs. The JSON metadata records processing time, the original XML SHA-256, bounding box, source and assumed speeds. It is a fixed snapshot; downloading later may change the roads and results.

Processing: keep supported drivable highway classes; omit explicit private/no access; interpret basic one-way/roundabout direction; calculate geodesic segment length; contract unambiguous degree-two chains while retaining geometry; retain the largest strongly connected component. Preserve distinct parallel arcs.

Not modeled: complete conditional/turn restrictions, every mode-specific access rule, lane capacity, emergency exemptions or map errors. Class-based speeds, signal delay and incident/traffic values are simulated. There are no patient records, ambulance GPS tracks or observed response times in this dataset. Node coordinates and road geometry are public; a route time is a model estimate, not measured emergency travel time.
