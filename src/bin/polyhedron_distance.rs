use radar_lib::geodesic_polyhedron::{self, generate_polyhedron};

fn main() {
    let (verts, tris) = generate_polyhedron(80);

    println!(
        "Prediction: {}",
        geodesic_polyhedron::distance_for_subdivision(80)
    );

    for tri in tris {
        println!(
            "{}\n{}\n{}",
            verts[tri.0].angle(&verts[tri.1]),
            verts[tri.0].angle(&verts[tri.2]),
            verts[tri.1].angle(&verts[tri.2]),
        );
    }
}
