use nalgebra::Vector3;

// (1-sqrt(5))/2
const GOLDEN_RATIO: f32 = 1.6180339887498948482045868343656381177;
// This angle arises from treating considering a rectangle with side lengths 2 and φ
// where φ is the golden ratio. We take the midpoint of a side with length 2, and consider the
// angle from that point to the two opposite corners. The following then falls out of the law of
// cosines. Make sure to remember that φ^2 = φ+1
// acos(1-2/(φ+2))
const POLY_ANGLE: f32 = 1.10714871779409040897235172451473772525787353515625;

// Gives an estimate for the mean spherical distance between points at a given subdivision level.
// Recall that a great circle distance is equal to the angle subtended by the endpoints
// multiplied by radius. We assume the sphere has radius 1, so that the distance equals the angle.
// This approximation gets better the more subdivisions there are
pub fn distance_for_subdivision(subdivs: usize) -> f32 {
    // Cover the sphere (of radius 1) with circles of radius r/2 centered at each of the points of the polyhedron. These circles should
    // approximately cover the surface of the sphere, giving us the following relation:
    // 4π=(10n^2+2)(πr^2/4)
    // Solving for r gives us
    // 16/(10n^2+2)=r^2

    let n = subdivs as f32;
    (16. / (10. * n * n + 2.)).sqrt()
}

// Reverses the operation of `distance_for_subdivision`.
pub fn subdivisions_for_distance(dist: f32) -> usize {
    assert!(dist >= 0.);
    // 16/(10n^2+2)=r^2 so
    // 16/(10r^2)-1/5=n^2
    (16. / (10. * dist * dist) - 1. / 5.).sqrt().round() as usize
}

// Gives an estimate for the mean spherical distance between points at a given subdivision level.
// This estimate arises from the observation that all these points are approximately equidistant,
// and that segments along one of the original edges of the icosahedron have an easily computable length.
// This estimate seems to be generally slightly worse than the one given by `distance_for_subdivision`.
pub fn distance_for_subdivision_alt(subdivs: usize) -> f32 {
    POLY_ANGLE / (subdivs as f32)
}

// Reverses the operation of `distance_for_subdivision_alt`.
pub fn subdivisions_for_distance_alt(dist: f32) -> usize {
    assert!(dist >= 0.);
    (dist / POLY_ANGLE).round() as usize
}

// Constant data for a subdivision 1 mesh, i.e. an icosahedron or D20
const φ: f32 = GOLDEN_RATIO;
const initial_verts: [Vector3<f32>; 12] = [
    Vector3::new(0., φ, -1.),
    Vector3::new(-φ, 1., 0.),
    Vector3::new(-1., 0., -φ),
    Vector3::new(1., 0., -φ),
    Vector3::new(φ, 1., 0.),
    Vector3::new(0., φ, 1.),
    Vector3::new(-1., 0., φ),
    Vector3::new(-φ, -1., 0.),
    Vector3::new(0., -φ, -1.),
    Vector3::new(φ, -1., 0.),
    Vector3::new(1., 0., φ),
    Vector3::new(0., -φ, 1.),
];
// Triangles oriented in antiwiddershins(clockwise) order
const initial_tris: [(usize, usize, usize); 20] = [
    (0, 2, 1),
    (0, 3, 2),
    (0, 4, 3),
    (0, 5, 4),
    (0, 1, 5),
    (7, 6, 1),
    (8, 7, 2),
    (9, 8, 3),
    (10, 9, 4),
    (6, 10, 5),
    (2, 7, 1),
    (3, 8, 2),
    (4, 9, 3),
    (5, 10, 4),
    (1, 6, 5),
    (11, 6, 7),
    (11, 7, 8),
    (11, 8, 9),
    (11, 9, 10),
    (11, 10, 6),
];

// Barycentric interpolation on the triangle specified by tri
fn get_vert(tri: (usize, usize, usize), a: f32, b: f32) -> Vector3<f32> {
    let v1 = initial_verts[tri.0] - initial_verts[tri.1];
    let v2 = initial_verts[tri.2] - initial_verts[tri.1];

    initial_verts[tri.1] + (a * v1 + b * v2)
}

/*
Generates a geodesic polyhedron, created by subdividing each face of an icosahedron
into new faces based on the number of subdivisions. subdivs represents the number of
segments each edge is subdivided into so that, for example, subdivision 2 yields
```
 /\
/\/\
```
while subdivision 3 yields
```
  /\
 /\/\
/\/\/\
```
Returns the vertex data in an array of vectors, and the face data in an array of triangle faces
represented as a triple of indices into the vertex data. These indices are correctly ordered for
3d rendering. The vertices are unnormallized, hence still lie on the faces of the original icosahedron
*/
pub fn generate_polyhedron(subdivs: usize) -> (Vec<Vector3<f32>>, Vec<(usize, usize, usize)>) {
    assert!(subdivs >= 1);

    if subdivs == 1 {
        return (
            std::array::IntoIter::new(initial_verts).collect(),
            std::array::IntoIter::new(initial_tris).collect(),
        );
    }

    // Computes the output vertices and the uniqueified indices
    let (uniq, out_verts) = {
        let mut out_verts = Vec::new();
        out_verts.reserve(10 * subdivs * subdivs + 2);

        let mut int_verts = Vec::new();

        // To generate each vertex, take the 3 vertices of a face in order.
        // The vector from the 2nd vertex to the 1st and 2nd to 3rd form a
        // basis for a plane. They are at 60 degrees from each other.
        // For a subdivision n, take each i/n multiple of these basis vectors
        // where i ranges from 0 to n. This gets you a parallelogram.
        // Our desired triangle is half of this, so we just cut off
        // the indices at the correct point in the triangle.
        let n = (subdivs) as f32;
        for tri in initial_tris.iter() {
            for i in 0..subdivs + 1 {
                for j in 0..(subdivs + 1 - i) {
                    int_verts.push(get_vert(*tri, (i as f32) / n, (j as f32) / n));
                }
            }
        }

        let mut acc = Vec::new();
        acc.reserve(out_verts.capacity());

        // Deduplicate vertices and generate the map from original vertex data to deduplicated data
        // Deduplication is accomplished by a simple distance check. In the ideal world you would use
        // the perfect knowledge of the face adjacency and ordering to avoid doing this, but that
        // is absurdly obnoxious, so I went with the simple, functional solution.
        for vert in int_verts.iter() {
            if let Some(other) = out_verts
                .iter()
                .enumerate()
                .find(|other: &(usize, &Vector3<f32>)| (other.1 - vert).norm_squared() < 1e-10)
            {
                acc.push(other.0);
            } else {
                acc.push(out_verts.len());
                out_verts.push(vert.clone());
            }
        }

        (acc, out_verts)
    };

    // This function gets the deduplicated index of vertex (i, j) of face.
    let get_out_vert_ind = |face: usize, i: usize, j: usize| {
        // The number of verts in a face
        let num = ((subdivs + 1) * (subdivs + 2)) / 2;
        let base = face * num;

        // The number of verts in rows 1 through k of a face, given subdivision n
        // Computed as total number of verts - number of verts in a triangle with n-k verts.
        fn ind(n: usize, k: usize) -> usize {
            (2 * n * k + k - k * k) / 2
        }

        uniq[base + ind(subdivs + 1, i) + j]
    };

    let mut out_tris = Vec::new();
    for (ind, _) in initial_tris.iter().enumerate() {
        // Do the triangles with two vertices on the bottom
        for i in 0..subdivs {
            for j in 0..(subdivs - i) {
                out_tris.push((
                    get_out_vert_ind(ind, i, j),
                    get_out_vert_ind(ind, i + 1, j),
                    get_out_vert_ind(ind, i, j + 1),
                ));
            }
        }

        // Do the triangles with two vertices on the top
        for i in 1..subdivs {
            for j in 0..(subdivs - i) {
                out_tris.push((
                    get_out_vert_ind(ind, i, j),
                    get_out_vert_ind(ind, i, j + 1),
                    get_out_vert_ind(ind, i - 1, j + 1),
                ));
            }
        }
    }

    (out_verts, out_tris)
}

// Functions identically to `generate_polyhedron`, but only returns the vertices of the polyhedron.
pub fn generate_polyhedron_verts(subdivs: usize) -> Vec<Vector3<f32>> {
    assert!(subdivs >= 1);

    if subdivs == 1 {
        return std::array::IntoIter::new(initial_verts).collect();
    }

    let mut out_verts = Vec::new();
    out_verts.reserve(10 * subdivs * subdivs + 2);

    let mut int_verts = Vec::new();

    let n = (subdivs) as f32;
    for tri in initial_tris.iter() {
        for i in 0..subdivs + 1 {
            for j in 0..(subdivs + 1 - i) {
                int_verts.push(get_vert(*tri, (i as f32) / n, (j as f32) / n));
            }
        }
    }

    for vert in int_verts.iter() {
        if None
            == out_verts
                .iter()
                .cloned()
                .enumerate()
                .find(|other: &(usize, Vector3<f32>)| (other.1 - vert).norm_squared() < 1e-10)
        {
            out_verts.push(vert.clone());
        }
    }

    out_verts
}

#[cfg(test)]
mod test {
    use super::generate_polyhedron;

    #[test]
    fn geodesic_polyhedron_count() {
        let get_num = |x: usize| 10 * (x) * (x) + 2;
        let out = generate_polyhedron(3);
        assert_eq!(out.0.len(), get_num(3));
    }
}
